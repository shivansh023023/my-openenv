"""
DataPipeline-Sentry: Core Environment Logic
=============================================
Simulates a Financial Data Lake that an AI Data Reliability Engineer must
manage.  Three tasks with deterministic data generation (seed=42) and
partial-reward grading.

Tasks
-----
1. **schema_drift** (Easy)   — Upstream API changed field names.
2. **late_data**    (Medium) — Trade records arriving out of order.
3. **poisoned_records** (Hard) — A data source is injecting corrupt data.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.models import (
    Action,
    ApplyMappingPatchAction,
    BufferLog,
    BufferStatus,
    InspectSchemaAction,
    LatencyMetrics,
    Observation,
    QuarantineSourceAction,
    SchemaHealth,
    StatisticalTracebackAction,
    SubmitAction,
    WatermarkReprocessAction,
)

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
NUM_TRADE_ROWS = 500        # medium task
NUM_BASELINE_ROWS = 10_000  # hard task
NUM_SOURCES = 5
POISONED_SOURCE = "source_3"
POISONED_SHIFT_SIGMA = 3


class DataPipelineSentryEnv:
    """OpenEnv-compliant environment for Financial Data Lake Reliability."""

    # ── public interface ─────────────────────────────────────────────────

    def __init__(self) -> None:
        self._task: str = "schema_drift"
        self._step_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0

        # internal state buckets
        self._schema_health = SchemaHealth()
        self._latency = LatencyMetrics()
        self._buffer_logs: List[BufferLog] = []
        self._data_snapshot: str = ""
        self._error_log: str = ""

        # task-specific data
        self._pipeline_df: Optional[pd.DataFrame] = None
        self._gold_mapping: Dict[str, str] = {}
        self._schema_inspected: bool = False
        self._mapping_applied: bool = False
        self._traceback_done: bool = False
        self._quarantined_source: Optional[str] = None
        self._watermark_applied: bool = False
        self._original_df: Optional[pd.DataFrame] = None

    # ── RESET ────────────────────────────────────────────────────────────

    def reset(self, task: str = "schema_drift") -> Observation:
        """Initialise a fresh episode for the given task."""
        self.__init__()
        self._task = task

        if task == "schema_drift":
            self._init_schema_drift()
        elif task == "late_data":
            self._init_late_data()
        elif task == "poisoned_records":
            self._init_poisoned_records()
        else:
            self._task = "schema_drift"
            self._error_log = f"Unknown task '{task}'. Defaulting to schema_drift."
            self._init_schema_drift()

        return self._observation()

    # ── STEP ─────────────────────────────────────────────────────────────

    def step(self, action: Action) -> tuple:
        """Execute *action* and return (Observation, reward, done)."""
        if self._done:
            return self._observation(), self._reward, True

        self._error_log = ""
        self._step_count += 1

        try:
            if isinstance(action, InspectSchemaAction):
                self._handle_inspect_schema()
            elif isinstance(action, ApplyMappingPatchAction):
                self._handle_apply_mapping_patch(action)
            elif isinstance(action, WatermarkReprocessAction):
                self._handle_watermark_reprocess(action)
            elif isinstance(action, StatisticalTracebackAction):
                self._handle_statistical_traceback()
            elif isinstance(action, QuarantineSourceAction):
                self._handle_quarantine_source(action)
            elif isinstance(action, SubmitAction):
                self._handle_submit()
            else:
                self._error_log = f"Unrecognised action type: {type(action).__name__}"
        except Exception as exc:  # noqa: BLE001
            self._error_log = f"Action failed: {exc}"

        return self._observation(), self._reward, self._done

    # ── STATE / GRADE ────────────────────────────────────────────────────

    def state(self) -> Dict[str, str]:
        """Return serialised internal state for the /state endpoint."""
        return {
            "task": self._task,
            "step_count": str(self._step_count),
            "done": str(self._done),
            "reward": str(self._reward),
            "data_preview": (
                self._pipeline_df.head(5).to_csv(index=False)
                if self._pipeline_df is not None
                else ""
            ),
        }

    def grade(self) -> float:
        """Return current reward as the grading score (0.0 – 1.0)."""
        return round(min(max(self._reward, 0.0), 1.0), 4)

    # ── TASK INITIALISERS ────────────────────────────────────────────────

    def _init_schema_drift(self) -> None:
        """Easy: upstream API renamed fields."""
        rng = np.random.default_rng(SEED)
        n = 100
        # Expected (canonical) schema
        expected = ["trade_id", "trade_price", "trade_volume", "timestamp"]
        # Drifted schema from "new API version"
        drifted = ["trade_id", "px", "vol", "timestamp", "_metadata"]
        self._gold_mapping = {"px": "trade_price", "vol": "trade_volume"}

        df = pd.DataFrame({
            "trade_id": np.arange(1, n + 1),
            "px": rng.uniform(100, 500, n).round(2),
            "vol": rng.integers(100, 10_000, n),
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="min").astype(str).tolist(),
            "_metadata": [json.dumps({"version": "2.1"}) for _ in range(n)],
        })
        self._pipeline_df = df
        self._schema_health = SchemaHealth(
            expected_fields=expected,
            actual_fields=drifted,
            drift_detected=True,
            drift_details="Fields 'trade_price'→'px', 'trade_volume'→'vol' renamed; new field '_metadata' added.",
        )
        self._latency = LatencyMetrics(ingestion_lag_ms=12.4, processing_time_ms=3.1, queue_depth=0)
        self._data_snapshot = (
            f"Pipeline data: {df.shape[0]} rows, columns={list(df.columns)}\n"
            f"Head:\n{df.head(3).to_string(index=False)}"
        )
        self._buffer_logs = [
            BufferLog(timestamp="2025-01-01T00:00:00Z", source_id="upstream_api_v2",
                      record_count=n, status=BufferStatus.pending),
        ]

    def _init_late_data(self) -> None:
        """Medium: out-of-order trade records."""
        rng = np.random.default_rng(SEED)
        n = NUM_TRADE_ROWS
        base_times = pd.date_range("2025-06-01 09:30", periods=n, freq="s")
        event_times = base_times.copy()

        # 15% of records arrive late (shifted forward by 5-60 s)
        late_mask = rng.random(n) < 0.15
        delays_ms = rng.integers(5_000, 60_000, n)

        arrival_times = []
        for i in range(n):
            if late_mask[i]:
                arrival_times.append(event_times[i] + pd.Timedelta(milliseconds=int(delays_ms[i])))
            else:
                arrival_times.append(event_times[i] + pd.Timedelta(milliseconds=int(rng.integers(50, 500))))

        prices = rng.uniform(100, 500, n).round(2)

        df = pd.DataFrame({
            "trade_id": np.arange(1, n + 1),
            "event_time": event_times.astype(str).tolist(),
            "arrival_time": [str(t) for t in arrival_times],
            "trade_price": prices,
            "source_id": [f"exchange_{rng.integers(1, 4)}" for _ in range(n)],
        })
        # Shuffle to simulate out-of-order arrival
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        self._pipeline_df = df
        self._original_df = df.copy()

        n_late = int(late_mask.sum())
        self._latency = LatencyMetrics(
            ingestion_lag_ms=float(delays_ms[late_mask].mean()) if n_late > 0 else 0.0,
            processing_time_ms=8.7,
            queue_depth=n_late,
        )
        self._schema_health = SchemaHealth(
            expected_fields=["trade_id", "event_time", "arrival_time", "trade_price", "source_id"],
            actual_fields=list(df.columns),
            drift_detected=False,
        )
        self._data_snapshot = (
            f"Pipeline data: {df.shape[0]} rows (shuffled arrival order)\n"
            f"Late records detected: ~{n_late}\n"
            f"Head:\n{df.head(3).to_string(index=False)}"
        )
        self._buffer_logs = [
            BufferLog(timestamp="2025-06-01T09:30:00Z", source_id="exchange_1",
                      record_count=n // 3, status=BufferStatus.pending),
            BufferLog(timestamp="2025-06-01T09:30:00Z", source_id="exchange_2",
                      record_count=n // 3, status=BufferStatus.pending),
            BufferLog(timestamp="2025-06-01T09:30:00Z", source_id="exchange_3",
                      record_count=n - 2 * (n // 3), status=BufferStatus.pending),
        ]

    def _init_poisoned_records(self) -> None:
        """Hard: one of five sources is injecting shifted data."""
        rng = np.random.default_rng(SEED)
        rows_per_source = NUM_BASELINE_ROWS // NUM_SOURCES

        frames = []
        for src_idx in range(1, NUM_SOURCES + 1):
            src_id = f"source_{src_idx}"
            if src_id == POISONED_SOURCE:
                # Shift mean by POISONED_SHIFT_SIGMA * global_std
                prices = rng.normal(150 + POISONED_SHIFT_SIGMA * 10, 10, rows_per_source).round(2)
            else:
                prices = rng.normal(150, 10, rows_per_source).round(2)

            chunk = pd.DataFrame({
                "record_id": np.arange(
                    (src_idx - 1) * rows_per_source + 1,
                    src_idx * rows_per_source + 1,
                ),
                "source_id": src_id,
                "trade_price": prices,
                "timestamp": pd.date_range(
                    "2025-09-01", periods=rows_per_source, freq="s"
                ).astype(str).tolist(),
            })
            frames.append(chunk)

        df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
        self._pipeline_df = df
        self._original_df = df.copy()

        self._latency = LatencyMetrics(ingestion_lag_ms=5.2, processing_time_ms=22.0, queue_depth=0)
        self._schema_health = SchemaHealth(
            expected_fields=["record_id", "source_id", "trade_price", "timestamp"],
            actual_fields=list(df.columns),
            drift_detected=False,
        )
        self._data_snapshot = (
            f"Baseline data: {df.shape[0]} rows from {NUM_SOURCES} sources\n"
            f"Columns: {list(df.columns)}\n"
            f"Global mean trade_price: {df['trade_price'].mean():.2f}\n"
            f"Head:\n{df.head(3).to_string(index=False)}"
        )
        self._buffer_logs = [
            BufferLog(
                timestamp="2025-09-01T00:00:00Z",
                source_id=f"source_{i}",
                record_count=rows_per_source,
                status=BufferStatus.pending,
            )
            for i in range(1, NUM_SOURCES + 1)
        ]

    # ── ACTION HANDLERS ──────────────────────────────────────────────────

    def _handle_inspect_schema(self) -> None:
        """Return schema health details and award partial reward."""
        if self._task == "schema_drift":
            if not self._schema_inspected:
                self._schema_inspected = True
                self._reward = 0.2
            self._data_snapshot = (
                f"Schema Inspection Report\n"
                f"Expected : {self._schema_health.expected_fields}\n"
                f"Actual   : {self._schema_health.actual_fields}\n"
                f"Drift    : {self._schema_health.drift_detected}\n"
                f"Details  : {self._schema_health.drift_details}"
            )
        elif self._task == "late_data":
            if self._pipeline_df is not None:
                self._reward = max(self._reward, 0.1)
                df = self._pipeline_df
                self._data_snapshot = (
                    f"Late-Data Inspection\n"
                    f"Total rows: {len(df)}\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Event time range: {df['event_time'].min()} → {df['event_time'].max()}\n"
                    f"Arrival order matches event order: "
                    f"{(df['event_time'] == df.sort_values('event_time')['event_time'].values).all()}"
                )
        else:
            self._data_snapshot = (
                f"Schema OK — no drift detected.\n"
                f"Columns: {list(self._pipeline_df.columns) if self._pipeline_df is not None else '[]'}"
            )

    def _handle_apply_mapping_patch(self, action: ApplyMappingPatchAction) -> None:
        if self._task != "schema_drift":
            self._error_log = "apply_mapping_patch is only valid for the schema_drift task."
            return
        if self._pipeline_df is None:
            self._error_log = "No pipeline data loaded."
            return

        mapping = action.field_mapping
        # Validate mapping against gold answer
        correct = all(
            mapping.get(k) == v for k, v in self._gold_mapping.items()
        ) and len(mapping) >= len(self._gold_mapping)

        # Apply rename
        self._pipeline_df = self._pipeline_df.rename(columns=mapping)
        # Drop _metadata if present
        if "_metadata" in self._pipeline_df.columns:
            self._pipeline_df = self._pipeline_df.drop(columns=["_metadata"])

        self._mapping_applied = True
        self._schema_health.actual_fields = list(self._pipeline_df.columns)
        self._schema_health.drift_detected = not correct
        self._schema_health.drift_details = "" if correct else "Mapping incomplete or incorrect."

        if correct:
            self._reward = 0.95
            self._data_snapshot = (
                f"Mapping applied successfully. Schema aligned.\n"
                f"Columns: {list(self._pipeline_df.columns)}\n"
                f"Head:\n{self._pipeline_df.head(3).to_string(index=False)}"
            )
        else:
            self._reward = max(self._reward, 0.4)
            self._data_snapshot = (
                f"Mapping applied but mismatch remains.\n"
                f"Current columns: {list(self._pipeline_df.columns)}\n"
                f"Expected: {self._schema_health.expected_fields}"
            )

    def _handle_watermark_reprocess(self, action: WatermarkReprocessAction) -> None:
        if self._task != "late_data":
            self._error_log = "watermark_reprocess is only valid for the late_data task."
            return
        if self._pipeline_df is None:
            self._error_log = "No pipeline data loaded."
            return

        threshold_ms = action.watermark_threshold_ms
        df = self._pipeline_df.copy()

        # Convert times
        df["event_time"] = pd.to_datetime(df["event_time"])
        df["arrival_time"] = pd.to_datetime(df["arrival_time"])
        df["lateness_ms"] = (
            (df["arrival_time"] - df["event_time"]).dt.total_seconds() * 1000
        ).astype(int)

        # Drop records later than the watermark threshold
        before = len(df)
        df = df[df["lateness_ms"] <= threshold_ms].copy()
        dropped = before - len(df)

        # Re-sort by event_time
        df = df.sort_values("event_time").reset_index(drop=True)

        # Recalculate 20-period rolling average
        df["rolling_avg_price"] = df["trade_price"].rolling(window=20, min_periods=1).mean().round(4)

        # Convert times back to str for serialisation
        df["event_time"] = df["event_time"].astype(str)
        df["arrival_time"] = df["arrival_time"].astype(str)

        self._pipeline_df = df
        self._watermark_applied = True

        # Grade: good threshold is 5000–60000 ms
        is_sorted = df["event_time"].is_monotonic_increasing
        good_threshold = 5_000 <= threshold_ms <= 60_000

        if is_sorted and good_threshold:
            self._reward = 0.95
        elif is_sorted:
            self._reward = max(self._reward, 0.5)
        else:
            self._reward = max(self._reward, 0.3)

        # Update buffer logs
        for bl in self._buffer_logs:
            bl.status = BufferStatus.processed

        self._data_snapshot = (
            f"Watermark reprocess complete (threshold={threshold_ms}ms).\n"
            f"Dropped {dropped} excessively late records. Remaining: {len(df)}.\n"
            f"Sorted by event_time: {is_sorted}\n"
            f"Rolling average recalculated.\n"
            f"Head:\n{df.head(5).to_string(index=False)}"
        )

    def _handle_statistical_traceback(self) -> None:
        if self._task != "poisoned_records":
            self._error_log = "statistical_traceback is only valid for the poisoned_records task."
            return
        if self._pipeline_df is None:
            self._error_log = "No pipeline data loaded."
            return

        df = self._pipeline_df
        stats_lines = ["Per-Source Statistical Report", "=" * 40]
        global_mean = df["trade_price"].mean()
        global_std = df["trade_price"].std()

        # Use median of per-source means for robust anomaly detection
        source_means = {
            src: df[df["source_id"] == src]["trade_price"].mean()
            for src in sorted(df["source_id"].unique())
        }
        median_of_means = float(np.median(list(source_means.values())))
        # MAD-based robust std
        deviations = [abs(m - median_of_means) for m in source_means.values()]
        mad = float(np.median(deviations)) if deviations else 1.0
        robust_std = mad * 1.4826  # scale MAD to std-equivalent

        for src in sorted(df["source_id"].unique()):
            subset = df[df["source_id"] == src]["trade_price"]
            src_mean = subset.mean()
            src_std = subset.std()
            z_score = abs(src_mean - median_of_means) / robust_std if robust_std > 0 else 0
            flag = " ⚠ ANOMALOUS" if z_score > 2 else ""
            stats_lines.append(
                f"{src}: mean={src_mean:.2f}, std={src_std:.2f}, "
                f"z_score={z_score:.2f}, n={len(subset)}{flag}"
            )

        stats_lines.append(f"\nGlobal: mean={global_mean:.2f}, std={global_std:.2f}")
        stats_lines.append(f"Robust baseline (median of means): {median_of_means:.2f}")
        self._data_snapshot = "\n".join(stats_lines)
        self._traceback_done = True
        self._reward = max(self._reward, 0.2)

    def _handle_quarantine_source(self, action: QuarantineSourceAction) -> None:
        if self._task != "poisoned_records":
            self._error_log = "quarantine_source is only valid for the poisoned_records task."
            return
        if self._pipeline_df is None:
            self._error_log = "No pipeline data loaded."
            return

        src = action.source_id
        self._quarantined_source = src

        # Update buffer logs
        for bl in self._buffer_logs:
            if bl.source_id == src:
                bl.status = BufferStatus.quarantined

        # Remove quarantined source
        before = len(self._pipeline_df)
        self._pipeline_df = self._pipeline_df[
            self._pipeline_df["source_id"] != src
        ].reset_index(drop=True)
        after = len(self._pipeline_df)

        # Recalculate baseline stats
        new_mean = self._pipeline_df["trade_price"].mean()
        new_std = self._pipeline_df["trade_price"].std()

        correct_source = src == POISONED_SOURCE
        if correct_source:
            self._reward = 0.95 if self._traceback_done else 0.6
        else:
            self._reward = max(self._reward, 0.2)

        self._data_snapshot = (
            f"Source '{src}' quarantined. Removed {before - after} records.\n"
            f"Remaining: {after} records.\n"
            f"Correct source identified: {correct_source}\n"
            f"Recalculated baseline — mean={new_mean:.2f}, std={new_std:.2f}\n"
            f"Head:\n{self._pipeline_df.head(5).to_string(index=False)}"
        )

    def _handle_submit(self) -> None:
        """Finalise the episode."""
        self._done = True
        # If reward hasn't been set through correct actions, give minimal
        if self._reward < 0.1:
            self._reward = 0.01
        self._data_snapshot = (
            f"Episode submitted. Final score: {self._reward:.4f}\n"
            f"Task: {self._task} | Steps: {self._step_count}"
        )

    # ── OBSERVATION BUILDER ──────────────────────────────────────────────

    def _observation(self) -> Observation:
        actions_for_task = {
            "schema_drift": ["inspect_schema", "apply_mapping_patch", "submit"],
            "late_data": ["inspect_schema", "watermark_reprocess", "submit"],
            "poisoned_records": [
                "inspect_schema",
                "statistical_traceback",
                "quarantine_source",
                "submit",
            ],
        }
        return Observation(
            task_name=self._task,
            step_count=self._step_count,
            data_snapshot=self._data_snapshot,
            error_log=self._error_log,
            latency_metrics=self._latency,
            schema_health=self._schema_health,
            buffer_logs=self._buffer_logs,
            available_actions=actions_for_task.get(self._task, ["submit"]),
            done=self._done,
        )
