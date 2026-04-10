"""
DataPipeline-Sentry: Pydantic Models
=====================================
Rich observation and action models for the Financial Data Lake Reliability
environment. Designed for OpenEnv Phase 2 compliance with structured
telemetry, schema health tracking, and buffer management.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Telemetry & Health Sub-models
# ---------------------------------------------------------------------------

class BufferStatus(str, Enum):
    """Status of a data buffer entry in the ingestion pipeline."""
    pending = "pending"
    processed = "processed"
    quarantined = "quarantined"


class LatencyMetrics(BaseModel):
    """Real-time latency telemetry for the data pipeline."""
    ingestion_lag_ms: float = Field(
        0.0, description="End-to-end lag from source to lake in milliseconds"
    )
    processing_time_ms: float = Field(
        0.0, description="Time spent in the current processing step"
    )
    queue_depth: int = Field(
        0, description="Number of records waiting in the ingestion queue"
    )


class SchemaHealth(BaseModel):
    """Schema comparison report between expected and actual data formats."""
    expected_fields: List[str] = Field(
        default_factory=list,
        description="Canonical field names the pipeline expects",
    )
    actual_fields: List[str] = Field(
        default_factory=list,
        description="Field names currently present in the upstream payload",
    )
    drift_detected: bool = Field(
        False, description="True when expected ≠ actual"
    )
    drift_details: str = Field(
        "", description="Human-readable summary of the detected drift"
    )


class BufferLog(BaseModel):
    """Single entry in the pipeline's ingestion buffer."""
    timestamp: str = Field(
        ..., description="ISO-8601 timestamp of the buffer event"
    )
    source_id: str = Field(
        ..., description="Identifier of the data source"
    )
    record_count: int = Field(
        0, description="Number of records in this buffer batch"
    )
    status: BufferStatus = Field(
        BufferStatus.pending,
        description="Current lifecycle status of this buffer batch",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observation returned to the agent after every reset / step.
    Contains pipeline telemetry, schema health, buffer logs, and a
    human-readable data snapshot.
    """
    task_name: str = Field(
        ..., description="Active task identifier"
    )
    step_count: int = Field(
        0, description="Number of steps taken so far"
    )
    data_snapshot: str = Field(
        "", description="Stringified preview of current pipeline data"
    )
    error_log: str = Field(
        "", description="Most recent error or diagnostic message"
    )
    latency_metrics: LatencyMetrics = Field(
        default_factory=LatencyMetrics,
        description="Current pipeline latency telemetry",
    )
    schema_health: SchemaHealth = Field(
        default_factory=SchemaHealth,
        description="Schema comparison report",
    )
    buffer_logs: List[BufferLog] = Field(
        default_factory=list,
        description="Recent ingestion buffer activity",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Actions the agent can take in the current state",
    )
    done: bool = Field(
        False, description="True when the episode has ended"
    )


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class InspectSchemaAction(BaseModel):
    """Inspect the current vs expected schema of the pipeline data."""
    action: Literal["inspect_schema"] = "inspect_schema"


class ApplyMappingPatchAction(BaseModel):
    """Apply a field-name mapping to fix schema drift."""
    action: Literal["apply_mapping_patch"] = "apply_mapping_patch"
    field_mapping: Dict[str, str] = Field(
        ...,
        description="Mapping of actual→expected field names, e.g. {'px': 'trade_price'}",
    )


class WatermarkReprocessAction(BaseModel):
    """Re-sort late-arriving records using an event-time watermark."""
    action: Literal["watermark_reprocess"] = "watermark_reprocess"
    watermark_threshold_ms: int = Field(
        ...,
        description="Maximum allowed lateness in milliseconds before a record is dropped",
    )


class StatisticalTracebackAction(BaseModel):
    """Run per-source statistical analysis to detect anomalous sources."""
    action: Literal["statistical_traceback"] = "statistical_traceback"


class QuarantineSourceAction(BaseModel):
    """Quarantine a specific data source identified as poisoned."""
    action: Literal["quarantine_source"] = "quarantine_source"
    source_id: str = Field(
        ..., description="Identifier of the source to quarantine (e.g. 'source_3')"
    )


class SubmitAction(BaseModel):
    """Finalise the episode and submit the current pipeline state for grading."""
    action: Literal["submit"] = "submit"


# Discriminated union of all possible actions
Action = Union[
    InspectSchemaAction,
    ApplyMappingPatchAction,
    WatermarkReprocessAction,
    StatisticalTracebackAction,
    QuarantineSourceAction,
    SubmitAction,
]


# ---------------------------------------------------------------------------
# Request / Response Wrappers
# ---------------------------------------------------------------------------

class ActionRequest(BaseModel):
    """Wrapper sent by the agent on POST /step."""
    action: Action = Field(..., discriminator="action")


class ResetRequest(BaseModel):
    """Wrapper sent on POST /reset."""
    task: Optional[str] = Field(
        "schema_drift",
        description="Task to initialise: schema_drift | late_data | poisoned_records",
    )


class StepResponse(BaseModel):
    """Payload returned from POST /step."""
    observation: Observation
    reward: float
    done: bool
