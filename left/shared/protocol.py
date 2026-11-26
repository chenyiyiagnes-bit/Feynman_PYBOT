from pydantic import BaseModel, Field


class RawWorkerDraft(BaseModel):
    bot_id: str
    wave_direction: str = "RETARDED_FORWARD"
    perspective: str
    raw_reasoning: str


class WorkerPathEvaluation(BaseModel):
    bot_id: str
    conclusion_key: str
    detail: str
    evidence_strength: float = Field(..., ge=0.1, le=1.0)
    assumption_burden: float = Field(..., ge=0.0, le=1.0)
    boundary_fragility: float = Field(..., ge=0.0, le=1.0)
    logical_leap: float = Field(..., ge=0.0, le=1.0)


class AmplitudePacket(BaseModel):
    bot_id: str
    wave_direction: str
    perspective: str
    conclusion_key: str
    detail: str
    critical_flaw: str
    real: float
    imag: float
    magnitude: float
    phase_rad: float
    phase_deg: float
    interference_regime: str


class SolveRequest(BaseModel):
    problem: str
    force_refresh: bool = False
