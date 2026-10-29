"""Versioned response contracts published through OpenAPI."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AssessmentInfo(BaseModel):
    status: Literal["urgent", "review", "unsupported", "retake", "research"]
    title: str
    summary: str
    next_steps: list[str]


class CaseContext(BaseModel):
    symptoms: list[str]
    duration: Literal["days", "weeks", "months", "unknown"]
    area: Literal["face", "body", "hands", "scalp", "other"]
    image_kind: Literal["clinical", "dermoscopy"]


class ImageQuality(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    issues: list[str]
    scope: str


class Candidate(BaseModel):
    code: str
    name: str
    score: float = Field(ge=0, le=1)


class EducationBlock(BaseModel):
    title: str
    body: str


class Reference(BaseModel):
    title: str
    url: HttpUrl


class ModelInfo(BaseModel):
    available: bool
    mode: str
    clinical_validation: Literal[False]
    class_count: int = Field(ge=0)
    message: str


class AnalysisResponse(BaseModel):
    id: str
    assessment: AssessmentInfo
    context: CaseContext
    quality: ImageQuality | None
    candidates: list[Candidate]
    education: list[EducationBlock]
    references: list[Reference]
    warnings: list[str]
    model: ModelInfo
    privacy: str
    elapsed_ms: int = Field(ge=0)
