from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class RunnerInfoOut(BaseModel):
    id: UUID
    name: str
    lastVideoId: Optional[UUID]


class AddRunnerIn(BaseModel):
    name: str


class RunSessionInfoOut(BaseModel):
    runSessionId: UUID
    runnerId: UUID
    runnerName: str
    date: datetime
    cameraCount: int
    fps: int
    avgVelocity: Optional[float] = None
    avgAcceleration: Optional[float] = None
    avgStepLength: Optional[float] = None
    totalTime: Optional[float] = None
    note: str
    status: str
    progress: int = 0
    isLongJump: bool = False
    computeLocations: List[str] = Field(default_factory=list)


class UnanalyzedRunSessionInfoOut(BaseModel):
    runSessionId: UUID
    runnerId: UUID
    runnerName: str
    date: datetime
    cameraCount: int
    fps: int
    note: str
    unuploadedCameraIndexes: List[int]
    videoPaths: List[Optional[str]]


class GraphDataOut(BaseModel):
    name: str
    y: list[float]


class GraphOut(BaseModel):
    title: str
    yLabel: str
    yMin: float
    yMax: float
    x: list[float]
    series: list[GraphDataOut]
    category: str = "metrics"


class AngleSampleOut(BaseModel):
    frame: int
    timeSec: float
    values: dict[str, Optional[float]]


class AnglesOut(BaseModel):
    columns: list[str]
    samples: list[AngleSampleOut]


class AnchorPointIn(BaseModel):
    x: float
    y: float
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None


class AnchorResultIn(BaseModel):
    points: list[AnchorPointIn]
    leftToMidDistanceM: Optional[float] = None
    midToRightDistanceM: Optional[float] = None
    topDistanceM: Optional[float] = None
    bottomDistanceM: Optional[float] = None

    @property
    def segmentedDistanceM(self) -> Optional[float]:
        if self.leftToMidDistanceM is None or self.midToRightDistanceM is None:
            return None
        return self.leftToMidDistanceM + self.midToRightDistanceM


class UploadVideoInfoIn(BaseModel):
    tempVideoId: str
    anchors: Optional[AnchorResultIn] = None


class UploadAllRequest(BaseModel):
    runnerId: str
    date: str
    fps: int
    cameraCount: int
    note: str
    videos: list[UploadVideoInfoIn]
    isLongJump: bool = False


class UploadSeperatelyStatus(BaseModel):
    runnerId: str
    runSessionId: str
    isAllUploaded: bool
    unuploadedCameraIndexes: List[int]


class UploadSeperatelyNewRequest(BaseModel):
    runnerId: str
    date: str
    fps: int
    cameraCount: int
    note: str
    cameraIndex: int
    tempVideoId: str
    anchors: Optional[AnchorResultIn] = None
    isLongJump: bool = False


class UploadSeperatelySelectRequest(BaseModel):
    runnerId: str
    runSessionId: str
    cameraIndex: int
    tempVideoId: str
    anchors: Optional[AnchorResultIn] = None


class StepSampleOut(BaseModel):
    stepIndex: int
    timeSec: float
    cam: int
    foot: Optional[str] = None
    eventType: Optional[str] = None
    stepLengthM: Optional[float] = None
    cadenceSpm: Optional[float] = None
    velocityMps: Optional[float] = None
    worldXM: Optional[float] = None
    worldYM: Optional[float] = None


class StepsOut(BaseModel):
    avgStepLengthM: Optional[float] = None
    avgCadenceSpm: Optional[float] = None
    steps: list[StepSampleOut]


class ToePointOut(BaseModel):
    x: float
    y: float
    worldXM: Optional[float] = None
    worldYM: Optional[float] = None
    score: float


class ToePathFrameOut(BaseModel):
    seqFrame: int
    origFrame: int
    timeSec: float
    points: dict[str, ToePointOut]


class ToePathOut(BaseModel):
    keypointNames: list[str]
    hasWorldCoords: bool
    frames: list[ToePathFrameOut]


class CreateLocalAnalysisRunIn(BaseModel):
    """Creates a new RunSession + its 'local' AnalysisRun (規劃書 Step 10)."""

    runnerId: UUID
    date: Optional[datetime] = None
    cameraCount: int = 1
    fps: int = 60
    isLongJump: bool = False
    note: str = ""
    comparisonGroupId: Optional[UUID] = None


class CreateLocalAnalysisRunOut(BaseModel):
    runSessionId: UUID
    analysisRunId: UUID


class AnalysisRunOut(BaseModel):
    analysisRunId: UUID
    runSessionId: UUID
    computeLocation: str
    comparisonGroupId: Optional[UUID] = None
    status: str
    schemaVersion: str
    engineVersion: str
    resultSummary: Optional[dict] = None
    createdAt: datetime
    completedAt: Optional[datetime] = None
