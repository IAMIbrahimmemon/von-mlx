"""Wire-protocol types, mirroring the von SDK exactly.

Response payloads are byte-compatible with the TypeSafe ``/v1/systemone``
envelope so an MLX-backed server is a drop-in replacement.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Noul(BaseModel):
    """A yes/no probability question."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: Optional[Dict[str, str]] = None


class Choice(BaseModel):
    """Pick one option from a fixed list, with a probability distribution."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, Optional[str]]


class Score(BaseModel):
    """A position on an ordered scale."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: List[Union[str, Dict[str, Any]]]


Question = Union[Noul, Choice, Score]


def noul(instructions: str, criteria: Optional[Dict[str, str]] = None) -> Noul:
    return Noul(instructions=instructions, criteria=criteria)


def choice(instructions: str, criteria: Dict[str, Optional[str]]) -> Choice:
    return Choice(instructions=instructions, criteria=criteria)


def score(instructions: str, criteria: List[Union[str, Dict[str, Any]]]) -> Score:
    return Score(instructions=instructions, criteria=criteria)


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["noul"] = "noul"
    noul: float = Field(..., description="Probability the condition is true")


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    answers: Dict[str, Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]]
    usage: Usage

    def __getitem__(self, item: str) -> Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]:
        return self.answers[item]
