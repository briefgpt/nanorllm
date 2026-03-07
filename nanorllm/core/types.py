from dataclasses import dataclass, field
from typing import Any

@dataclass
class Action:
   value: Any

@dataclass
class RewardOutput:
   reward: float = 0.0
   metadata: dict[str, Any] = field(default_factory=dict)
   is_correct: bool | None = None