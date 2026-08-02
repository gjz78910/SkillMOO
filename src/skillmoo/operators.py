from __future__ import annotations

from dataclasses import dataclass
import random

FAMILIES = ("pass", "cost", "length")


@dataclass
class OperatorPolicy:
    pass_weight: float = 1.0
    cost_weight: float = 1.0
    length_weight: float = 1.0
    alpha: float = 0.2
    floor: float = 0.15

    def pick_family(self, rng: random.Random) -> str:
        weights = self._weights()
        return rng.choices(FAMILIES, weights=weights, k=1)[0]

    def update(self, winning_family: str, success: bool) -> None:
        if winning_family not in FAMILIES:
            return
        if success:
            self._apply_gain(winning_family)
            return
        self._apply_decay(winning_family)

    def as_dict(self) -> dict[str, float]:
        return {
            "pass": round(self.pass_weight, 6),
            "cost": round(self.cost_weight, 6),
            "length": round(self.length_weight, 6),
        }

    def _weights(self) -> tuple[float, float, float]:
        return (self.pass_weight, self.cost_weight, self.length_weight)

    def _apply_gain(self, family: str) -> None:
        self._set_weight(family, self._weight_of(family) * (1.0 + self.alpha))

    def _apply_decay(self, family: str) -> None:
        self._set_weight(family, self._weight_of(family) * (1.0 - self.alpha))

    def _weight_of(self, family: str) -> float:
        if family == "pass":
            return self.pass_weight
        if family == "cost":
            return self.cost_weight
        return self.length_weight

    def _set_weight(self, family: str, value: float) -> None:
        bounded = max(self.floor, value)
        if family == "pass":
            self.pass_weight = bounded
            return
        if family == "cost":
            self.cost_weight = bounded
            return
        self.length_weight = bounded
