from __future__ import annotations

import random

from skillmoo.operators import OperatorPolicy


def test_operator_policy_pick_family_returns_known_family() -> None:
    policy = OperatorPolicy()
    family = policy.pick_family(random.Random(7))
    assert family in {"pass", "cost", "length"}


def test_operator_policy_updates_weights_with_floor() -> None:
    policy = OperatorPolicy(alpha=0.5, floor=0.2)
    policy.update("pass", success=True)
    assert policy.pass_weight == 1.5
    policy.update("pass", success=False)
    assert policy.pass_weight == 0.75
    policy.update("pass", success=False)
    policy.update("pass", success=False)
    assert policy.pass_weight >= 0.2
