from __future__ import annotations

from skillmoo.operators import OperatorPolicy


def test_operator_weight_regression_sequence() -> None:
    policy = OperatorPolicy(alpha=0.2, floor=0.15)
    policy.update("pass", success=True)
    policy.update("pass", success=False)
    policy.update("cost", success=False)
    policy.update("length", success=True)
    assert policy.as_dict() == {
        "pass": 0.96,
        "cost": 0.8,
        "length": 1.2,
    }
