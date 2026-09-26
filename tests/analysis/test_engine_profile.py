"""Portable engine assumptions reject values that could falsify feasibility."""

from __future__ import annotations

import json

import pytest

from tilefoundry.analysis.engine_profile import (
    DeviceBudget,
    EngineDeployment,
    EngineOptions,
    EngineWorkload,
    NetworkRoute,
    RoutingProfile,
    engine_options_from_dict,
    load_engine_options,
)


def test_profile_round_trip_preserves_missing_facts_and_routing(tmp_path):
    expected = EngineOptions(
        EngineDeployment(
            (DeviceBudget(0, reserve_bytes=16), DeviceBudget(1, capacity_bytes=2048)),
            (NetworkRoute(0, 1, 1000, 5, ("fabric",)), NetworkRoute(1, 0)),
            "synthetic test deployment",
        ),
        EngineWorkload(8, "tokens", 1000, ("cache",)),
        (
            RoutingProfile(
                "AllToAllDispatch:0", ((1, 2), (0, 1)), ((2, 3), (0, 1)), ((2, 0), (3, 1))
            ),
        ),
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(expected.to_dict()))
    assert load_engine_options(path) == expected
    assert load_engine_options(path).deployment.routes[1].bandwidth_bytes_per_second is None


@pytest.mark.parametrize(
    "data, message",
    [
        ({"workload": {"work_items": True}}, "work_items"),
        ({"deployment": {"devices": [{"rank": 0, "capacity_bytes": -1}]}}, "capacity_bytes"),
        (
            {
                "deployment": {
                    "routes": [{"source": 0, "destination": 1, "bandwidth_bytes_per_second": 0}]
                }
            },
            "bandwidth",
        ),
        ({"deployment": {"routes": [{"source": 0, "destination": 0}]}}, "different ranks"),
        ({"deployment": {"devices": [{"rank": 0}, {"rank": 0}]}}, "duplicate"),
        ({"workload": {"state_inputs": "cache"}}, "array"),
        (
            {"deployment": {"routes": [{"source": 0, "destination": 1, "resources": "fabric"}]}},
            "array",
        ),
        ({"deployment": {"devices": [{"rank": 0, "capacity": 1024}]}}, "fields"),
        ({"workload": {"latency_budget_ns": float("nan")}}, "latency_budget_ns"),
        ({"deployment": {"routes": [{"source": 0}]}}, "invalid engine profile"),
    ],
)
def test_invalid_profile_is_refused(data, message):
    with pytest.raises(ValueError, match=message):
        engine_options_from_dict(data)


@pytest.mark.parametrize(
    "tokens, routes, experts, message",
    [
        (((2,),), ((1,),), ((1,),), "cannot exceed"),
        (((0,),), ((1,),), ((1,),), "empty transfers"),
        (((1,),), ((2,),), ((1,),), "expert_counts sum"),
        (((1, 0),), ((1,),), ((1,),), "rank-ordered"),
    ],
)
def test_routing_counts_cannot_silently_change_payload(tokens, routes, experts, message):
    with pytest.raises(ValueError, match=message):
        RoutingProfile("dispatch", tokens, routes, experts)
