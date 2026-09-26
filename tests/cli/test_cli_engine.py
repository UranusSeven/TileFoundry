"""Engine profiles enter through analyze and retain the same text/JSON evidence."""

from __future__ import annotations

import json

import pytest

from tests.fixtures.distributed import engine
from tilefoundry import cli


def _profile(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text(
        json.dumps(
            {
                "deployment": {
                    "source": "synthetic CLI test",
                    "devices": [
                        {"rank": 0, "capacity_bytes": 600},
                        {"rank": 1, "capacity_bytes": 600},
                    ],
                    "routes": [
                        {
                            "source": 0,
                            "destination": 1,
                            "bandwidth_bytes_per_second": 1000000000,
                            "latency_ns": 10,
                        },
                        {
                            "source": 1,
                            "destination": 0,
                            "bandwidth_bytes_per_second": 1000000000,
                            "latency_ns": 10,
                        },
                    ],
                },
                "workload": {"work_items": 4, "latency_budget_ns": 2000, "state_inputs": ["state"]},
            }
        )
    )
    return path


def test_cli_engine_profile_prices_and_serializes_the_invocation(tmp_path, capsys):
    profile = _profile(tmp_path)
    output = tmp_path / "engine.json"
    args = [
        "analyze",
        f"{engine.__file__}:ShardedProjection",
        str(output),
        "--engine",
        "--engine-profile",
        str(profile),
    ]
    assert cli.main([*args, "--json"]) == 0
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text())
    found = report["function_records"]["engine"]
    assert found["predicted_ns"] == 1044
    assert found["feasible"] is True
    assert found["ranks"][0]["peak_hbm_bytes"] == 544
    assert found["options"]["deployment"]["source"] == "synthetic CLI test"
    assert cli.main(args) == 0
    text = output.read_text()
    assert "predicted_ns=1044" in text or "predicted-ns=1044" in text
    assert "544" in text and "engine" in text


def test_cli_engine_reports_unknown_links_without_a_profile(tmp_path):
    output = tmp_path / "unknown.json"
    assert (
        cli.main(
            ["analyze", f"{engine.__file__}:ShardedProjection", str(output), "--engine", "--json"]
        )
        == 0
    )
    found = json.loads(output.read_text())["function_records"]["engine"]
    assert found["predicted_ns"] is None and found["feasible"] is None
    assert any("missing network" in note for note in found["diagnostics"])


def test_cli_refuses_ignored_or_malformed_engine_profiles(tmp_path, capsys):
    profile = _profile(tmp_path)
    output = tmp_path / "report.json"
    args = [
        "analyze",
        f"{engine.__file__}:ShardedProjection",
        str(output),
        "--engine-profile",
        str(profile),
    ]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert "requires --engine" in capsys.readouterr().err
    profile.write_text('{"workload": {"latency_budget_ns": -1}}')
    assert cli.main([*args, "--engine"]) == 1
    assert not output.exists()
    assert "latency_budget_ns" in capsys.readouterr().err


def test_checked_strategy_selection_changes_with_the_memory_budget(tmp_path, capsys):
    profile = {
        "deployment": {
            "source": "synthetic comparison",
            "devices": [{"rank": rank, "reserve_bytes": 48000} for rank in range(4)],
            "routes": [
                {
                    "source": source,
                    "destination": destination,
                    "bandwidth_bytes_per_second": 1000000000,
                    "latency_ns": 10,
                }
                for source in range(4)
                for destination in range(4)
                if source != destination
            ],
        },
        "workload": {"work_items": 16, "latency_budget_ns": 45000},
    }
    path = tmp_path / "comparison.json"
    selected = []
    for reserve in (48000, 0):
        for device in profile["deployment"]["devices"]:
            device["reserve_bytes"] = reserve
        path.write_text(json.dumps(profile))
        findings = {}
        for name in ("DP4", "TP2", "TP4"):
            if reserve:
                assert (
                    cli.main(
                        [
                            "check",
                            f"{engine.__file__}:{name}",
                            "--reference",
                            f"{engine.__file__}:StrategyReference",
                            "--distributed",
                            "--dim",
                            "tokens=16",
                            "--inputs",
                            "random",
                            "--weights",
                            "random",
                            "--device",
                            "cpu",
                            "--out",
                            "output",
                            "--fn",
                            "allclose",
                            "--atol",
                            "1e-4",
                            "--rtol",
                            "1e-4",
                        ]
                    )
                    == 0
                )
            output = tmp_path / f"{name}.json"
            assert (
                cli.main(
                    [
                        "analyze",
                        f"{engine.__file__}:{name}",
                        str(output),
                        "--engine",
                        "--dim",
                        "tokens=16",
                        "--engine-profile",
                        str(path),
                        "--json",
                    ]
                )
                == 0
            )
            findings[name] = json.loads(output.read_text())["function_records"]["engine"]
        feasible = [name for name in findings if findings[name]["feasible"] is True]
        selected.append(max(feasible, key=lambda name: findings[name]["throughput_per_second"]))
        assert findings["TP2"]["predicted_ns"] == 38932
        assert findings["TP4"]["slo_met"] is False
    assert selected == ["TP2", "DP4"]
    assert capsys.readouterr().err == ""
