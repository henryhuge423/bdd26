"""Result aggregation must not turn missing runs or repeated seeds into stronger evidence."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/p0p3_numbers.py"


def module():
    assert SCRIPT.exists(), "artifact-only report aggregator missing"
    spec = importlib.util.spec_from_file_location("p0p3_numbers", SCRIPT)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_mean_requires_all_three_splits():
    m = module()
    with pytest.raises(ValueError):
        m.mean_complete({1: {"mPQ": .5}, 2: {"mPQ": .6}})
    assert m.mean_complete({1: {"mPQ": .3}, 2: {"mPQ": .6}, 3: {"mPQ": .9}})["mPQ"] == pytest.approx(.6)


def test_repeated_seed_is_two_runs_but_one_distinct_seed():
    m = module()
    rows = [{"family": "x2", "split": 2, "seed": 1, "run": "split2_seed1"},
            {"family": "x2", "split": 2, "seed": 1, "run": "split2_seed2"}]
    s = m.seed_coverage(rows)["x2"]["2"]
    assert s["runs"] == 2 and s["distinct_seeds"] == [1]
    assert s["duplicate_seeds"] == [1]


def test_incomplete_experiments_are_not_reported_complete(tmp_path):
    with pytest.raises(FileNotFoundError):
        module().collect(tmp_path)


def test_mean_rejects_nonfinite_metrics_and_inconsistent_columns():
    m = module()
    with pytest.raises(ValueError):
        m.mean_complete({1: {"a": .1}, 2: {"a": float("nan")}, 3: {"a": .3}})
    with pytest.raises(ValueError):
        m.mean_complete({1: {"a": .1}, 2: {"b": .2}, 3: {"a": .3}})


def test_seed_identity_comes_from_config_not_directory(tmp_path):
    import json
    run = tmp_path / "split2_seed2"
    run.mkdir()
    assert module().seed_from_run(run) is None
    (run / "config.json").write_text(json.dumps({"seed": 1}))
    assert module().seed_from_run(run) == 1


def test_class_counts_distinguish_objects_and_images(tmp_path):
    import gzip
    p = tmp_path / "gt.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("image,cls\n0,4\n0,4\n1,1\n2,4\n")
    assert module().class_counts(p) == {"gt_nuclei": 4, "dead_nuclei": 3, "dead_images": 2}


def test_p2_report_binds_selection_bytes_and_executed_parameters(tmp_path):
    import json
    m = module()
    p = tmp_path / "selection.json"
    selected = {"name": "identity", "max_area": None, "min_prob": 0., "interior_only": False}
    choice = {"split": 1, "fold": 2, "role": "validation selection", "selected": selected}
    p.write_text(json.dumps(choice))
    manifest = {"split": 1, "fold": 3, "role": "frozen test; identity fallback if validation NO_GO",
                "selection_sha256": m.sha256(p), "selected": selected.copy()}
    assert m.validate_selection_link(p, manifest, 1, "p2") == choice
    choice["selected"]["name"] = "not-executed"
    p.write_text(json.dumps(choice))
    with pytest.raises(ValueError):
        m.validate_selection_link(p, manifest, 1, "p2")
    manifest["selection_sha256"] = m.sha256(p)
    with pytest.raises(ValueError):
        m.validate_selection_link(p, manifest, 1, "p2")


def test_p0_report_binds_parameters_and_fold(tmp_path):
    import json
    m = module()
    p = tmp_path / "selection.json"
    choice = {"split": 1, "validation_fold": 2, "code": {"files": {}},
              "arms": {"base": {"actual_seed": 19, "selected": {"frac": .25, "a_min": 40}}}}
    p.write_text(json.dumps(choice))
    manifest = {"split": 1, "fold": 3, "role": "test", "arm": "base", "variant": "ms",
                "actual_seed": 19, "code": {"files": {}}, "selection_sha256": m.sha256(p),
                "parameters": {"frac": .25, "a_min": 40}}
    m.validate_selection_link(p, manifest, 1, "p0")
    manifest["parameters"]["a_min"] = 20
    with pytest.raises(ValueError):
        m.validate_selection_link(p, manifest, 1, "p0")
    manifest["parameters"]["a_min"] = 40
    manifest["fold"] = 1
    with pytest.raises(ValueError):
        m.validate_selection_link(p, manifest, 1, "p0")


def bootstrap_fixture(root):
    return {
        "fold": 3, "reference": "base_ms", "boot": 2000,
        "groups": {
            "base_ms": {"runs": [str(root / "p0/split1/test/base/ms/eval")],
                        "mean": {"mPQ": .5, "bPQ": .66, "Dead": .1},
                        "per_run": [[.5, .66, 0., 0., 0., .1, 0.]]},
            "fusion": {"runs": [str(root / "p2/split1/test/eval")],
                       "mean": {"mPQ": .5001, "bPQ": .659, "Dead": .11},
                       "per_run": [[.5001, .659, 0., 0., 0., .11, 0.]],
                       "delta": {"mPQ": .0001, "bPQ": -.001, "Dead": .01},
                       "ci95": {"mPQ": [-.001, .001], "bPQ": [-.002, .001], "Dead": [0., .02]}}
        }
    }


@pytest.mark.parametrize("mutation", ["fold", "reference", "base_path", "fusion_path", "mean", "per_run", "delta", "ci"])
def test_bootstrap_is_bound_to_split_reference_and_point_estimates(tmp_path, mutation):
    m = module()
    data = bootstrap_fixture(tmp_path)
    base = {"mPQ": .5, "bPQ": .66, "DeadPQ": .1}
    fusion = {"mPQ": .5001, "bPQ": .659, "DeadPQ": .11}
    m.validate_bootstrap(data, tmp_path, 1, base, fusion)
    if mutation == "fold":
        data["fold"] = 1
    elif mutation == "reference":
        data["reference"] = "other"
    elif mutation == "base_path":
        data["groups"]["base_ms"]["runs"] = [str(tmp_path / "p0/split2/test/base/ms/eval")]
    elif mutation == "fusion_path":
        data["groups"]["fusion"]["runs"] = [str(tmp_path / "p2/split2/test/eval")]
    elif mutation == "mean":
        data["groups"]["fusion"]["mean"]["Dead"] = .2
    elif mutation == "per_run":
        data["groups"]["fusion"]["per_run"][0][5] = .2
    elif mutation == "delta":
        data["groups"]["fusion"]["delta"]["Dead"] = .1
    else:
        data["groups"]["fusion"]["ci95"]["Dead"] = [float("nan"), .02]
    with pytest.raises(ValueError):
        m.validate_bootstrap(data, tmp_path, 1, base, fusion)
