"""Example labels describe requested depths rather than nonbinding caps."""

import json
from pathlib import Path

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_budget_grid_pins_branch_depth_and_enables_each_rerank_cap():
    spec = json.loads((EXAMPLES / "candidate-window-ablation.json").read_text())
    combinations = set()
    for arm in spec["arms"]:
        settings = arm["overlay"]
        depth = settings["retrieval.minimum_candidates"]
        cap = settings["retrieval.rerank_max_candidates"]
        assert depth == settings["retrieval.maximum_candidates"]
        assert (
            settings["retrieval.rerank_window_multiple"] * spec["harness"]["top_k"]
            >= cap
        )
        assert settings["retrieval.rerank_window_floor"] >= cap
        assert arm["name"] == f"depth-{depth}-rerank-{cap}"
        combinations.add((depth, cap))
    assert combinations == {(d, r) for d in (40, 80, 160) for r in (20, 30, 50)}


def test_examples_use_portable_inputs_and_no_implicit_target_skips():
    for path in EXAMPLES.glob("*.json"):
        spec = json.loads(path.read_text())
        assert not Path(spec["source_project"]).is_absolute()
        assert "skip_targets" not in spec["harness"]
        judged = spec["judgments"]
        if isinstance(judged, list):
            assert all(not Path(j["path"]).is_absolute() for j in judged)
            assert all(j["name"].startswith("exploratory-") for j in judged)
        else:
            assert not Path(judged).is_absolute()
