"""What the offline review page may contain, show, and refuse.

The page is the only surface an author reads, so its guarantees are about the file
itself rather than about behaviour a browser would have to be trusted to produce.
The claims tested here are that the page embeds exactly the validated records, that
nothing in a passage can become markup or reach the network, that the page's own
text names no engine, ordering, or result, and that an original link cannot be
pointed at anything but a file in the bundle.

The JavaScript completeness check is a second copy of the validator's rules and is
not executed by these tests. That is stated rather than hidden: what counts is the
toolkit's reading of the saved file.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag_experiments.annotation.review import render_review
from rag_experiments.annotation.schema import (
    RUBRIC,
    canonical_digest,
    make_template,
)
from rag_experiments.errors import ExperimentError

POOL_SHA = "a" * 64

_DATA_OPEN = '<script type="application/json" id="review-data">'

#: Text that would end the data block, start an element, or fetch something, if any
#: of it were embedded or built as markup rather than as a string.
INJECTION = "</script><img src=x onerror=alert(1)> & <b>bold</b>"

#: What the page itself must never carry. A scheme, a network call, a browser
#: storage, or a remote resource would each make a page opened by hand reach
#: somewhere.
NETWORK_TOKENS = (
    "http://",
    "https://",
    "//cdn",
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "EventSource",
    "sendBeacon",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "document.write",
    "eval(",
    "new Function",
    "import(",
    "importScripts",
    "@import",
    "url(",
    "src=",
)

#: Words that would tell a reader which engine, ordering, or result a passage came
#: from. `original` contains "origin", so whole words only.
BLINDING_WORDS = ("origin", "rank", "score", "arm", "engine", "chunk", "hybrid", "bm25")

#: Keys that would put an internal identity or a retrieval fact in front of the
#: reader. Checked against every key at every depth of the embedded records.
FORBIDDEN_KEYS = (
    "chunk_id",
    "target_id",
    "query_id",
    "document_id",
    "rank",
    "score",
    "scores",
    "origin",
    "arm",
    "engine",
    "retrieval_method",
    "reranked",
    "rerank_fallback",
    "selected",
    "generation_id",
    "run_path",
    "pool_builder",
)


def build_pool() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pool_id": "pool-01",
        "role": "exploratory",
        "rubric": copy.deepcopy(RUBRIC),
        "questions": [
            {"question_id": "q-1", "query": "what did the trial measure"},
            {"question_id": "q-2", "query": "who signed the consent form"},
        ],
        "items": [
            {
                "item_id": "it-a",
                "question_id": "q-1",
                "passage": "the trial measured resting heart rate in 40 adults",
                "source": {
                    "title": "A Trial of Things",
                    "authors": "R. Author",
                    "year": "2019",
                    "doi": "10.1000/xyz",
                    "source_relative_path": "sources/trial.pdf",
                    "locator": {"page": 12, "section": "Methods"},
                    "original": "originals/one.pdf",
                },
            },
            {
                "item_id": "it-b",
                "question_id": "q-1",
                "passage": "the trial measured resting heart rate in 40 adults",
                "source": {
                    "locator": {"page": 4},
                    "original": "originals/two.pdf",
                },
            },
            {
                "item_id": "it-c",
                "question_id": "q-2",
                "passage": "consent was obtained in writing",
                "source": {
                    "locator": {"page": 7},
                    "original": "originals/three.pdf",
                },
            },
        ],
        "pairs": [
            {
                "pair_id": "pr-1",
                "question_id": "q-1",
                "left_item_id": "it-a",
                "right_item_id": "it-b",
            }
        ],
    }


def rendered(html: str) -> tuple[str, str, dict[str, Any]]:
    """The page's own text, the raw data block, and that block parsed."""

    start = html.index(_DATA_OPEN) + len(_DATA_OPEN)
    end = html.index("</script>", start)
    static = html[:start] + html[end:]
    return static, html[start:end], json.loads(html[start:end])


def walk_keys(value: Any) -> set[str]:
    """Every key at every depth of a record."""

    keys: set[str] = set()
    if isinstance(value, dict):
        for key, entry in value.items():
            keys.add(key)
            keys |= walk_keys(entry)
    elif isinstance(value, list):
        for entry in value:
            keys |= walk_keys(entry)
    return keys


def render(pool: dict[str, Any] | None = None) -> str:
    return render_review(
        pool if pool is not None else build_pool(), pool_sha256=POOL_SHA
    )


# The page is built from the validated records, and only those.


def test_the_page_embeds_exactly_the_validated_pool() -> None:
    pool = build_pool()
    _, _, data = rendered(render(pool))

    assert data["pool"] == pool


def test_the_page_embeds_the_pending_template_for_that_pool() -> None:
    pool = build_pool()
    _, _, data = rendered(render(pool))

    assert data["template"] == make_template(pool, pool_sha256=POOL_SHA)
    assert data["template"]["rubric_acknowledged"] is False
    assert data["template"]["pool_sha256"] == POOL_SHA
    assert data["template"]["rubric_sha256"] == canonical_digest(pool["rubric"])
    for row in data["template"]["items"]:
        assert row["relevance"] is None
        assert row["usability"] is None
        assert row["source_verified"] is None


def test_no_substitution_token_is_left_behind() -> None:
    assert "__REVIEW_DATA__" not in render()


def test_the_page_is_a_single_html_document() -> None:
    html = render()
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<script") == 2


def test_a_pool_that_is_not_this_contract_never_reaches_a_page() -> None:
    """The page renders a validated pool, so a broken one is refused first."""

    pool = build_pool()
    pool["items"][0]["source"]["original"] = "javascript:alert(1)"
    with pytest.raises(ExperimentError):
        render(pool)


def test_no_path_from_this_machine_reaches_the_page(tmp_path: Path) -> None:
    """The page carries the pool's relative links and nothing else."""

    probe = tmp_path / "runs" / "probe-20261003"
    probe.mkdir(parents=True)
    assert str(probe) not in render()
    assert str(tmp_path) not in render()


# Injected text is text.


def test_injected_passage_text_cannot_close_the_data_block() -> None:
    pool = build_pool()
    pool["items"][0]["passage"] = INJECTION
    pool["items"][0]["source"]["title"] = INJECTION
    pool["questions"][0]["query"] = INJECTION
    static, block, data = rendered(render(pool))

    assert "<" not in block
    assert "</script" not in block
    assert "<img" not in block
    assert "\\u003c" in block
    assert data["pool"]["items"][0]["passage"] == INJECTION
    assert data["pool"]["questions"][0]["query"] == INJECTION
    assert static.count("<script") == 2


def test_injected_text_is_never_assigned_as_markup() -> None:
    """A passage is set as text, so an element in it cannot be created."""

    static, _, _ = rendered(render())

    assert "innerHTML" not in static
    assert "insertAdjacentHTML" not in static
    assert "outerHTML" not in static
    assert "document.write" not in static
    assert "textContent" in static


def test_the_line_separators_javascript_reads_as_line_ends_are_escaped() -> None:
    pool = build_pool()
    pool["items"][0]["passage"] = "before after end"
    _, block, data = rendered(render(pool))

    assert "\\u2028" in block
    assert "\\u2029" in block
    assert data["pool"]["items"][0]["passage"] == "before after end"


def test_an_ampersand_and_a_bracket_are_escaped_so_no_tag_can_be_assembled() -> None:
    pool = build_pool()
    pool["items"][0]["passage"] = "a & b < c > d"
    _, block, _ = rendered(render(pool))

    assert "\\u0026" in block
    assert "\\u003c" in block
    assert "\\u003e" in block
    assert "<" not in block and "&" not in block


# The page reaches nothing and stores nothing.


@pytest.mark.parametrize("token", NETWORK_TOKENS)
def test_the_page_carries_no_network_or_storage_call(token: str) -> None:
    static, _, _ = rendered(render())

    assert token not in static


def test_the_page_names_no_engine_ordering_or_result() -> None:
    static, _, _ = rendered(render())

    for word in BLINDING_WORDS:
        assert re.search(rf"\b{word}s?\b", static, re.IGNORECASE) is None, word


def test_the_embedded_records_hold_no_internal_identity() -> None:
    _, _, data = rendered(render())
    keys = walk_keys(data)

    for key in FORBIDDEN_KEYS:
        assert key not in keys, key
    # The blinded identities are the ones the pool carries, and nothing else.
    assert {"pool_id", "question_id", "item_id", "pair_id"} <= keys


def test_the_page_offers_no_remote_origins_to_link_to() -> None:
    pool = build_pool()
    pool["items"][0]["source"]["original"] = "https://example.invalid/one.pdf"
    with pytest.raises(ExperimentError, match="scheme"):
        render(pool)


# The extraction warning is stated on the page, not only in the record.


def test_the_page_says_the_text_is_an_extraction_and_not_a_transcript() -> None:
    static, block, data = rendered(render())
    notice = data["pool"]["rubric"]["extraction_notice"]

    assert "not a transcript" in notice
    assert "only" in notice and "authority" in notice
    # The notice is data the page sets as text, so it reaches the reader as prose.
    assert notice in block
    assert 'byId("extraction").textContent' in static
    assert "extracted" in static


def test_the_page_repeats_that_only_the_original_is_the_authority() -> None:
    static, _, _ = rendered(render())

    assert "Only the original says what the source says" in static


def test_the_rubric_is_shown_with_its_definitions_before_anything_is_marked() -> None:
    """Every grade the reader can pick is defined on the page, from the pool's rubric."""

    static, _, data = rendered(render())
    rubric = data["pool"]["rubric"]

    assert "rubric-dimensions" in static
    for name in ("relevance", "usability", "relation", "source_verified"):
        choices = rubric["dimensions"][name]["choices"]
        assert choices
        for choice in choices:
            assert choice["definition"]
            assert choice["label"]
    # The page builds the rubric section from the pool's own copy rather than a
    # second list written beside it, so a pool that narrows its rubric shows its own.
    assert "RUBRIC.instructions.forEach" in static
    assert "RUBRIC.rules.forEach" in static
    assert "text(choice.definition)" in static
    assert "dimension.choices.forEach" in static


def test_a_finished_annotation_is_stated_not_to_be_a_quality_result() -> None:
    static, _, _ = rendered(render())

    assert "not a quality result" in static
    assert "not a basis for a policy decision" in static


def test_the_acknowledgement_gates_the_handover_and_saving_stays_available() -> None:
    static, _, _ = rendered(render())

    assert 'id="ack"' in static
    assert 'id="save-complete" type="button" disabled' in static
    assert 'id="save-partial" type="button"' in static
    assert "rubric_acknowledged" in static


def test_the_first_choice_of_every_grade_is_pending() -> None:
    static, _, _ = rendered(render())

    assert 'option("", "Pending")' in static
    assert 'id="save-complete"' in static


# An original is opened by hand from the bundle, and the link cannot be anything else.


def test_the_link_is_set_from_a_value_the_page_rechecks() -> None:
    static, _, _ = rendered(render())

    assert "function safeOriginal(value)" in static
    assert "link.href = safe" in static
    assert 'link.rel = "noopener noreferrer"' in static


@pytest.mark.parametrize(
    "value",
    [
        "javascript:alert(1)",
        "https://example.invalid/a.pdf",
        "/etc/passwd",
        "../a.pdf",
        "originals/../a.pdf",
        'originals/a".pdf',
        "originals\\a.pdf",
    ],
)
def test_the_page_refuses_an_original_that_is_not_a_bundle_file(value: str) -> None:
    pool = build_pool()
    pool["items"][0]["source"]["original"] = value
    with pytest.raises(ExperimentError):
        render(pool)


def test_the_page_says_when_no_original_is_linked() -> None:
    static, _, _ = rendered(render())

    assert "No original file is linked here." in static


# Repeated passages are rendered like any other, because the builder hid the grouping.


def test_a_repeated_passage_is_not_marked_as_one() -> None:
    pool = build_pool()
    assert pool["items"][0]["passage"] == pool["items"][1]["passage"]
    static, _, data = rendered(render(pool))

    repeats = [
        item
        for item in data["pool"]["items"]
        if item["passage"] == pool["items"][0]["passage"]
    ]
    assert len(repeats) == 2
    assert {item["item_id"] for item in repeats} == {"it-a", "it-b"}
    for word in ("duplicate", "repeated", "already shown", "occurrence", "same as"):
        assert re.search(word, static, re.IGNORECASE) is None, word


def test_one_card_is_built_per_item_and_one_per_pair() -> None:
    static, _, _ = rendered(render())

    assert "POOL.items.forEach(function (item)" in static
    assert "POOL.pairs.forEach(function (pair)" in static


# Saving and reopening is local, and the file that comes back is checked again.


def test_a_saved_file_is_built_in_the_browser_and_downloaded() -> None:
    static, _, _ = rendered(render())

    assert "new Blob([exportText()]" in static
    assert "URL.createObjectURL" in static
    assert "URL.revokeObjectURL" in static
    assert 'annotations-" + POOL.pool_id + ".json' in static


def test_the_saved_file_carries_the_identity_and_digests_it_was_read_from() -> None:
    static, _, _ = rendered(render())

    for key in ("pool_id", "pool_sha256", "rubric_sha256", "rubric_acknowledged"):
        assert f"{key}:" in static


def test_reopening_a_file_is_a_read_of_a_chosen_file_and_not_a_fetch() -> None:
    static, _, _ = rendered(render())

    assert '<input type="file" id="import"' in static
    assert "file.text()" in static
    assert "Nothing on this page was changed." in static


def test_the_page_rechecks_a_reopened_file_against_the_same_rules() -> None:
    static, _, _ = rendered(render())

    assert 'typeof document_.rubric_acknowledged !== "boolean"' in static
    for reason in (
        "which this page does not read",
        "this pool does not hold",
        "which this page does not offer",
        "not checked in the original",
        "with no note",
    ):
        assert reason in static, reason


def test_a_source_claim_needs_a_note_and_two_checked_originals_in_the_page() -> None:
    static, _, _ = rendered(render())

    assert '"contradiction"' in static
    assert '"independent_corroboration"' in static
    assert "row.relation" in static


def test_the_page_states_it_stores_nothing_and_loses_unsaved_work() -> None:
    static, _, _ = rendered(render())

    assert "This page stores nothing" in static
    assert "reads nothing from the network" in static
    assert "beforeunload" in static


# The page is a template with one substitution, so it is the same page every time.


def test_two_renders_of_one_pool_differ_only_in_the_embedded_records() -> None:
    pool = build_pool()
    first = render_review(pool, pool_sha256=POOL_SHA)
    second = render_review(copy.deepcopy(pool), pool_sha256=POOL_SHA)

    assert first == second


def test_a_digest_the_caller_did_not_compute_is_refused() -> None:
    with pytest.raises(ExperimentError, match="not a sha256 digest"):
        render_review(build_pool(), pool_sha256="not-a-digest")
