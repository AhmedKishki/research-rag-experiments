"""What the blinded pool and the annotation are allowed to say, and what they refuse.

These tests are adversarial on purpose. A validator that accepts a malformed record
and a validator that invents a grade are both ways of producing a number nobody
attributed to a reader, so every case here is a record that must be refused, and
every pending case is one that must be reported as pending rather than decided.

Nothing here grades a passage. The pools are synthetic text with opaque
identifiers, which is the same shape a builder mints, and the assertions are about
the contract rather than about any passage.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from rag_experiments.annotation.schema import (
    RUBRIC,
    canonical_digest,
    make_template,
    validate_annotations,
    validate_pool,
)
from rag_experiments.errors import ExperimentError

#: A digest of bytes nobody here computed, standing in for the one the caller owns.
POOL_SHA = "a" * 64
OTHER_SHA = "b" * 64

#: Text that would end the page's data block if it were embedded unescaped. The pool
#: is data an author wrote nowhere, so it is still treated as hostile.
INJECTION = "</script><img src=x onerror=alert(1)> & <b>bold</b>"


def build_pool() -> dict[str, Any]:
    """Two questions, three passages, one pair: the smallest pool with a pair in it."""

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
                    "title": "A Trial",
                    "authors": "R. Author; B. Author",
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
                    "locator": {"page": 7, "figures": [1, 2]},
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


def build_annotated(
    pool: dict[str, Any] | None = None, **overrides: Any
) -> dict[str, Any]:
    """A pool with every grade made, the state a finished review is in."""

    target = pool if pool is not None else build_pool()
    annotations = make_template(target, pool_sha256=POOL_SHA, annotator="reader")
    annotations["rubric_acknowledged"] = True
    for row in annotations["items"]:
        row.update(relevance="direct", usability="usable", source_verified=True)
    annotations["pairs"][0]["relation"] = "copy"
    annotations.update(overrides)
    return annotations


def pool_with(**overrides: Any) -> dict[str, Any]:
    """The fixture pool with top-level keys replaced, for the refusal cases."""

    pool = build_pool()
    pool.update(overrides)
    return pool


# The template and the pool contract.


def test_a_valid_pool_is_accepted() -> None:
    validate_pool(build_pool())


def test_a_template_lists_every_row_and_leaves_every_grade_pending() -> None:
    pool = build_pool()
    template = make_template(pool, pool_sha256=POOL_SHA)

    assert template["pool_id"] == "pool-01"
    assert template["pool_sha256"] == POOL_SHA
    assert template["rubric_sha256"] == canonical_digest(pool["rubric"])
    assert template["annotator"] == ""
    assert template["rubric_acknowledged"] is False
    assert [row["item_id"] for row in template["items"]] == ["it-a", "it-b", "it-c"]
    assert [row["pair_id"] for row in template["pairs"]] == ["pr-1"]
    for row in template["items"]:
        assert row["relevance"] is None
        assert row["usability"] is None
        # Pending is null, and null is not False: an unchecked original is a fact
        # about the reading and not a finding that the passage is wrong.
        assert row["source_verified"] is None
        assert row["notes"] == ""
    assert template["pairs"][0]["relation"] is None
    assert template["pairs"][0]["notes"] == ""


def test_the_template_is_json_serializable() -> None:
    """A record is read by other tools, so it has to survive a round trip."""

    pool = build_pool()
    template = make_template(pool, pool_sha256=POOL_SHA)
    assert json.loads(json.dumps(template)) == template


def test_the_rubric_carries_definitions_and_assigns_nothing() -> None:
    """The rubric says what a grade means, and names no passage."""

    for name in ("relevance", "usability", "relation", "source_verified"):
        dimension = RUBRIC["dimensions"][name]
        assert dimension["question"]
        for choice in dimension["choices"]:
            assert choice["definition"].strip()
            assert choice["label"].strip()
    assert "uncertain" in {
        c["value"] for c in RUBRIC["dimensions"]["relevance"]["choices"]
    }
    assert RUBRIC["dimensions"]["source_verified"]["uncertain_value"] is None
    serialized = json.dumps(RUBRIC)
    for passage in ("it-a", "it-b", "pr-1", "q-1"):
        assert passage not in serialized


def test_canonical_digest_ignores_serialization() -> None:
    left = {"b": 1, "a": [1, 2, {"d": True, "c": None}]}
    right = json.loads('{"a": [1, 2, {"c": null, "d": true}], "b": 1}')
    assert canonical_digest(left) == canonical_digest(right)
    assert canonical_digest(left) != canonical_digest({"a": [1, 2], "b": 1})
    assert len(canonical_digest(left)) == 64


# Pool refusals: keys that would unblind the pool, and text a reader cannot use.


@pytest.mark.parametrize("leaked", ["origin", "rank", "score", "target_id", "arm"])
def test_a_key_that_would_unblind_the_pool_is_refused(leaked: str) -> None:
    """An unexpected key is refused rather than dropped, so a leak cannot ride in."""

    pool = build_pool()
    pool[leaked] = "hybrid+rerank"
    with pytest.raises(ExperimentError, match="does not define"):
        validate_pool(pool)


@pytest.mark.parametrize("leaked", ["chunk_id", "query_id", "retrieval_method"])
def test_a_key_on_an_item_that_would_unblind_it_is_refused(leaked: str) -> None:
    pool = build_pool()
    pool["items"][0][leaked] = "chunk-0007"
    with pytest.raises(ExperimentError, match="does not define"):
        validate_pool(pool)


@pytest.mark.parametrize("leaked", ["rerank_fallback", "score", "arm"])
def test_a_key_in_a_source_that_would_unblind_it_is_refused(leaked: str) -> None:
    pool = build_pool()
    pool["items"][0]["source"][leaked] = "bm25"
    with pytest.raises(ExperimentError, match="does not define"):
        validate_pool(pool)


def test_a_doi_is_a_bibliographic_fact_and_is_kept() -> None:
    validate_pool(build_pool())


@pytest.mark.parametrize(
    "identifier", ["it a", "it/a", "it.a", "chunk-0007=x", "../../etc"]
)
def test_an_identifier_that_is_not_an_opaque_token_is_refused(identifier: str) -> None:
    pool = build_pool()
    pool["items"][0]["item_id"] = identifier
    with pytest.raises(ExperimentError, match="not an opaque token"):
        validate_pool(pool)


def test_an_identifier_long_enough_to_carry_a_path_is_refused() -> None:
    pool = build_pool()
    pool["items"][0]["item_id"] = "a" * 65
    with pytest.raises(ExperimentError, match="opaque token"):
        validate_pool(pool)


def test_an_injected_identifier_is_refused_before_it_can_reach_a_page() -> None:
    pool = build_pool()
    pool["items"][0]["item_id"] = 'it-a" onerror="alert(1)'
    with pytest.raises(ExperimentError, match="not an opaque token"):
        validate_pool(pool)


@pytest.mark.parametrize(
    "original",
    [
        "javascript:alert(1)",
        "https://example.invalid/original.pdf",
        "file:///etc/passwd",
        "/etc/passwd",
        "../../elsewhere/original.pdf",
        "originals/../../escape.pdf",
        'originals/one".pdf',
        "originals\\one.pdf",
    ],
)
def test_an_original_that_leaves_the_bundle_is_refused(original: str) -> None:
    """An original is opened by hand; it is never fetched and never escapes."""

    pool = build_pool()
    pool["items"][0]["source"]["original"] = original
    with pytest.raises(ExperimentError):
        validate_pool(pool)


def test_a_locator_that_is_not_a_position_is_refused() -> None:
    pool = build_pool()
    pool["items"][0]["source"]["locator"] = {"pages": {"start": 3}}
    with pytest.raises(ExperimentError, match="locator"):
        validate_pool(pool)


def test_an_empty_locator_says_nothing_about_where_a_passage_came_from() -> None:
    pool = build_pool()
    pool["items"][0]["source"]["locator"] = {}
    with pytest.raises(ExperimentError, match="empty locator"):
        validate_pool(pool)


def test_a_pool_without_a_role_cannot_be_read_as_a_measurement() -> None:
    with pytest.raises(ExperimentError, match="states no role"):
        validate_pool(pool_with(role="  "))


def test_a_pool_with_no_passages_is_refused() -> None:
    with pytest.raises(ExperimentError, match="no passages"):
        validate_pool(pool_with(items=[]))


def test_a_passage_naming_a_question_the_pool_does_not_hold_is_refused() -> None:
    pool = build_pool()
    pool["items"][0]["question_id"] = "q-9"
    with pytest.raises(ExperimentError, match="does not hold"):
        validate_pool(pool)


def test_a_repeated_passage_identifier_is_refused() -> None:
    pool = build_pool()
    pool["items"][1]["item_id"] = "it-a"
    with pytest.raises(ExperimentError, match="twice"):
        validate_pool(pool)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ("it-a", "it-a", "with itself"),
        ("it-a", "it-c", "different questions"),
        ("it-a", "it-z", "does not hold"),
    ],
)
def test_a_pair_that_does_not_join_two_passages_of_one_question_is_refused(
    left: str, right: str, message: str
) -> None:
    pool = build_pool()
    pool["pairs"][0]["left_item_id"] = left
    pool["pairs"][0]["right_item_id"] = right
    with pytest.raises(ExperimentError, match=message):
        validate_pool(pool)


def test_a_repeated_pair_identifier_is_refused() -> None:
    pool = build_pool()
    pool["pairs"].append(dict(pool["pairs"][0]))
    with pytest.raises(ExperimentError, match="twice"):
        validate_pool(pool)


# Binding an annotation to the pool it was read from.


def test_a_foreign_pool_name_is_refused() -> None:
    annotations = build_annotated()
    annotations["pool_id"] = "pool-02"
    with pytest.raises(ExperimentError, match="names pool 'pool-02'"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_foreign_pool_digest_is_refused() -> None:
    annotations = build_annotated()
    annotations["pool_sha256"] = OTHER_SHA
    with pytest.raises(ExperimentError, match="different pool bytes|digesting to"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_digest_that_is_not_one_is_refused() -> None:
    annotations = build_annotated()
    with pytest.raises(ExperimentError, match="not a sha256 digest"):
        validate_annotations(build_pool(), annotations, pool_sha256="")


def test_a_foreign_rubric_digest_is_refused() -> None:
    annotations = build_annotated()
    annotations["rubric_sha256"] = OTHER_SHA
    with pytest.raises(ExperimentError, match="rubric digesting to"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_rubric_that_moved_under_the_reader_is_refused() -> None:
    """The same pool with a different rubric is a different bundle."""

    pool = build_pool()
    annotations = make_template(pool, pool_sha256=POOL_SHA)
    other = build_pool()
    other["rubric"]["dimensions"]["relevance"]["choices"][3]["definition"] = "changed"
    with pytest.raises(ExperimentError, match="rubric digesting to"):
        validate_annotations(other, annotations, pool_sha256=POOL_SHA)


def test_an_unnamed_annotator_is_refused_when_it_is_not_text() -> None:
    annotations = build_annotated()
    annotations["annotator"] = 7
    with pytest.raises(ExperimentError, match="must be text"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_an_acknowledgement_that_is_not_a_boolean_is_refused(value: Any) -> None:
    """It is the reader's statement; a tool may not stand in for it."""

    annotations = build_annotated()
    annotations["rubric_acknowledged"] = value
    with pytest.raises(ExperimentError, match="must be true or false"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_an_acknowledgement_a_tool_set_by_accident_is_refused() -> None:
    """A present-but-empty acknowledgement is not a reader's statement either."""

    annotations = build_annotated()
    del annotations["rubric_acknowledged"]
    with pytest.raises(ExperimentError, match="missing"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_an_unknown_annotation_key_is_refused() -> None:
    annotations = build_annotated()
    annotations["quality"] = 0.9
    with pytest.raises(ExperimentError, match="does not define"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


# Inventory: a row that is absent is refused, a grade inside a row that is present
# is reported as pending.


def test_a_dropped_row_is_refused_rather_than_read_as_pending() -> None:
    """A missing row and an ungraded row mean different things."""

    annotations = build_annotated()
    annotations["items"].pop()
    with pytest.raises(ExperimentError, match="missing 1 of the pool's 3 item rows"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_dropped_pair_row_is_refused() -> None:
    annotations = build_annotated()
    annotations["pairs"] = []
    with pytest.raises(ExperimentError, match="missing 1 of the pool's 1 pair rows"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_repeated_row_is_refused() -> None:
    annotations = build_annotated()
    annotations["items"].append(dict(annotations["items"][0]))
    with pytest.raises(ExperimentError, match="repeats"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_row_naming_a_passage_the_pool_does_not_hold_is_refused() -> None:
    annotations = build_annotated()
    annotations["items"][0]["item_id"] = "it-z"
    with pytest.raises(ExperimentError, match="does not hold"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_an_unknown_field_in_a_row_is_refused() -> None:
    annotations = build_annotated()
    annotations["items"][0]["origin"] = "arm-2"
    with pytest.raises(ExperimentError, match="does not define"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_two_mixed_dupicates_are_refused_rather_than_merged() -> None:
    """One duplicate is enough to refuse; two rows never silently become one."""

    pool = build_pool()
    annotations = make_template(pool, pool_sha256=POOL_SHA)
    annotations["items"][1]["item_id"] = "it-a"
    annotations["pairs"][0]["pair_id"] = "pr-1"
    annotations["pairs"].append(dict(annotations["pairs"][0]))
    with pytest.raises(ExperimentError, match="repeats"):
        validate_annotations(pool, annotations, pool_sha256=POOL_SHA)


# Grades: a number, a boolean, or a word this rubric never offered is not a grade.


@pytest.mark.parametrize("value", [0, 1, True, False, "direct ", "Direct", ["direct"]])
def test_a_grade_that_is_not_one_of_the_rubric_words_is_refused(value: Any) -> None:
    annotations = build_annotated()
    annotations["items"][0]["relevance"] = value
    with pytest.raises(ExperimentError):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


@pytest.mark.parametrize("value", [0, 1, "true", "yes", 1.0])
def test_a_source_check_that_is_not_true_or_false_is_refused(value: Any) -> None:
    """`0` compares equal to `False` in Python, so a number must not pass as one."""

    annotations = build_annotated()
    annotations["items"][0]["source_verified"] = value
    with pytest.raises(ExperimentError):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_a_source_left_unchecked_is_a_finding_not_an_error() -> None:
    """Not opening the original is a decision the reader made, so it is kept.

    It differs from pending, which is the state before any decision was made, and
    it differs from a checked original. Counting only the checked ones keeps the
    three states apart.
    """

    annotations = build_annotated()
    annotations["items"][0]["source_verified"] = False
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["items"]["source_verified"] == 2
    assert read["items"]["fully_graded"] == 3
    assert read["status"] == "complete"
    assert read["pending"]["items"] == []


def test_a_grade_this_pool_rubric_does_not_offer_is_refused() -> None:
    """A bundle may narrow its own rubric; a row may not use a word it dropped."""

    pool = build_pool()
    annotations = make_template(pool, pool_sha256=POOL_SHA)
    annotations["annotator"] = "reader"
    annotations["rubric_acknowledged"] = True
    for row in annotations["items"]:
        row.update(relevance="direct", usability="usable", source_verified=True)
    annotations["pairs"][0]["relation"] = "copy"
    narrowed = build_pool()
    narrowed["rubric"]["dimensions"]["relevance"]["choices"] = [
        choice
        for choice in narrowed["rubric"]["dimensions"]["relevance"]["choices"]
        if choice["value"] != "direct"
    ]
    annotations["rubric_sha256"] = canonical_digest(narrowed["rubric"])
    with pytest.raises(ExperimentError, match="does not offer"):
        validate_annotations(narrowed, annotations, pool_sha256=POOL_SHA)


def test_a_rubric_that_offers_a_grade_this_toolkit_cannot_read_is_refused() -> None:
    pool = build_pool()
    pool["rubric"]["dimensions"]["usability"]["choices"][0]["value"] = "fine"
    with pytest.raises(ExperimentError, match="cannot interpret"):
        validate_pool(pool)


def test_a_rubric_without_an_undecided_grade_is_refused() -> None:
    pool = build_pool()
    choices = pool["rubric"]["dimensions"]["relevance"]["choices"]
    pool["rubric"]["dimensions"]["relevance"]["choices"] = [
        choice for choice in choices if choice["value"] != "uncertain"
    ]
    with pytest.raises(ExperimentError, match="could not decide"):
        validate_pool(pool)


def test_a_rubric_grade_with_no_definition_is_refused() -> None:
    pool = build_pool()
    pool["rubric"]["dimensions"]["relation"]["choices"][0]["definition"] = "  "
    with pytest.raises(ExperimentError, match="states no definition"):
        validate_pool(pool)


# Pending is pending: the states a review passes through.


def test_an_untouched_template_is_incomplete_and_reads_no_failures() -> None:
    """Every grade pending is a review in progress, not a set of negative grades."""

    pool = build_pool()
    annotations = make_template(pool, pool_sha256=POOL_SHA)
    read = validate_annotations(pool, annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["items"] == {
        "total": 3,
        "fully_graded": 0,
        "source_verified": 0,
        "noted": 0,
    }
    assert read["pairs"] == {"total": 1, "graded": 0, "noted": 0}
    assert read["pending"] == {"items": ["it-a", "it-b", "it-c"], "pairs": ["pr-1"]}
    assert read["uncertain"] == {"items": [], "pairs": []}
    assert read["policy_ready"] is False


def test_one_pending_grade_keeps_the_annotation_incomplete() -> None:
    pool = build_pool()
    annotations = make_template(pool, pool_sha256=POOL_SHA)
    annotations["annotator"] = "reader"
    annotations["rubric_acknowledged"] = True
    for row in annotations["items"]:
        row.update(relevance="direct", usability="usable", source_verified=True)
    annotations["items"][0]["usability"] = None
    annotations["pairs"][0]["relation"] = "copy"
    read = validate_annotations(pool, annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["pending"]["items"] == ["it-a"]
    assert read["items"]["fully_graded"] == 2


def test_an_unread_source_is_pending_rather_than_a_refusal() -> None:
    pool = build_pool()
    annotations = build_annotated()
    annotations["items"][2]["source_verified"] = None
    read = validate_annotations(pool, annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["items"]["source_verified"] == 2
    assert read["pending"]["items"] == ["it-c"]


def test_an_explicit_uncertain_grade_wants_a_second_reader() -> None:
    annotations = build_annotated()
    annotations["items"][1]["relevance"] = "uncertain"
    annotations["items"][1]["notes"] = "reads as either a quotation or a paraphrase"
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "adjudication_required"
    assert read["uncertain"] == {"items": ["it-b"], "pairs": []}


def test_an_uncertain_pair_wants_a_second_reader() -> None:
    annotations = build_annotated()
    annotations["pairs"][0]["relation"] = "uncertain"
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "adjudication_required"
    assert read["uncertain"]["pairs"] == ["pr-1"]


def test_an_uncertain_grade_outranks_another_row_still_pending() -> None:
    """A finished row that wants adjudication is reported as such, not as unfinished."""

    annotations = build_annotated()
    annotations["items"][1]["usability"] = "uncertain"
    annotations["items"][2]["relevance"] = None
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "adjudication_required"
    assert read["pending"]["items"] == ["it-c"]
    assert read["pending"]["pairs"] == []


def test_a_fully_graded_annotation_is_complete_and_still_not_policy_ready() -> None:
    """A finished reading is not a quality result and says so in the record."""

    read = validate_annotations(build_pool(), build_annotated(), pool_sha256=POOL_SHA)

    assert read["status"] == "complete"
    assert read["policy_ready"] is False
    assert read["rubric_acknowledged"] is True
    assert read["annotator_present"] is True
    assert read["pending"] == {"items": [], "pairs": []}
    assert read["uncertain"] == {"items": [], "pairs": []}


def test_a_complete_annotation_still_needs_the_rubric_acknowledged() -> None:
    annotations = build_annotated()
    annotations["rubric_acknowledged"] = False
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["rubric_acknowledged"] is False


def test_a_complete_annotation_still_needs_a_named_reader() -> None:
    annotations = build_annotated(annotator="   ")
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["annotator_present"] is False


def test_notes_are_counted_without_being_required() -> None:
    annotations = build_annotated()
    annotations["items"][0]["notes"] = "the passage answers a different question"
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["items"]["noted"] == 1
    assert read["status"] == "complete"


# A claim about two sources is a claim, and it has to rest on the sources.


def test_a_contradiction_with_a_note_and_two_checked_originals_is_accepted() -> None:
    annotations = build_annotated()
    annotations["pairs"][0]["relation"] = "contradiction"
    annotations["pairs"][0]["notes"] = "one states 40 adults, the other 41"
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "complete"


def test_a_contradiction_with_no_note_is_refused() -> None:
    annotations = build_annotated()
    annotations["pairs"][0]["relation"] = "contradiction"
    with pytest.raises(ExperimentError, match="with no note"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


def test_an_independent_corroboration_with_no_note_is_refused() -> None:
    """Shared wording between two extractions is a diagnostic, not a finding."""

    annotations = build_annotated()
    annotations["pairs"][0]["relation"] = "independent_corroboration"
    annotations["pairs"][0]["notes"] = "   "
    with pytest.raises(ExperimentError, match="with no note"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


@pytest.mark.parametrize("unchecked", ["it-a", "it-b"])
@pytest.mark.parametrize("checked", [None, False])
def test_a_source_claim_with_an_unchecked_side_is_refused(
    unchecked: str, checked: Any
) -> None:
    annotations = build_annotated()
    for row in annotations["items"]:
        row["source_verified"] = True
    annotations["items"][0]["item_id"] = "it-a"
    annotations["items"][1]["item_id"] = "it-b"
    target = next(row for row in annotations["items"] if row["item_id"] == unchecked)
    target["source_verified"] = checked
    annotations["pairs"][0]["relation"] = "contradiction"
    annotations["pairs"][0]["notes"] = "the two disagree"
    with pytest.raises(ExperimentError, match="not checked in the original"):
        validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)


@pytest.mark.parametrize(
    "relation", ["copy", "overlap", "related_distinct", "unrelated"]
)
def test_a_textual_relation_needs_no_source_and_no_note(relation: str) -> None:
    """Repetition is observable in the text, so it needs no opened original."""

    annotations = build_annotated()
    annotations["pairs"][0]["relation"] = relation
    for row in annotations["items"]:
        row["source_verified"] = None
    read = validate_annotations(build_pool(), annotations, pool_sha256=POOL_SHA)

    assert read["status"] == "incomplete"
    assert read["pairs"]["graded"] == 1


# Hostile text in the pool is text, not markup.


def test_injected_passage_text_is_kept_as_text() -> None:
    pool = build_pool()
    pool["items"][0]["passage"] = INJECTION
    pool["items"][0]["source"]["title"] = INJECTION

    validate_pool(pool)
    template = make_template(pool, pool_sha256=POOL_SHA)

    assert pool["items"][0]["passage"] == INJECTION
    assert json.loads(json.dumps(template)) == template
