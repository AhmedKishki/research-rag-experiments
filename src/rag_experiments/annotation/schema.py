"""The pool an author reads, and the annotation they write back.

Two records meet here. The *blinded pool* is what the reader is shown: questions,
extracted passage text, scholarly source facts, and opaque identifiers. It carries
no engine name, no ordering, no result value, no designated target, and no raw
query, chunk, or target id, because an annotator who can see which arm produced a
passage grades the arm rather than the passage. The *annotation* is what the reader
returns: the same version and digests, an annotator name, a rubric
acknowledgement, and one pending row per pool item and per pool pair.

Three properties are load-bearing.

A grade is pending or it is a reading. `null` means the annotator has not decided,
which is not a negative grade, and a validator that counted pending rows as
failures would be inventing a judgment on their behalf. The choices are the
rubric's, and nothing in this module derives a grade from a passage's length, its
metadata, a system's output, or the fact that a passage was designated as a target.
The rubric ships here as data; it assigns nothing.

An annotation is bound to the pool it was read from. It carries the digest of the
pool file's bytes and the digest of that pool's own rubric, so a template from one
bundle cannot be filled in against another, and so a rubric that changed after an
annotator read it cannot be presented as the one they read. A complete annotation
means every grade is one the annotator chose. It is not a quality result, not a
confirmation of any ordering, and not a basis for a policy decision.

A claim about sources is a claim, not an inference. Repeated or near-identical
text is a diagnostic; it cannot establish that two passages corroborate each other
or contradict each other. Those two relations therefore require both passages to
be checked in the original and the reason written down, and a validator that
accepted them without either would be recording a machine's suspicion as a
human's finding.

What this module does not do, and cannot:

- It does not know whether an identifier leaks. It refuses identifiers outside a
  restricted opaque-token shape, which keeps separators, quotes, and markup out of
  them, but the builder that assigns ids owns the promise that no identifier
  contains a raw query, chunk, or target id. Blindness is kept where ids are
  minted.
- It does not check the pool file against those bytes. It compares the digest the
  caller passes, so the caller owns that it is the digest of the pool being shown.
- It reads no original, no PDF, and no transcript. Nothing here can confirm that an
  extraction says what its source says.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..errors import ExperimentError

#: The blinded pool's own version, and the annotation's. Both are 1, and both are
#: stated in the records rather than inferred, because a reader has to be able to
#: refuse a pool written by a builder that means something else by these keys.
POOL_SCHEMA_VERSION = 1
ANNOTATION_SCHEMA_VERSION = 1

#: The only top-level keys a pool may carry. An unexpected key is refused rather
#: than ignored: a key named after an engine, an ordering, or a result value is the
#: exact way a blinded pool stops being blinded, and dropping it quietly would leave
#: the reader looking at a pool whose builder had already leaked.
POOL_KEYS = (
    "schema_version",
    "pool_id",
    "role",
    "rubric",
    "questions",
    "items",
    "pairs",
)

QUESTION_KEYS = ("question_id", "query")
ITEM_KEYS = ("item_id", "question_id", "passage", "source")
PAIR_KEYS = ("pair_id", "question_id", "left_item_id", "right_item_id")

#: What a card may show about where a passage came from. `locator` is a position
#: inside the source, `original` a relative link to the file beside the review
#: page, and `source_relative_path` the path the source occupies inside the corpus.
#: Scholarly identifying facts are allowed because an annotator confirming a
#: quotation needs to know which work it is from. Nothing describing how the
#: passage was retrieved or selected is allowed, and there is no key for it.
SOURCE_KEYS = (
    "title",
    "authors",
    "year",
    "doi",
    "source_relative_path",
    "locator",
    "original",
)
SOURCE_REQUIRED_KEYS = ("locator", "original")

#: The annotation's own keys. `pool_sha256` is the digest of the blinded pool file's
#: bytes as served and `rubric_sha256` the canonical digest of that pool's rubric,
#: so the record names what the reader was looking at. `rubric_acknowledged` is the
#: annotator's own statement that they read the rubric before grading; a record
#: that never acknowledges it cannot be complete.
ANNOTATION_KEYS = (
    "schema_version",
    "pool_id",
    "pool_sha256",
    "rubric_sha256",
    "rubric_acknowledged",
    "annotator",
    "items",
    "pairs",
)

ITEM_ROW_KEYS = ("item_id", "relevance", "usability", "source_verified", "notes")
PAIR_ROW_KEYS = ("pair_id", "relation", "notes")

#: The grade sets. A bundle's rubric may draw from these, reorder them, and reword
#: their labels. It may not introduce a grade this validator cannot interpret, and
#: it may not drop the undecided one, because then `adjudication_required` would
#: silently mean nothing.
RELEVANCE_CHOICES = ("irrelevant", "contextual", "direct", "uncertain")
USABILITY_CHOICES = ("usable", "needs_context", "unusable", "uncertain")
RELATION_CHOICES = (
    "copy",
    "overlap",
    "independent_corroboration",
    "contradiction",
    "related_distinct",
    "unrelated",
    "uncertain",
)

#: `source_verified` is a tri-state, and the middle state is `null`. `False` means
#: the original was not opened, which is not a finding that the passage is wrong.
SOURCE_VERIFIED_CHOICES = (False, True)

#: The graded dimensions, by the name the records use for them.
DIMENSIONS = ("relevance", "usability", "relation", "source_verified")

DIMENSION_CHOICES: dict[str, tuple[Any, ...]] = {
    "relevance": RELEVANCE_CHOICES,
    "usability": USABILITY_CHOICES,
    "relation": RELATION_CHOICES,
    "source_verified": SOURCE_VERIFIED_CHOICES,
}

#: The value that says the annotator could not decide, per dimension. `null`
#: already says *not yet*, so the source dimension has no such value: whether the
#: original was opened is a fact about the reading, and it is left pending rather
#: than given a third value.
UNCERTAIN_VALUE: dict[str, Any] = {
    "relevance": "uncertain",
    "usability": "uncertain",
    "relation": "uncertain",
    "source_verified": None,
}

#: Relations that assert something about two *sources*. Two passages that repeat
#: each other prove that the text repeats; they cannot prove that two authors
#: independently corroborated or contradicted one another. So these require both
#: originals to have been opened and the reason written down.
SOURCE_CLAIM_RELATIONS = ("contradiction", "independent_corroboration")

#: The statuses a validated annotation may carry.
STATUS_INCOMPLETE = "incomplete"
STATUS_ADJUDICATION_REQUIRED = "adjudication_required"
STATUS_COMPLETE = "complete"

#: An identifier in a blinded pool is an opaque token: letters, digits, underscore,
#: and dash. This is a shape, not a format, and the builder chooses the tokens. What
#: it buys is that no separator, quote, space, colon, or markup character can ride
#: in an identifier, so an id cannot become a path or break a document. It cannot
#: prove an id is not a disguised chunk id; that promise is the builder's, restated
#: here so nobody assumes this module kept it.
_OPAQUE_ID = re.compile(r"\A[A-Za-z0-9_-]+\Z")
_OPAQUE_ID_MAX_LENGTH = 64

_HEX_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")

#: The rubric a bundle freezes into its pools. It is data: it says what each grade
#: means and what the reader may not do, and it assigns no grade to anything. Every
#: pool carries its own copy, and an annotation binds to that copy by digest rather
#: than to this constant by name, so a rubric revised before an annotator reads it
#: is a new pool rather than a silent change under them.
RUBRIC: dict[str, Any] = {
    "schema_version": 1,
    "name": "blinded-pool-review",
    "scope": (
        "Each card shows one question and extracted passage text, or two passages "
        "for one question, with nothing saying which engine, ordering, or "
        "designated target produced either passage."
    ),
    "extraction_notice": (
        "Every passage here is extracted text. It is not a transcript, not a page "
        "image, and not a copy of the source. The linked original file is the only "
        "authority for what the source says, and the extraction may have lost "
        "layout, hyphenation, footnotes, or figure captions."
    ),
    "instructions": [
        (
            "Read the question first, then the passage, and grade what the passage "
            "itself states rather than what it reminds you of."
        ),
        (
            "A grade is your reading of this text. It is not a measurement and not a "
            "statement about any engine."
        ),
        (
            "Leave a grade pending rather than guessing. A pending grade is recorded "
            "as pending and is never read as a negative one."
        ),
        (
            "Choose Uncertain when two readings are equally available, and write the "
            "reason in the notes so the row can be adjudicated rather than guessed at."
        ),
        (
            "Open the original whenever a quotation, number, or attribution carries "
            "the answer, and mark the source verified only once you have done so."
        ),
    ],
    "dimensions": {
        "relevance": {
            "label": "Relevance to the question",
            "question": "Does this passage bear on the question that was asked?",
            "uncertain_value": "uncertain",
            "choices": [
                {
                    "value": "irrelevant",
                    "label": "Irrelevant",
                    "definition": (
                        "The passage does not bear on the question, or is about a "
                        "different subject from the one asked about."
                    ),
                },
                {
                    "value": "contextual",
                    "label": "Contextual",
                    "definition": (
                        "The passage is on the subject and may be useful "
                        "background, but it does not address what was asked."
                    ),
                },
                {
                    "value": "direct",
                    "label": "Direct",
                    "definition": (
                        "The passage states what the question asks for, or answers "
                        "it as far as this text allows."
                    ),
                },
                {
                    "value": "uncertain",
                    "label": "Uncertain",
                    "definition": (
                        "Two readings are equally available, or the question is too "
                        "vague to decide between them. Write the reason."
                    ),
                },
            ],
        },
        "usability": {
            "label": "Usability as it stands",
            "question": "Could this passage go into an answer without more work?",
            "uncertain_value": "uncertain",
            "choices": [
                {
                    "value": "usable",
                    "label": "Usable",
                    "definition": (
                        "The passage can be quoted or paraphrased into an answer as "
                        "it stands."
                    ),
                },
                {
                    "value": "needs_context",
                    "label": "Needs context",
                    "definition": (
                        "The passage is on topic but is missing something an answer "
                        "needs, such as what it refers to or who is speaking."
                    ),
                },
                {
                    "value": "unusable",
                    "label": "Unusable",
                    "definition": (
                        "The passage cannot support an answer, whether because it is "
                        "empty, garbled, truncated, or simply says something else."
                    ),
                },
                {
                    "value": "uncertain",
                    "label": "Uncertain",
                    "definition": (
                        "You cannot tell whether the passage is usable without "
                        "finding the original. Write the reason."
                    ),
                },
            ],
        },
        "relation": {
            "label": "Relation between two passages",
            "question": "How do these two passages stand to one another?",
            "uncertain_value": "uncertain",
            "choices": [
                {
                    "value": "copy",
                    "label": "Copy",
                    "definition": (
                        "One passage carries the other's wording and adds nothing."
                    ),
                },
                {
                    "value": "overlap",
                    "label": "Overlap",
                    "definition": (
                        "The passages share material, and neither adds anything the "
                        "other lacks."
                    ),
                },
                {
                    "value": "independent_corroboration",
                    "label": "Independent corroboration",
                    "definition": (
                        "Two separate sources make the same claim. Open both "
                        "originals, mark both verified, and write the reason. Shared "
                        "wording is not independence."
                    ),
                },
                {
                    "value": "contradiction",
                    "label": "Contradiction",
                    "definition": (
                        "The two claims cannot both hold. Open both originals, mark "
                        "both verified, and write the reason. Differing wording is "
                        "not a contradiction."
                    ),
                },
                {
                    "value": "related_distinct",
                    "label": "Related but distinct",
                    "definition": (
                        "The passages treat the same subject and each adds something "
                        "the other does not."
                    ),
                },
                {
                    "value": "unrelated",
                    "label": "Unrelated",
                    "definition": (
                        "The passages do not bear on each other beyond being in the "
                        "same pool."
                    ),
                },
                {
                    "value": "uncertain",
                    "label": "Uncertain",
                    "definition": (
                        "You cannot tell how they stand to each other. Write the "
                        "reason."
                    ),
                },
            ],
        },
        "source_verified": {
            "label": "Checked against the original",
            "question": "Did you open the original and confirm it says this?",
            "uncertain_value": None,
            "choices": [
                {
                    "value": False,
                    "label": "Not checked",
                    "definition": (
                        "The original was not opened for this passage. This is not a "
                        "finding that the passage is wrong."
                    ),
                },
                {
                    "value": True,
                    "label": "Checked original",
                    "definition": (
                        "The original was opened and it says what the extraction shows."
                    ),
                },
            ],
        },
    },
    "rules": [
        "A pending grade is null and is not zero.",
        (
            "Grades are never derived from a passage's length, its metadata, a system's "
            "output, or the fact that a passage was designated as a target."
        ),
        (
            "An independent corroboration or a contradiction is a claim about sources, "
            "so both originals must be checked and the reason written down."
        ),
        (
            "A complete annotation is a finished reading. It is not a quality result, "
            "not a confirmation of any ordering, and not a basis for a policy decision."
        ),
    ],
}


def canonical_digest(value: Any) -> str:
    """The digest of a JSON value in one canonical form.

    Keys are sorted and separators are fixed, so two values that are equal as data
    digest identically however either was serialized. That is what lets a rubric
    digest name the rubric rather than the bytes a writer happened to emit.
    """

    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def make_template(
    pool: Any, *, pool_sha256: str, annotator: str = ""
) -> dict[str, Any]:
    """The pending annotation for one pool: every row present, every grade null.

    The template is a full inventory rather than a blank. A reader who has graded
    two passages and stopped has still produced a record, and a record listing only
    the rows they happened to reach could not be told apart from one where the rest
    were dropped on purpose. Null grades are therefore pending, and this function
    refuses a pool it cannot fully enumerate.
    """

    validate_pool(pool)
    _require_digest(pool_sha256, "The blinded pool's digest")
    if not isinstance(annotator, str):
        raise ExperimentError(
            f"The annotator name must be text, not {type(annotator).__name__}."
        )
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "pool_id": pool["pool_id"],
        "pool_sha256": pool_sha256,
        "rubric_sha256": canonical_digest(pool["rubric"]),
        "rubric_acknowledged": False,
        "annotator": annotator,
        "items": [
            {
                "item_id": item["item_id"],
                "relevance": None,
                "usability": None,
                "source_verified": None,
                "notes": "",
            }
            for item in pool["items"]
        ],
        "pairs": [
            {"pair_id": pair["pair_id"], "relation": None, "notes": ""}
            for pair in pool["pairs"]
        ],
    }


def validate_pool(pool: Any) -> None:
    """Refuse a blinded pool that is not this contract, and pass one that is.

    Every check here is about the pool being readable and blind-safe as a
    structure: named keys, string types, opaque identifiers, resolvable references,
    and an original link that stays beside the review page. None of it can tell
    whether an identifier or a passage was chosen in a way that leaks which engine
    produced it.
    """

    _require_object(pool, "The blinded pool")
    _require_keys(pool, POOL_KEYS, POOL_KEYS, "The blinded pool")
    _require_version(pool, POOL_SCHEMA_VERSION, "The blinded pool")
    _require_opaque(pool, "pool_id", "The blinded pool")
    if not isinstance(pool["role"], str) or not pool["role"].strip():
        raise ExperimentError(
            "The blinded pool states no role. A reader has to be able to see whether "
            "these passages are a held-out measurement or an exploratory partition "
            "before grading them."
        )
    _rubric(pool)
    questions = _questions(pool)
    items = _items(pool, questions)
    _pairs(pool, questions, items)


def validate_annotations(
    pool: Any, annotations: Any, *, pool_sha256: str
) -> dict[str, Any]:
    """Read an annotation against its pool, and report how far it has got.

    Structural faults are refused, because an annotation naming another pool,
    repeating a row, dropping a row, carrying a grade this rubric does not offer, or
    claiming a corroboration or contradiction without checked sources is not a
    partly finished reading but a record that cannot be read at all. A *missing
    grade* is not a fault: a review in progress is the normal state of this file, and
    it is reported as incomplete.

    The returned status is about the annotation and nothing else:

    - `incomplete`: at least one required grade is still pending, or the rubric was
      never acknowledged, or no annotator named themselves.
    - `adjudication_required`: at least one grade is explicitly Uncertain. That row
      is finished and wants a second reader, which is a different thing from being
      unfinished.
    - `complete`: every grade is one the annotator chose, against a rubric they
      acknowledged. That is a finished reading and not a quality result.
    """

    validate_pool(pool)
    _require_digest(pool_sha256, "The blinded pool's digest")
    _require_object(annotations, "The annotation")
    _require_keys(annotations, ANNOTATION_KEYS, ANNOTATION_KEYS, "The annotation")
    _require_version(annotations, ANNOTATION_SCHEMA_VERSION, "The annotation")
    if annotations["pool_id"] != pool["pool_id"]:
        raise ExperimentError(
            f"The annotation names pool {annotations['pool_id']!r} and the pool being "
            f"checked is {pool['pool_id']!r}. Grades written against one pool "
            "describe that pool's passages and cannot be read against another's."
        )
    if annotations["pool_sha256"] != pool_sha256:
        raise ExperimentError(
            f"The annotation records pool bytes digesting to "
            f"{_short(annotations['pool_sha256'])} and the pool being checked "
            f"digests to {_short(pool_sha256)}. The reader graded one set of passages "
            "and this file holds another, or the pool changed under them."
        )
    expected_rubric = canonical_digest(pool["rubric"])
    if annotations["rubric_sha256"] != expected_rubric:
        raise ExperimentError(
            f"The annotation was filled in against a rubric digesting to "
            f"{_short(annotations['rubric_sha256'])} and this pool's rubric digests "
            f"to {_short(expected_rubric)}. A grade means what the rubric in front of "
            "the reader meant, so a rubric that moved under them cannot be presented "
            "as the one they graded against."
        )
    acknowledged = annotations["rubric_acknowledged"]
    if not isinstance(acknowledged, bool):
        raise ExperimentError(
            "The annotation's `rubric_acknowledged` must be true or false, not "
            f"{type(acknowledged).__name__}. It is the reader's statement that they "
            "graded against the rubric, and no tool sets it for them."
        )
    annotator = annotations["annotator"]
    if not isinstance(annotator, str):
        raise ExperimentError(
            f"The annotation's `annotator` must be text, not "
            f"{type(annotator).__name__}. An unnamed reading cannot be attributed to "
            "a reader."
        )

    rubric = pool["rubric"]
    item_ids = [item["item_id"] for item in pool["items"]]
    pair_ids = [pair["pair_id"] for pair in pool["pairs"]]
    item_rows = _rows(annotations["items"], item_ids, "item_id", ITEM_ROW_KEYS, "item")
    pair_rows = _rows(annotations["pairs"], pair_ids, "pair_id", PAIR_ROW_KEYS, "pair")

    item_grades: dict[str, dict[str, Any]] = {}
    graded_items = 0
    verified_items = 0
    noted_items = 0
    uncertain_items: list[str] = []
    pending_items: list[str] = []
    for identifier, row in zip(item_ids, item_rows):
        where = f"Annotation item {identifier!r}"
        grades = {
            dimension: _grade(row[dimension], dimension, rubric, where)
            for dimension in ("relevance", "usability", "source_verified")
        }
        item_grades[identifier] = grades
        decided = [value for value in grades.values() if value is not None]
        if len(decided) == 3:
            graded_items += 1
        else:
            pending_items.append(identifier)
        if grades["source_verified"] is True:
            verified_items += 1
        if row["notes"]:
            noted_items += 1
        uncertain_items.extend(
            identifier
            for dimension, value in grades.items()
            if value is not None and value == UNCERTAIN_VALUE[dimension]
        )

    graded_pairs = 0
    noted_pairs = 0
    uncertain_pairs: list[str] = []
    pending_pairs: list[str] = []
    for identifier, row in zip(pair_ids, pair_rows):
        where = f"Annotation pair {identifier!r}"
        relation = _grade(row["relation"], "relation", rubric, where)
        if relation is None:
            pending_pairs.append(identifier)
        else:
            graded_pairs += 1
            if relation == UNCERTAIN_VALUE["relation"]:
                uncertain_pairs.append(identifier)
        if row["notes"]:
            noted_pairs += 1
        if relation in SOURCE_CLAIM_RELATIONS:
            pair = next(
                entry for entry in pool["pairs"] if entry["pair_id"] == identifier
            )
            _require_source_claim(
                identifier,
                relation,
                row["notes"],
                (pair["left_item_id"], pair["right_item_id"]),
                item_grades,
            )

    acknowledged_and_named = acknowledged and bool(annotator.strip())
    if uncertain_items or uncertain_pairs:
        status = STATUS_ADJUDICATION_REQUIRED
    elif pending_items or pending_pairs or not acknowledged_and_named:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_COMPLETE
    return {
        "status": status,
        # A finished reading is not a result. Nothing here may be read as evidence
        # that any engine retrieved well, and this field states that.
        "policy_ready": False,
        "rubric_acknowledged": acknowledged,
        "annotator_present": bool(annotator.strip()),
        "items": {
            "total": len(item_ids),
            "fully_graded": graded_items,
            "source_verified": verified_items,
            "noted": noted_items,
        },
        "pairs": {
            "total": len(pair_ids),
            "graded": graded_pairs,
            "noted": noted_pairs,
        },
        "uncertain": {"items": uncertain_items, "pairs": uncertain_pairs},
        "pending": {"items": pending_items, "pairs": pending_pairs},
    }


def _rubric(pool: dict[str, Any]) -> dict[str, Any]:
    """The pool's own rubric, read as a rubric this validator can interpret."""

    rubric = pool["rubric"]
    _require_object(rubric, "The blinded pool's rubric")
    _require_version(rubric, POOL_SCHEMA_VERSION, "The blinded pool's rubric")
    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ExperimentError(
            "The blinded pool's rubric states no `dimensions`, so no grade could be "
            "read against it."
        )
    absent = [name for name in DIMENSIONS if name not in dimensions]
    if absent:
        raise ExperimentError(
            f"The blinded pool's rubric defines no {', '.join(absent)} dimension. An "
            "annotation graded against it would carry values nothing can interpret."
        )
    for name in DIMENSIONS:
        _require_dimension(dimensions[name], name)
    return rubric


def _require_dimension(dimension: Any, name: str) -> None:
    """One rubric dimension, with grades this module can act on."""

    where = f"The blinded pool's rubric {name} dimension"
    _require_object(dimension, where)
    for key in ("label", "question"):
        value = dimension.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ExperimentError(
                f"{where} states no {key}. The reader has to be shown what the grade "
                "asks before making it."
            )
    choices = dimension.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ExperimentError(f"{where} lists no grades, so nothing can be chosen.")
    known = DIMENSION_CHOICES[name]
    seen: list[Any] = []
    for entry in choices:
        _require_object(entry, f"A {where} grade")
        if "value" not in entry:
            raise ExperimentError(
                f"A {where} grade states no value, so a saved row could not be read "
                "back as that grade."
            )
        value = entry["value"]
        _require_choice_value(value, known, name, where)
        if value in seen:
            raise ExperimentError(f"{where} offers {value!r} twice.")
        seen.append(value)
        if not isinstance(entry.get("label"), str) or not entry["label"].strip():
            raise ExperimentError(f"The {where} grade {value!r} states no label.")
        definition = entry.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            raise ExperimentError(
                f"The {where} grade {value!r} states no definition. A grade the "
                "reader cannot look up is a coin toss."
            )
    if name == "source_verified":
        if dimension.get("uncertain_value") is not None:
            raise ExperimentError(
                f"{where} names an undecided value. Whether the original was opened "
                "is a fact about the reading, and it stays pending as null rather than "
                "being given a value."
            )
        return
    if dimension.get("uncertain_value") != UNCERTAIN_VALUE[name] or (
        UNCERTAIN_VALUE[name] not in seen
    ):
        raise ExperimentError(
            f"{where} does not offer {UNCERTAIN_VALUE[name]!r} as the value that says "
            "the reader could not decide. Without it, a finished row that wants a "
            "second reader would read as a decided one."
        )


def _require_choice_value(
    value: Any, known: tuple[Any, ...], name: str, where: str
) -> None:
    """A grade value must be one this module knows, and of the right JSON type.

    `0` is refused where `False` is legal and `"direct"` where a string is, because
    Python compares them equal and a validator that accepted them would read a
    number as a finding.
    """

    if name == "source_verified":
        if isinstance(value, bool):
            return
        raise ExperimentError(
            f"{where} offers {value!r}, and a source check is true or false."
        )
    if isinstance(value, str) and value in known:
        return
    raise ExperimentError(
        f"{where} offers {value!r}, which is not one of "
        f"{', '.join(repr(choice) for choice in known)}. A bundle may reorder these "
        "and reword their labels, and it may not introduce a grade this toolkit "
        "cannot interpret."
    )


def _questions(pool: dict[str, Any]) -> dict[str, str]:
    questions = pool["questions"]
    if not isinstance(questions, list) or not questions:
        raise ExperimentError(
            "The blinded pool lists no questions. A passage cannot be graded against "
            "a question that is not there."
        )
    seen: dict[str, str] = {}
    for index, question in enumerate(questions, 1):
        where = f"Question {index} of the blinded pool"
        _require_object(question, where)
        _require_keys(question, QUESTION_KEYS, QUESTION_KEYS, where)
        _require_opaque(question, "question_id", where)
        _require_text(question["query"], "query", where)
        identifier = question["question_id"]
        if identifier in seen:
            raise ExperimentError(
                f"The blinded pool names question {identifier!r} twice. Two cards "
                "would answer one question and neither could be told apart."
            )
        seen[identifier] = question["query"]
    return seen


def _items(pool: dict[str, Any], questions: dict[str, str]) -> dict[str, str]:
    items = pool["items"]
    if not isinstance(items, list) or not items:
        raise ExperimentError(
            "The blinded pool lists no passages, so there is nothing to grade."
        )
    seen: dict[str, str] = {}
    for index, item in enumerate(items, 1):
        where = f"Passage {index} of the blinded pool"
        _require_object(item, where)
        _require_keys(item, ITEM_KEYS, ITEM_KEYS, where)
        _require_opaque(item, "item_id", where)
        _require_text(item["passage"], "passage", where)
        question_id = item["question_id"]
        if not isinstance(question_id, str) or question_id not in questions:
            raise ExperimentError(
                f"{where} names question {question_id!r}, which the pool does not "
                "hold. A passage shown beside no question cannot be graded for "
                "relevance."
            )
        _require_source(item["source"], where)
        identifier = item["item_id"]
        if identifier in seen:
            raise ExperimentError(
                f"The blinded pool names passage {identifier!r} twice. A repeated "
                "identifier is a repeated reading of one passage."
            )
        seen[identifier] = question_id
    return seen


def _pairs(
    pool: dict[str, Any], questions: dict[str, str], items: dict[str, str]
) -> None:
    pairs = pool["pairs"]
    if not isinstance(pairs, list):
        raise ExperimentError(
            "The blinded pool's `pairs` is not a list, so no relation can be read."
        )
    seen: set[str] = set()
    for index, pair in enumerate(pairs, 1):
        where = f"Pair {index} of the blinded pool"
        _require_object(pair, where)
        _require_keys(pair, PAIR_KEYS, PAIR_KEYS, where)
        _require_opaque(pair, "pair_id", where)
        identifier = pair["pair_id"]
        if identifier in seen:
            raise ExperimentError(
                f"The blinded pool names pair {identifier!r} twice. One comparison is "
                "one row."
            )
        seen.add(identifier)
        question_id = pair["question_id"]
        if not isinstance(question_id, str) or question_id not in questions:
            raise ExperimentError(
                f"{where} names question {question_id!r}, which the pool does not hold."
            )
        left = pair["left_item_id"]
        right = pair["right_item_id"]
        for key, value in (("left_item_id", left), ("right_item_id", right)):
            if not isinstance(value, str) or value not in items:
                raise ExperimentError(
                    f"{where} names {key} {value!r}, which the pool does not hold."
                )
        if left == right:
            raise ExperimentError(
                f"{where} pairs passage {left!r} with itself. A passage is not in a "
                "relation with itself, and the row would read as a finding."
            )
        for value in (left, right):
            if items[value] != question_id:
                raise ExperimentError(
                    f"{where} pairs two passages, one of which answers question "
                    f"{items[value]!r} while the pair names {question_id!r}. A "
                    "relation between passages of two different questions would be "
                    "graded against a question neither side answers."
                )


def _require_source(source: Any, where: str) -> None:
    """The facts a card may show about where a passage came from."""

    _require_object(source, f"{where} source")
    _require_keys(source, SOURCE_KEYS, SOURCE_REQUIRED_KEYS, f"{where} source")
    for key in ("title", "authors", "year", "doi", "source_relative_path"):
        value = source.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ExperimentError(
                f"{where} source states an empty {key}. A blank bibliographic fact is "
                "no fact, and a reader cannot tell a missing one from a broken one."
            )
    locator = source["locator"]
    _require_object(locator, f"{where} source locator")
    if not locator:
        raise ExperimentError(
            f"{where} source carries an empty locator, so nothing says where in the "
            "source the passage came from."
        )
    for key, value in locator.items():
        _require_locator_value(value, key, where)
    _require_relative_link(source["original"], f"{where} source original")


def _require_locator_value(value: Any, key: str, where: str) -> None:
    """A locator is a position, so it holds plain facts and nothing else.

    A locator able to hold a nested object could hold anything, and the card would
    then render a structure the pool never described.
    """

    if isinstance(value, (bool, str, int, float)):
        return
    if isinstance(value, list) and all(
        isinstance(entry, (str, int)) for entry in value
    ):
        return
    raise ExperimentError(
        f"{where} source locator holds {key!r} as {type(value).__name__}. A locator "
        "is a position inside the source, so it carries text, numbers, or a list of "
        "them."
    )


def _require_relative_link(value: Any, where: str) -> None:
    """The original must be a file beside the review page, and nothing else.

    The link is opened by hand from a page opened by hand, so it has to resolve on
    this machine without reaching anything. A scheme, an absolute path, a parent
    segment, or a quote character is refused rather than rendered.
    """

    _require_text(value, "original", where)
    if ":" in value or "\\" in value:
        raise ExperimentError(
            f"{where} states {value!r}, which names a scheme or a Windows path rather "
            "than a file beside this page. An original is opened locally and is "
            "never fetched from anywhere."
        )
    if value.startswith("/"):
        raise ExperimentError(
            f"{where} states {value!r}, an absolute path. An original is reached "
            "through this bundle's own directory."
        )
    for segment in value.split("/"):
        if segment in ("", ".", ".."):
            raise ExperimentError(
                f"{where} states {value!r}, which leaves or repeats a path segment. An "
                "original is reached through this bundle's own directory."
            )
    for character in ('"', "'", "<", ">"):
        if character in value:
            raise ExperimentError(
                f"{where} states {value!r}, which holds {character!r}. A path in a "
                "link is refused rather than escaped."
            )


def _rows(
    rows: Any, expected: list[str], id_key: str, allowed: tuple[str, ...], where: str
) -> list[dict[str, Any]]:
    """One row per expected identity, in the pool's order, with nothing extra.

    A missing row is refused rather than read as pending, because a pending row and
    a dropped row mean different things: one is a review in progress and the other
    is a record that cannot say what was skipped. A pending *grade* inside a row
    that is present is pending, and that is reported by the caller.
    """

    if not isinstance(rows, list):
        raise ExperimentError(
            f"The annotation's {where} rows are {type(rows).__name__}, not a list, so "
            "they cannot be matched against the pool."
        )
    by_identity: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        label = f"Annotation {where} row {index}"
        _require_object(row, label)
        _require_keys(row, allowed, allowed, label)
        if not isinstance(row["notes"], str):
            raise ExperimentError(f"{label} notes must be text")
        identity = row[id_key]
        if not isinstance(identity, str) or not identity.strip():
            raise ExperimentError(f"{label} names no {id_key}.")
        if identity not in expected:
            raise ExperimentError(
                f"{label} names {identity!r}, which this pool does not hold. An "
                "annotation row the pool never listed cannot be graded from it."
            )
        if identity in by_identity:
            raise ExperimentError(
                f"{label} repeats {identity!r}, which is already graded above it. A "
                "repeated row is a second reading of one passage and would silently "
                "overwrite the first."
            )
        by_identity[identity] = row
    absent = [identity for identity in expected if identity not in by_identity]
    if absent:
        shown = ", ".join(repr(identity) for identity in absent[:5])
        more = " and more" if len(absent) > 5 else ""
        raise ExperimentError(
            f"The annotation is missing {len(absent)} of the pool's {len(expected)} "
            f"{where} rows: {shown}{more}. A row absent from the file cannot be told "
            "apart from one dropped on purpose, so it is refused."
        )
    return [by_identity[identity] for identity in expected]


def _grade(value: Any, dimension: str, rubric: dict[str, Any], where: str) -> Any:
    """A pending grade, or one this rubric offers. Nothing else is a grade."""

    if value is None:
        return None
    _require_choice_value(value, DIMENSION_CHOICES[dimension], dimension, where)
    offered = [entry["value"] for entry in rubric["dimensions"][dimension]["choices"]]
    if value not in offered:
        raise ExperimentError(
            f"{where} grades {dimension} {value!r}, which this pool's rubric does not "
            "offer. A grade the reader could not have chosen from the rubric in front "
            "of them did not come from that rubric."
        )
    return value


def _require_source_claim(
    pair_id: str,
    relation: str,
    notes: Any,
    sides: tuple[str, str],
    item_grades: dict[str, dict[str, Any]],
) -> None:
    """A corroboration or a contradiction has to rest on the sources themselves.

    Both passages must have been checked in the original, and the reason written
    down. Without this, a machine's suspicion that two passages overlap would be
    filed as a human's finding, which the toolkit's own contract already forbids:
    repeated or near-identical text is a diagnostic.
    """

    if not isinstance(notes, str) or not notes.strip():
        raise ExperimentError(
            f"Annotation pair {pair_id!r} is graded {relation!r} with no note. A claim "
            "that two sources agree or disagree is a claim about sources, and the "
            "reason it was reached is what makes it auditable."
        )
    unchecked = [
        identifier
        for identifier in sides
        if item_grades[identifier]["source_verified"] is not True
    ]
    if not unchecked:
        return
    subject = "is" if len(unchecked) == 1 else "are"
    raise ExperimentError(
        f"Annotation pair {pair_id!r} is graded {relation!r} while "
        f"{', '.join(repr(identifier) for identifier in unchecked)} {subject} not "
        "checked in the original. Wording shared or differing between two extractions "
        "is a diagnostic; confirming what the sources say is the reader's work."
    )


def _require_object(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise ExperimentError(
            f"{where} is {type(value).__name__}, not an object, so it cannot be read "
            "against this contract."
        )


def _require_keys(
    record: dict[str, Any],
    allowed: tuple[str, ...],
    required: tuple[str, ...],
    where: str,
) -> None:
    unknown = sorted(set(record) - set(allowed))
    if unknown:
        raise ExperimentError(
            f"{where} carries {', '.join(repr(key) for key in unknown)}, which this "
            "contract does not define. A key naming an engine, an ordering, or a "
            "result would put it in front of the reader, so an unexpected key is "
            "refused rather than ignored."
        )
    absent = [key for key in required if key not in record]
    if absent:
        raise ExperimentError(
            f"{where} is missing {', '.join(repr(key) for key in absent)}. A record "
            "stating no such field cannot be read as agreeing about it."
        )


def _require_version(record: dict[str, Any], expected: int, where: str) -> None:
    version = record.get("schema_version")
    if isinstance(version, bool) or version != expected:
        raise ExperimentError(
            f"{where} declares version {version!r} and this toolkit reads {expected}. "
            "The two are different records, so neither can be read as the other."
        )


def _require_opaque(record: dict[str, Any], key: str, where: str) -> None:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"{where} states no {key}.")
    if len(value) > _OPAQUE_ID_MAX_LENGTH:
        raise ExperimentError(
            f"{where} states a {key} of {len(value)} characters. An identifier in a "
            "blinded pool is an opaque token, and one long enough to carry a path or "
            "a raw identifier is not opaque."
        )
    if not _OPAQUE_ID.fullmatch(value):
        raise ExperimentError(
            f"{where} states {key} {value!r}, which is not an opaque token. A blinded "
            "identifier is letters, digits, underscore, and dash, so it cannot carry a "
            "path separator, a raw query or chunk identifier, or markup into the "
            "review page."
        )


def _require_text(value: Any, key: str, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(
            f"{where} states no {key}. Empty text is not a passage, a query, or a "
            "link, and a reader cannot be shown it."
        )


def _require_digest(value: Any, where: str) -> None:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise ExperimentError(
            f"{where} is {value!r}, which is not a sha256 digest of the bytes it "
            "names. A record bound to a digest nobody computed is bound to nothing."
        )


def _short(digest: Any) -> str:
    """The head of a digest, so a refusal names which file it means."""

    if isinstance(digest, str) and digest:
        return digest[:16]
    return repr(digest)


__all__ = [
    "ANNOTATION_KEYS",
    "ANNOTATION_SCHEMA_VERSION",
    "DIMENSIONS",
    "DIMENSION_CHOICES",
    "ITEM_ROW_KEYS",
    "PAIR_ROW_KEYS",
    "POOL_KEYS",
    "POOL_SCHEMA_VERSION",
    "RELATION_CHOICES",
    "RELEVANCE_CHOICES",
    "RUBRIC",
    "SOURCE_CLAIM_RELATIONS",
    "SOURCE_KEYS",
    "STATUS_ADJUDICATION_REQUIRED",
    "STATUS_COMPLETE",
    "STATUS_INCOMPLETE",
    "UNCERTAIN_VALUE",
    "USABILITY_CHOICES",
    "canonical_digest",
    "make_template",
    "validate_annotations",
    "validate_pool",
]
