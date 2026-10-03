"""The offline page an author grades on, built from nothing but a blinded pool.

The page is one self-contained HTML file. It is opened by hand from the bundle
directory, it reaches nothing, and it stores nothing: there is no server, no
`fetch`, no stylesheet or script from elsewhere, and no browser storage. Every
grade lives in memory until the reader saves, which is why the page says so and
warns before a close that would lose the work. The bundle's own files are read by
the browser when the reader clicks an original link, and nothing else is read.

Everything on it comes from two validated records: the blinded pool, and that
pool's pending annotation template. Both are embedded as JSON, so the page grades
exactly the bytes it was built from and cannot drift from them. A passage, a
query, or a bibliographic fact is set with `textContent` and never as markup, so
a pool carrying `</script>` in a passage shows those characters rather than
starting an element. The embedded JSON escapes `<`, `>`, `&`, and the two line
separators JavaScript treats as line ends, which is what keeps a passage from
closing the data block it sits in. The original link is re-checked in the page
before it is set, so a link cannot become a scheme or climb out of the bundle.

What the page withholds is what the pool withheld. A card is a passage beside a
question with its source facts, or two passages of one question side by side, and
nothing on it says which produced either passage. A repeated passage is rendered
like any other, because the builder decides what to hide and a page that marked
repeats would hand back the grouping it was built to remove.

What it cannot do:

- It cannot confirm that an original says what an extraction shows. It offers the
  file and a three-state choice of pending, not checked, or checked, and the
  choice is the reader's.
- It cannot adjudicate. A grade marked uncertain is reported as wanting a second
  reader, and the page does not offer to be one.
- Its completeness check is a mirror of `schema.validate_annotations` in
  JavaScript, so the rules are written twice. The toolkit's own reading of the
  saved file is the one that counts; this copy is for the reader while they work.
- Nothing leaves the page except the JSON the reader saves.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import SOURCE_CLAIM_RELATIONS, make_template, validate_pool

#: The characters that would otherwise let a value close the element it is embedded
#: in. `<` is the one that matters: it ends the data block for a script element. `>`
#: and `&` are escaped with it so no partial tag can be assembled, and the two
#: Unicode line separators are escaped because JavaScript ends a string literal at
#: them even though JSON does not.
_JSON_ESCAPES = (
    ("&", "\\u0026"),
    ("<", "\\u003c"),
    (">", "\\u003e"),
    (" ", "\\u2028"),
    (" ", "\\u2029"),
)

#: The token in the page that the embedded records replace.
_DATA_TOKEN = "__REVIEW_DATA__"

#: The relations this page requires an original and a note for. It reads the same
#: list the validator refuses without, so a page and a toolkit judge one claim the
#: same way.
_SOURCE_CLAIMS = list(SOURCE_CLAIM_RELATIONS)


def render_review(pool: Any, *, pool_sha256: str) -> str:
    """The review page for one pool, as one HTML file to be opened by hand.

    The pool is validated first, so a broken reference or an original link that
    leaves the bundle is refused here rather than rendered into a page a reader
    would then trust. The embedded template is that pool's pending annotation:
    every item and every pair has a row, every grade is null, and the rubric
    acknowledgement is false until the reader ticks it.
    """

    validate_pool(pool)
    template = make_template(pool, pool_sha256=pool_sha256)
    return REVIEW_DOCUMENT.replace(
        _DATA_TOKEN, _embed({"pool": pool, "template": template})
    )


def _embed(payload: dict[str, Any]) -> str:
    """Serialize records for a script element, with nothing able to close it."""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for character, escape in _JSON_ESCAPES:
        text = text.replace(character, escape)
    return text


#: The page. Static text with one substitution, so that nothing in it can be built
#: out of a pool's own values.
REVIEW_DOCUMENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Blinded pool review</title>
<style>
:root {
  color-scheme: light dark;
  --edge: #8884;
  --ink: #111;
  --paper: #fff;
  --muted: #555;
  --flag: #7a4b00;
}
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  padding: 1.5rem 1rem 4rem;
  max-width: 60rem;
  font: 16px/1.55 system-ui, sans-serif;
  color: var(--ink);
  background: var(--paper);
}
h1 { font-size: 1.4rem; margin: 0 0 0.5rem; }
h2 { font-size: 1.1rem; margin: 0 0 0.5rem; }
h3 { font-size: 0.95rem; margin: 0.8rem 0 0.3rem; }
section {
  border: 1px solid var(--edge);
  border-radius: 6px;
  padding: 0.9rem 1rem;
  margin: 0 0 1rem;
}
p { margin: 0 0 0.5rem; }
.notice { color: var(--flag); font-weight: 600; }
.scope, .hint { color: var(--muted); font-size: 0.9rem; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.1rem 0.6rem;
     margin: 0.4rem 0 0.6rem; font-size: 0.88rem; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
ol, ul { margin: 0.3rem 0; padding-left: 1.2rem; }
li { margin: 0.15rem 0; }
.ack { display: block; margin-top: 0.8rem; padding: 0.5rem 0.6rem;
       border: 1px solid var(--edge); border-radius: 4px; }
.controls { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.controls input { font: inherit; padding: 0.3rem 0.4rem; border: 1px solid var(--edge);
                  border-radius: 4px; background: transparent; color: inherit; }
#search { flex: 1 1 12rem; }
#annotator { width: 100%; }
button {
  font: inherit; padding: 0.35rem 0.7rem; border: 1px solid var(--edge);
  border-radius: 4px; background: transparent; color: inherit; cursor: pointer;
}
button[aria-pressed="true"] { border-color: currentColor; font-weight: 600; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.progress { font-size: 0.88rem; color: var(--muted); }
#where { margin-left: auto; }
.passage { margin: 0.4rem 0; padding: 0.6rem 0.8rem; border-left: 3px solid var(--edge);
           white-space: pre-wrap; overflow-wrap: anywhere; }
.question { font-weight: 600; overflow-wrap: anywhere; }
.sides { display: grid; gap: 0.8rem; }
@media (min-width: 44rem) { .sides { grid-template-columns: 1fr 1fr; } }
.sides h4 { margin: 0 0 0.2rem; font-size: 0.85rem; color: var(--muted); }
.labels { display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.6rem 0 0.4rem; }
.field { display: flex; flex-direction: column; gap: 0.15rem; }
.field-label { font-size: 0.8rem; color: var(--muted); }
select, textarea, input[type="text"], input[type="search"] {
  font: inherit; color: inherit; background: transparent;
  border: 1px solid var(--edge); border-radius: 4px; padding: 0.25rem 0.35rem;
}
textarea { width: 100%; min-height: 3.5rem; resize: vertical; }
a { color: inherit; }
pre { overflow-wrap: anywhere; white-space: pre-wrap; margin: 0.4rem 0 0;
      font-size: 0.85rem; }
.hidden { display: none; }
.status { font-weight: 600; }
.handover { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; }
</style>
</head>
<body>

<header>
  <h1>Blinded pool review</h1>
  <p class="notice" id="extraction"></p>
  <p class="scope" id="scope"></p>
  <p class="hint" id="identity"></p>
</header>

<section id="rubric-section">
  <h2>Rubric</h2>
  <p class="hint">How to read the cards.</p>
  <ul id="rubric-instructions"></ul>
  <div id="rubric-dimensions"></div>
  <h3>Rules</h3>
  <ul id="rubric-rules"></ul>
  <label class="ack">
    <input type="checkbox" id="ack">
    <span id="ack-text"></span>
  </label>
  <p class="hint">
    Marking is allowed before you tick this. A complete annotation needs the tick,
    because a grade means what the rubric in front of you meant.
  </p>
</section>

<section class="controls">
  <button id="mode-items" type="button" aria-pressed="true">Passages</button>
  <button id="mode-pairs" type="button" aria-pressed="false">Pairs</button>
  <input type="search" id="search" placeholder="Search the text on this page">
  <button id="prev" type="button">Previous</button>
  <button id="next" type="button">Next</button>
  <button id="all" type="button">Show all</button>
  <span class="progress" id="progress"></span>
  <span class="progress" id="where"></span>
</section>

<section>
  <label class="hint" for="annotator">Annotator</label>
  <input type="text" id="annotator" placeholder="Your name, so the reading can be attributed">
</section>

<main id="cards"></main>

<section id="handover">
  <h2>Save and reopen</h2>
  <p class="hint" id="storage-note"></p>
  <div class="handover">
    <button id="save-partial" type="button">Save what is graded so far</button>
    <button id="save-complete" type="button" disabled>Hand over a graded file</button>
    <label class="hint" for="import">Reopen a saved file</label>
    <input type="file" id="import" accept="application/json,.json">
  </div>
  <p class="status" id="status"></p>
  <pre id="report"></pre>
</section>

<script type="application/json" id="review-data">__REVIEW_DATA__</script>
<script>
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("review-data").textContent);
  var POOL = DATA.pool;
  var STATE = DATA.template;
  var RUBRIC = POOL.rubric;
  var DIMENSIONS = RUBRIC.dimensions;
  var ITEM_LABELS = ["relevance", "usability", "source_verified"];
  var PAIR_LABELS = ["relation"];
  var SOURCE_CLAIMS = ["contradiction", "independent_corroboration"];
  var UNCERTAIN = "uncertain";

  var questionText = {};
  POOL.questions.forEach(function (question) {
    questionText[question.question_id] = question.query;
  });
  var itemById = {};
  POOL.items.forEach(function (item) { itemById[item.item_id] = item; });
  var itemRow = {};
  STATE.items.forEach(function (row) { itemRow[row.item_id] = row; });
  var pairRow = {};
  STATE.pairs.forEach(function (row) { pairRow[row.pair_id] = row; });
  var pairSides = {};
  POOL.pairs.forEach(function (pair) {
    pairSides[pair.pair_id] = [pair.left_item_id, pair.right_item_id];
  });

  var view = "items";
  var shown = [];
  var position = 0;
  var dirty = false;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function byId(name) { return document.getElementById(name); }

  function option(value, label) {
    var node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    return node;
  }

  function rawOf(value) {
    if (value === null || value === undefined) { return ""; }
    return String(value);
  }

  function gradeFrom(raw) {
    if (raw === "") { return null; }
    if (raw === "true") { return true; }
    if (raw === "false") { return false; }
    return raw;
  }

  function text(value) {
    if (typeof value !== "string") { return ""; }
    return value;
  }

  function locatorText(value) {
    if (Object.prototype.toString.call(value) === "[object Array]") {
      return value.join(", ");
    }
    return String(value);
  }

  function safeOriginal(value) {
    if (typeof value !== "string" || value === "") { return null; }
    if (value.indexOf(":") >= 0 || value.indexOf("\\") >= 0) { return null; }
    if (value.charAt(0) === "/") { return null; }
    if (value.indexOf('"') >= 0 || value.indexOf("'") >= 0) { return null; }
    if (value.indexOf("<") >= 0 || value.indexOf(">") >= 0) { return null; }
    var parts = value.split("/");
    for (var index = 0; index < parts.length; index += 1) {
      if (parts[index] === "" || parts[index] === "." || parts[index] === "..") {
        return null;
      }
    }
    return value;
  }

  function sourceBlock(source) {
    var block = el("div", "source");
    var list = el("dl", "bib");
    ["title", "authors", "year", "doi", "source_relative_path"].forEach(
      function (key) {
        if (text(source[key]) === "") { return; }
        list.appendChild(el("dt", null, key.replace(/_/g, " ")));
        list.appendChild(el("dd", null, text(source[key])));
      }
    );
    var locator = source.locator || {};
    Object.keys(locator).forEach(function (key) {
      list.appendChild(el("dt", null, "locator " + key));
      list.appendChild(el("dd", null, locatorText(locator[key])));
    });
    block.appendChild(list);
    var safe = safeOriginal(source.original);
    if (safe === null) {
      block.appendChild(el("span", "hint", "No original file is linked here."));
    } else {
      var link = el("a", "original", "Open the original file");
      link.href = safe;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      block.appendChild(link);
    }
    block.appendChild(el("p", "hint",
      "This text was extracted. Only the original says what the source says."));
    return block;
  }

  function labelField(dimension, row, key) {
    var wrap = el("div", "field");
    wrap.appendChild(el("label", "field-label", DIMENSIONS[dimension].label));
    var select = el("select", "field-select");
    select.appendChild(option("", "Pending"));
    DIMENSIONS[dimension].choices.forEach(function (choice) {
      select.appendChild(option(String(choice.value), choice.label));
    });
    select.value = rawOf(row[key]);
    select.addEventListener("change", function () {
      row[key] = gradeFrom(select.value);
      dirty = true;
      refresh();
    });
    wrap.appendChild(select);
    return wrap;
  }

  function noteField(row) {
    var wrap = el("div", "field");
    wrap.appendChild(el("label", "field-label", "Why, if it is not obvious"));
    var notes = el("textarea", "field-notes");
    notes.value = text(row.notes);
    notes.addEventListener("input", function () {
      row.notes = notes.value;
      dirty = true;
      refresh();
    });
    wrap.appendChild(notes);
    return wrap;
  }

  function buildItemCard(item) {
    var row = itemRow[item.item_id];
    var card = el("section", "card");
    card.appendChild(el("h3", null, "Passage " + item.item_id));
    card.appendChild(el("p", "question", text(questionText[item.question_id])));
    card.appendChild(el("blockquote", "passage", text(item.passage)));
    card.appendChild(sourceBlock(item.source));
    var labels = el("div", "labels");
    ITEM_LABELS.forEach(function (key) {
      labels.appendChild(labelField(key, row, key));
    });
    card.appendChild(labels);
    card.appendChild(noteField(row));
    return card;
  }

  function buildSide(itemId) {
    var item = itemById[itemId];
    var side = el("div", "side");
    side.appendChild(el("h4", null, "Passage " + itemId));
    side.appendChild(el("blockquote", "passage", text(item.passage)));
    side.appendChild(sourceBlock(item.source));
    return side;
  }

  function buildPairCard(pair) {
    var row = pairRow[pair.pair_id];
    var card = el("section", "card");
    card.appendChild(el("h3", null, "Pair " + pair.pair_id));
    card.appendChild(el("p", "question", text(questionText[pair.question_id])));
    var sides = el("div", "sides");
    sides.appendChild(buildSide(pair.left_item_id));
    sides.appendChild(buildSide(pair.right_item_id));
    card.appendChild(sides);
    var labels = el("div", "labels");
    PAIR_LABELS.forEach(function (key) {
      labels.appendChild(labelField(key, row, key));
    });
    card.appendChild(labels);
    card.appendChild(noteField(row));
    return card;
  }

  function rebuild() {
    var needle = byId("search").value.trim().toLowerCase();
    var container = byId("cards");
    container.textContent = "";
    shown = [];
    if (view === "items") {
      POOL.items.forEach(function (item) {
        if (needle !== "") {
          var itemText = [item.item_id, item.passage,
                          questionText[item.question_id], item.source.title,
                          item.source.authors].map(text).join(" ").toLowerCase();
          if (itemText.indexOf(needle) < 0) { return; }
        }
        container.appendChild(buildItemCard(item));
        shown.push(item.item_id);
      });
    } else {
      POOL.pairs.forEach(function (pair) {
        if (needle !== "") {
          var pairText = [pair.pair_id, itemById[pair.left_item_id].passage,
                          itemById[pair.right_item_id].passage,
                          questionText[pair.question_id]].map(text).join(" ")
            .toLowerCase();
          if (pairText.indexOf(needle) < 0) { return; }
        }
        container.appendChild(buildPairCard(pair));
        shown.push(pair.pair_id);
      });
    }
    position = 0;
    paint();
  }

  function paint() {
    var container = byId("cards");
    for (var index = 0; index < container.children.length; index += 1) {
      container.children[index].classList.toggle("hidden", index !== position);
    }
    byId("prev").disabled = shown.length < 2;
    byId("next").disabled = shown.length < 2;
    byId("where").textContent = shown.length === 0
      ? "nothing matches"
      : (view === "items" ? "passage " : "pair ") + (position + 1) + " of "
        + shown.length;
  }

  function readStatus() {
    var uncertain = 0;
    var pendingItems = 0;
    var pendingPairs = 0;
    STATE.items.forEach(function (row) {
      var decided = 0;
      ITEM_LABELS.forEach(function (key) {
        if (row[key] === null || row[key] === undefined) { return; }
        decided += 1;
        if (row[key] === UNCERTAIN) { uncertain += 1; }
      });
      if (decided < ITEM_LABELS.length) { pendingItems += 1; }
    });
    STATE.pairs.forEach(function (row) {
      if (row.relation === null || row.relation === undefined) {
        pendingPairs += 1;
      } else if (row.relation === UNCERTAIN) {
        uncertain += 1;
      }
    });
    var acknowledged = byId("ack").checked;
    var named = text(STATE.annotator).trim() !== "";
    var status;
    if (uncertain > 0) {
      status = "adjudication_required";
    } else if (pendingItems > 0 || pendingPairs > 0 || !acknowledged || !named) {
      status = "incomplete";
    } else {
      status = "complete";
    }
    return {
      status: status,
      acknowledged: acknowledged,
      named: named,
      pendingItems: pendingItems,
      pendingPairs: pendingPairs,
      uncertain: uncertain,
      gradedItems: STATE.items.length - pendingItems,
      gradedPairs: STATE.pairs.length - pendingPairs
    };
  }

  function report(read) {
    var lines = [];
    lines.push("annotation status: " + read.status);
    lines.push("rubric acknowledged: " + (read.acknowledged ? "yes" : "no"));
    lines.push("annotator named: " + (read.named ? "yes" : "no"));
    lines.push("passages graded: " + read.gradedItems + " of " + STATE.items.length);
    lines.push("pairs graded: " + read.gradedPairs + " of " + STATE.pairs.length);
    lines.push("passages still pending: " + read.pendingItems);
    lines.push("pairs still pending: " + read.pendingPairs);
    lines.push("grades marked uncertain: " + read.uncertain);
    lines.push("");
    lines.push("A finished annotation is a finished reading. It is not a quality result, not a confirmation of any ordering, and not a basis for a policy decision.");
    return lines.join("\n");
  }

  function refresh() {
    var read = readStatus();
    byId("progress").textContent = view === "items"
      ? read.gradedItems + " of " + STATE.items.length + " passages graded"
      : read.gradedPairs + " of " + STATE.pairs.length + " pairs graded";
    var problems = inspect(JSON.parse(exportText()));
    byId("save-complete").disabled = read.status !== "complete" || problems.length > 0;
    byId("status").textContent = problems.length > 0 ? "source_evidence_required" : read.status;
    byId("report").textContent = report(read) + (problems.length > 0
      ? "\n\nBefore handing this reading over:\n" + problems.join("\n") : "");
  }

  function exportText() {
    return JSON.stringify({
      schema_version: STATE.schema_version,
      pool_id: STATE.pool_id,
      pool_sha256: STATE.pool_sha256,
      rubric_sha256: STATE.rubric_sha256,
      rubric_acknowledged: byId("ack").checked,
      annotator: text(STATE.annotator),
      items: STATE.items.map(function (row) {
        return {
          item_id: row.item_id,
          relevance: row.relevance,
          usability: row.usability,
          source_verified: row.source_verified,
          notes: text(row.notes)
        };
      }),
      pairs: STATE.pairs.map(function (row) {
        return { pair_id: row.pair_id, relation: row.relation, notes: text(row.notes) };
      })
    }, null, 2) + "\n";
  }

  function download() {
    var blob = new Blob([exportText()], { type: "application/json" });
    var handle = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = handle;
    link.download = "annotations-" + POOL.pool_id + ".json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(handle);
    dirty = false;
    refresh();
  }

  function has(record, key) {
    return Object.prototype.hasOwnProperty.call(record, key);
  }

  function isList(value) {
    return Object.prototype.toString.call(value) === "[object Array]";
  }

  function unknownKeys(record, allowed) {
    return Object.keys(record).filter(function (key) {
      return allowed.indexOf(key) < 0;
    });
  }

  function inspectRows(rows, identityKey, gradeKeys, label) {
    var reasons = [];
    var allowed = [identityKey].concat(gradeKeys, ["notes"]);
    if (!isList(rows)) {
      return ["the " + label + " rows are missing"];
    }
    var expected = identityKey === "item_id" ? STATE.items.length : STATE.pairs.length;
    if (rows.length !== expected) {
      reasons.push("the file holds " + rows.length + " " + label + " rows and this "
        + "pool has " + expected);
    }
    var seen = {};
    rows.forEach(function (row, index) {
      if (row === null || typeof row !== "object" || isList(row)) {
        reasons.push(label + " row " + (index + 1) + " is not an object");
        return;
      }
      var extra = unknownKeys(row, allowed);
      if (extra.length > 0) {
        reasons.push(label + " row " + (index + 1) + " carries " + extra.join(", ")
          + ", which this page does not read");
      }
      var identity = row[identityKey];
      if (typeof identity !== "string") {
        reasons.push(label + " row " + (index + 1) + " names nothing");
        return;
      }
      if (seen[identity]) {
        reasons.push(label + " row " + (index + 1) + " repeats " + identity);
      }
      seen[identity] = true;
      var inventory = identityKey === "item_id" ? itemRow : pairRow;
      if (!has(inventory, identity)) {
        reasons.push(label + " row " + (index + 1) + " names " + identity
          + ", which this pool does not hold");
      }
      if (typeof row.notes !== "string") {
        reasons.push(identity + " holds notes as " + typeof row.notes
          + ", which is not text");
      }
      gradeKeys.forEach(function (field) {
        var value = row[field];
        if (value === null || value === undefined) { return; }
        var accepted = DIMENSIONS[field].choices.some(function (choice) {
          return choice.value === value;
        });
        if (accepted) { return; }
        reasons.push(identity + " grades " + field + " as " + JSON.stringify(value)
          + ", which this page does not offer");
      });
    });
    return reasons;
  }

  function inspectClaims(pairs, items) {
    var reasons = [];
    if (!isList(pairs) || !isList(items)) { return reasons; }
    var verified = {};
    items.forEach(function (row) {
      if (row !== null && typeof row === "object" && row.source_verified === true) {
        verified[row.item_id] = true;
      }
    });
    pairs.forEach(function (row) {
      if (row === null || typeof row !== "object") { return; }
      if (SOURCE_CLAIMS.indexOf(row.relation) < 0) { return; }
      var sides = pairSides[row.pair_id];
      if (!sides) { return; }
      if (typeof row.notes !== "string" || row.notes.trim() === "") {
        reasons.push("pair " + row.pair_id + " is graded " + row.relation
          + " with no note");
      }
      sides.forEach(function (identifier) {
        if (!verified[identifier]) {
          reasons.push("pair " + row.pair_id + " is graded " + row.relation
            + " while " + identifier + " is not checked in the original");
        }
      });
    });
    return reasons;
  }

  function inspect(document_) {
    var reasons = [];
    var allowed = ["schema_version", "pool_id", "pool_sha256", "rubric_sha256",
                   "rubric_acknowledged", "annotator", "items", "pairs"];
    if (document_ === null || typeof document_ !== "object" || isList(document_)) {
      return ["the file is not a JSON object"];
    }
    if (document_.schema_version !== 1) {
      reasons.push("the file declares version " + document_.schema_version
        + " and this page reads 1");
    }
    var extra = unknownKeys(document_, allowed);
    if (extra.length > 0) {
      reasons.push("the file carries " + extra.join(", ")
        + ", which this page does not read");
    }
    if (document_.pool_id !== STATE.pool_id) {
      reasons.push("the file names pool " + document_.pool_id + " and this page "
        + "shows " + STATE.pool_id);
    }
    if (document_.pool_sha256 !== STATE.pool_sha256) {
      reasons.push("the file was filled in against different pool bytes");
    }
    if (document_.rubric_sha256 !== STATE.rubric_sha256) {
      reasons.push("the file was filled in against a different rubric");
    }
    if (typeof document_.rubric_acknowledged !== "boolean") {
      reasons.push("the file's rubric acknowledgement is not true or false");
    }
    if (typeof document_.annotator !== "string") {
      reasons.push("the file's annotator is not text");
    }
    reasons = reasons.concat(
      inspectRows(document_.items, "item_id", ITEM_LABELS, "item"),
      inspectRows(document_.pairs, "pair_id", PAIR_LABELS, "pair"),
      inspectClaims(document_.pairs, document_.items)
    );
    return reasons;
  }

  function adopt(document_) {
    document_.items.forEach(function (row) {
      var target = itemRow[row.item_id];
      if (!target) { return; }
      ITEM_LABELS.forEach(function (key) { target[key] = row[key]; });
      target.notes = text(row.notes);
    });
    document_.pairs.forEach(function (row) {
      var target = pairRow[row.pair_id];
      if (!target) { return; }
      PAIR_LABELS.forEach(function (key) { target[key] = row[key]; });
      target.notes = text(row.notes);
    });
    STATE.annotator = text(document_.annotator);
    byId("annotator").value = STATE.annotator;
    byId("ack").checked = document_.rubric_acknowledged === true;
  }

  function fillRubric() {
    document.title = "Blinded pool review " + POOL.pool_id;
    byId("identity").textContent = "Pool " + POOL.pool_id + ", role " + POOL.role
      + ": " + POOL.questions.length + " questions, " + POOL.items.length
      + " passages, " + POOL.pairs.length + " pairs. The identifiers here are opaque "
      + "and say nothing about where a passage came from.";
    byId("extraction").textContent = text(RUBRIC.extraction_notice);
    byId("scope").textContent = text(RUBRIC.scope);
    RUBRIC.instructions.forEach(function (entry) {
      byId("rubric-instructions").appendChild(el("li", null, entry));
    });
    RUBRIC.rules.forEach(function (entry) {
      byId("rubric-rules").appendChild(el("li", null, entry));
    });
    ["relevance", "usability", "relation", "source_verified"].forEach(
      function (name) {
        var dimension = DIMENSIONS[name];
        var block = el("div", "dimension");
        block.appendChild(el("h3", null, text(dimension.label)));
        block.appendChild(el("p", "hint", text(dimension.question)));
        var list = el("dl", "bib");
        dimension.choices.forEach(function (choice) {
          list.appendChild(el("dt", null, text(choice.label)));
          list.appendChild(el("dd", null, text(choice.definition)));
        });
        block.appendChild(list);
        byId("rubric-dimensions").appendChild(block);
      }
    );
    byId("ack-text").textContent =
      "I have read this rubric, and I am grading the passages against it.";
    byId("storage-note").textContent =
      "This page stores nothing. Your grades stay in this tab until you save a file, "
      + "and closing it loses them. It reads nothing from the network.";
  }

  function setMode(next) {
    view = next;
    byId("mode-items").setAttribute("aria-pressed", String(next === "items"));
    byId("mode-pairs").setAttribute("aria-pressed", String(next === "pairs"));
    rebuild();
    refresh();
  }

  function wire() {
    byId("ack").addEventListener("change", function () {
      STATE.rubric_acknowledged = byId("ack").checked;
      refresh();
    });
    byId("annotator").addEventListener("input", function () {
      STATE.annotator = byId("annotator").value;
      dirty = true;
      refresh();
    });
    byId("search").addEventListener("input", rebuild);
    byId("prev").addEventListener("click", function () {
      if (position > 0) { position -= 1; paint(); }
    });
    byId("next").addEventListener("click", function () {
      if (position < shown.length - 1) { position += 1; paint(); }
    });
    byId("all").addEventListener("click", function () {
      byId("search").value = "";
      rebuild();
    });
    byId("mode-items").addEventListener("click", function () { setMode("items"); });
    byId("mode-pairs").addEventListener("click", function () { setMode("pairs"); });
    byId("save-partial").addEventListener("click", download);
    byId("save-complete").addEventListener("click", download);
    byId("import").addEventListener("change", function () {
      var chosen = byId("import").files;
      if (!chosen || chosen.length === 0) { return; }
      var file = chosen[0];
      file.text().then(function (contents) {
        var parsed = null;
        var reasons = ["the file is not readable JSON"];
        try {
          parsed = JSON.parse(contents);
          reasons = inspect(parsed);
        } catch (error) {
          parsed = null;
        }
        if (reasons.length > 0 || parsed === null) {
          byId("report").textContent = "This file was not loaded. It describes a "
            + "reading this page cannot accept:\n\n" + reasons.join("\n")
            + "\n\nNothing on this page was changed.";
          return;
        }
        adopt(parsed);
        dirty = true;
        rebuild();
        refresh();
        byId("report").textContent = report(readStatus())
          + "\n\nLoaded from " + file.name + ". Pending values were restored as pending.";
      });
      byId("import").value = "";
    });
    window.addEventListener("beforeunload", function (event) {
      if (!dirty) { return; }
      event.preventDefault();
      event.returnValue = "";
    });
  }

  fillRubric();
  wire();
  rebuild();
  refresh();
}());
</script>
</body>
</html>
"""


__all__ = ["REVIEW_DOCUMENT", "render_review"]
