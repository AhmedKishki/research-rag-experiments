"""Smoke-test the offline page in a temporary headless browser profile."""

import copy
import json
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from rag_experiments.annotation.review import render_review
from rag_experiments.annotation.schema import RUBRIC, validate_annotations


def test_offline_browser_initialization_grading_and_export(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("A local headless browser is optional for the review smoke test")
    source = {
        "title": "Fixture",
        "locator": {"type": "pdf_page", "page": 1},
        "original": "originals/fixture.pdf",
    }
    pool = {
        "schema_version": 1,
        "pool_id": "pool-browser",
        "role": "exploratory",
        "rubric": copy.deepcopy(RUBRIC),
        "questions": [{"question_id": "Q-a", "query": "What is stated?"}],
        "items": [
            {
                "item_id": "I-a",
                "question_id": "Q-a",
                "passage": "<img src='https://example.invalid/' onerror='window.PWNED=true'> First claim.",
                "source": source,
            },
            {
                "item_id": "I-b",
                "question_id": "Q-a",
                "passage": "Second claim.",
                "source": source,
            },
        ],
        "pairs": [
            {
                "pair_id": "P-a",
                "question_id": "Q-a",
                "left_item_id": "I-a",
                "right_item_id": "I-b",
            }
        ],
    }
    page = render_review(pool, pool_sha256="a" * 64)
    exercise = """
<pre id="browser-result"></pre>
<script>
(async function () {
  const result = {initial: document.getElementById('status').textContent};
  document.getElementById('ack').checked = true;
  document.getElementById('ack').dispatchEvent(new Event('change'));
  result.incompleteDisabled = document.getElementById('save-complete').disabled;
  const annotator = document.getElementById('annotator');
  annotator.value = 'synthetic test'; annotator.dispatchEvent(new Event('input'));
  function chooseAll() {
    for (const select of document.querySelectorAll('#cards select')) {
      select.value = select.options[1].value;
      select.dispatchEvent(new Event('change'));
    }
  }
  chooseAll(); document.getElementById('next').click(); chooseAll();
  document.getElementById('mode-pairs').click(); chooseAll();
  result.complete = document.getElementById('status').textContent;
  result.completeEnabled = !document.getElementById('save-complete').disabled;
  result.problems = document.getElementById('report').textContent;
  result.xss = Boolean(window.PWNED);
  const oldClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () { if (!this.download) oldClick.call(this); };
  URL.createObjectURL = function (blob) {
    blob.text().then(async text => {
      result.annotation = JSON.parse(text);
      async function importReading(reading) {
        const input = document.getElementById('import');
        Object.defineProperty(input, 'files', {configurable: true, value: [
          {name: 'synthetic.json', text: () => Promise.resolve(JSON.stringify(reading))}
        ]});
        input.dispatchEvent(new Event('change'));
        await Promise.resolve(); await Promise.resolve();
      }
      await importReading(result.annotation);
      result.reopenedComplete = document.getElementById('status').textContent;
      const wrongBoolean = JSON.parse(text);
      wrongBoolean.items[0].source_verified = 'false';
      await importReading(wrongBoolean);
      result.rejectsStringBoolean = document.getElementById('report').textContent.includes('not loaded');
      const wrongInventory = JSON.parse(text);
      wrongInventory.items[0].item_id = 'P-a';
      await importReading(wrongInventory);
      result.rejectsOtherInventory = document.getElementById('report').textContent.includes('not loaded');
      const pending = JSON.parse(text);
      pending.annotator = ''; pending.rubric_acknowledged = false;
      for (const row of pending.items) {
        row.relevance = null; row.usability = null; row.source_verified = null;
      }
      for (const row of pending.pairs) row.relation = null;
      await importReading(pending);
      result.reopenedPending = document.getElementById('status').textContent;
      document.getElementById('browser-result').textContent = JSON.stringify(result);
    });
    return 'blob:synthetic';
  };
  URL.revokeObjectURL = function () {};
  document.getElementById('save-partial').click();
})();
</script>
"""
    path = tmp_path / "review.html"
    path.write_text(page.replace("</body>", exercise + "</body>"), encoding="utf-8")
    completed = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--user-data-dir=" + str(tmp_path / "browser-profile"),
            "--virtual-time-budget=3000",
            "--dump-dom",
            path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-1000:]

    class ResultParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.active = False
            self.value = ""

        def handle_starttag(self, tag, attrs):
            if dict(attrs).get("id") == "browser-result":
                self.active = True

        def handle_endtag(self, tag):
            if tag == "pre":
                self.active = False

        def handle_data(self, data):
            if self.active:
                self.value += data

    parser = ResultParser()
    parser.feed(completed.stdout)
    assert parser.value, "Offline page did not finish the synthetic interaction"
    result = json.loads(parser.value)
    assert result["initial"] == "incomplete"
    assert result["incompleteDisabled"] is True
    assert result["complete"] == "complete", result["problems"]
    assert result["completeEnabled"] is True
    assert result["xss"] is False
    assert result["reopenedComplete"] == "complete"
    assert result["rejectsStringBoolean"] is True
    assert result["rejectsOtherInventory"] is True
    assert result["reopenedPending"] == "incomplete"
    assert (
        validate_annotations(pool, result["annotation"], pool_sha256="a" * 64)["status"]
        == "complete"
    )
