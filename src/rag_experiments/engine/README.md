# `engine/`

Owns the research-rag tree an experiment measures, and the settings that tree
resolves. It may never copy the app, reimplement a setting, or answer a
question about the engine that the app's own code can answer.

Two things live here because both depend on *which* tree is under test:

- `locate.py` names a tree, its git revision, and whether it is dirty, so a
  record states the engine a number came from.
- `resolve_settings.py` runs inside the tree under test. It is a module the
  parent invokes as a subprocess with that tree first on `PYTHONPATH`, so
  `research_rag` resolves to the patched engine rather than to the installed
  one. That is what makes a code arm's new settings, and its changed defaults,
  visible to the harness instead of being validated against the wrong registry.
