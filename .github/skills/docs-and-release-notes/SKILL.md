---
name: docs-and-release-notes
description: 'Use when: updating torml documentation, NumPyDoc docstrings, Sphinx docs, user guide, API reference, examples, RELEASES.md, whats_new, PR docs checklist.'
argument-hint: '<feature or release note>'
---

# Docs and Release Notes

Use this skill when a change affects public API, examples, documentation, or release notes.

## Documentation Procedure

1. Add NumPyDoc/scikit-learn-style docstrings for public classes and functions.
2. Include shapes in parameters and returns.
3. Add small runnable examples where useful.
4. Update `doc/user_guide/` for conceptual/narrative changes.
5. Update `doc/reference/` for public API additions.

## Release Notes Procedure

1. Update `RELEASES.md` for user-visible changes.
2. Group under Added, Changed, Deprecated, Removed, Fixed, or Security.
3. Mention the public symbol path, e.g. `torml.linear_model.LinearRegression`.
4. Keep `doc/whats_new/` as an index/link location if used.

## Docstring Template

Use Parameters, Returns, Raises, Examples, See Also, Notes, and References sections when relevant.
