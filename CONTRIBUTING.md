# Contributing to torml

## Development setup

torml uses [uv](https://docs.astral.sh/uv/) to manage the Python interpreter,
virtual environment, and dependencies.

```bash
uv sync --extra dev --extra test
uv run pre-commit install
```

`uv sync` creates `.venv/`, installs the pinned interpreter from
`.python-version` (downloading it automatically if it isn't already on your
machine), and installs `torml` plus the `dev`/`test` extras from
`pyproject.toml`, writing a `uv.lock` lockfile for reproducible installs.
Run project commands through `uv run` (e.g. `uv run pytest`,
`uv run pre-commit run --all-files`) so they execute inside that environment
without needing to `source .venv/bin/activate` first.

`pylint` runs as a pre-commit hook with `language: system`, so it must be able
to import `torml`'s real dependencies (including `torch`). Always run
`uv sync --extra dev --extra test` before `uv run pre-commit`, otherwise the
`pylint` hook will fail with import errors that have nothing to do with your
change.

To build the docs under `doc/`, include the `doc` extra as well
(`uv sync --extra dev --extra test --extra doc`), then run
`uv run sphinx-build -b html doc doc/_build`. Note that plain
`uv sync --extra doc` prunes the `dev`/`test` packages, so always list all
three extras together.

## Source of truth

- `AGENTS.md` and `TORML_CODING_GUIDELINES.md` are the primary references for
  repository structure, estimator API conventions, testing, documentation,
  and release-note policy.
- The `.github/skills/` files (`implement-estimator`, `pytorch-backend`,
  `validation-and-testing`, `docs-and-release-notes`) encode the step-by-step
  procedure for common changes.

## Before submitting

- [ ] Run tests locally: `uv run pytest`
- [ ] Run `uv run pre-commit run --all-files`
- [ ] Add/update tests for new or changed behavior
- [ ] Update documentation for public API changes
- [ ] Update `RELEASES.md` under `## [Unreleased]`

See `.github/pull_request_template.md` for the full PR checklist.
