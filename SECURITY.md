# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Report vulnerabilities privately via
[private vulnerability reporting](https://github.com/ms-kumar/torml/security/advisories/new).
Do not open public issues for unpatched vulnerabilities.

Please include:

- Affected version(s) (`torml.__version__`, `uv.lock` entries if dependency-related)
- Steps to reproduce or proof of concept
- Impact assessment, if known

Expect an initial response within 7 days. Fixes land as a patch release with
a `RELEASES.md` entry crediting the reporter (unless anonymity is requested).

## Scope Notes

- `torml` has one runtime dependency (`torch`); most advisories will concern
  the lockfile (`uv.lock`) or CI tooling, and are fixed by version bumps.
- GitHub secret scanning and push protection are enabled; CI runs
  CodeQL, Dependabot alerts, `pip-audit`-clean installs, and a pinned
  (`pre-commit` + `uv.lock`) toolchain.
