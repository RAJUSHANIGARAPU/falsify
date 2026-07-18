# Contributing to falsify

Thanks for your interest. `falsify` is a small, focused library — the goal is that it
stays correct, honest, and dependency-free. A few constraints keep it that way; please
read them before opening a PR.

## Non-negotiable design constraints

- **Zero runtime dependencies.** The library must import and run on a bare Python
  standard library. A PR that adds a runtime dependency will not be merged — if a
  feature seems to need one, open an issue first so we can discuss an alternative.
  (Test-only dev tools under `[project.optional-dependencies].dev` are fine.)
- **Python 3.9+.** CI runs the suite on 3.9–3.13; keep syntax and stdlib usage within
  that range.
- **Statistical claims must be sourced.** Any new statistical method should cite the
  paper or reference it implements (e.g. the Deflated Sharpe Ratio follows Bailey &
  López de Prado; the HAC estimator follows Newey-West). Correctness here is the whole
  point of the library.
- **Fail loudly on bad input.** Prefer raising on degenerate input over returning `NaN`
  or a silently-wrong number. Keep the misuse-resistant API shape (the `*_from_returns`
  entry points exist so callers can't make the annualized-Sharpe footgun).

## Development setup

```bash
git clone https://github.com/RAJUSHANIGARAPU/falsify
cd falsify
pip install -e ".[dev]"
pytest
```

## Submitting a change

1. Fork and create a branch from `main`.
2. Add or update tests for your change — numerical code needs regression tests, and
   invariants (e.g. monotonicity of a haircut) make good property tests.
3. Run `pytest` locally; make sure it passes on your Python version.
4. Keep the diff focused; one logical change per PR with a clear description of *what*
   and *why*.
5. Open the PR. CI runs the test matrix (3.9–3.13) and builds/validates the package.

## Releases

Releases are automated — see the **Releasing** section in the README. In short: bump
`version` in `pyproject.toml`, add a `CHANGELOG.md` entry, tag, and publish a GitHub
Release; the `Publish to PyPI` workflow uploads the distribution via trusted publishing.

## Reporting issues

Bugs, incorrect results, and unclear docs are all worth an issue. For a numerical bug,
include the inputs and the expected vs. actual output so it can be reproduced.
