# Contributing

Alpha Engine is a personal project by [@cipher813](https://github.com/cipher813). Contributions are welcome.

## How to contribute

- **Bug reports**: Open an issue with steps to reproduce, expected vs actual behavior, and relevant logs.
- **Bug fixes**: PRs welcome. Include a test that fails before the fix and passes after.
- **Features**: Open an issue first to discuss the approach before writing code.

## Development workflow

1. Fork the repo and create a feature branch from `main`
2. Make changes, add tests
3. Run the test suite: `pytest tests/ -v`
4. Open a PR against `main` with a clear description of what changed and why

## Code standards

- All tests must pass before merging
- No secrets, API keys, or proprietary config in commits (gitleaks pre-commit hook enforced)
- Follow existing patterns — if unsure, check how similar code is structured elsewhere in the repo

## Related repos

This module is part of the [Nous Ergon](https://nousergon.ai) trading system. See the [system overview](https://github.com/cipher813/alpha-engine#readme) for architecture and the [documentation index](https://github.com/cipher813/alpha-engine-docs#readme) for all repos.
