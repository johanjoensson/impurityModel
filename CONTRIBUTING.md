Fork the project from github. Create a topic branch, issue a pull request, and wait for review.

Before committing, install the git hooks once so Black and cython-lint run on every commit:

    pip install pre-commit
    pre-commit install

The same hooks run in CI (see `.github/workflows/lint.yml`), so running them locally keeps your
commits from failing the lint check.
