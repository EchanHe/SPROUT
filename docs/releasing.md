# Releasing

`sprout-seg` and `napari-sprout` are released together, with the same version, by `.github/workflows/release.yml`. PyPI trusts that workflow (trusted publishing): no tokens.

## Once (done for 1.1.0)

- PyPI and TestPyPI: a pending publisher per package, repository `EchanHe/SPROUT`, workflow `release.yml`, environments `pypi-sprout-seg`, `pypi-napari-sprout` (PyPI) and `testpypi-sprout-seg`, `testpypi-napari-sprout` (TestPyPI).
- GitHub environment `pypi-sprout-seg` with a required reviewer: the upload to PyPI waits for an approval.
- Zenodo: GitHub integration on for this repository (a DOI per GitHub release).

## Each release

1. Version: `__version__` in `sprout_core/__init__.py` and `napari-sprout/napari_sprout/__init__.py`, and `sprout-seg~=X.Y.0` in `napari-sprout/pyproject.toml` when X.Y changes (`tests/test_packaging.py` checks they agree).
2. `CHANGELOG.md`: `## X.Y.Z (YYYY-MM-DD)` (the workflow refuses "unreleased"); `CITATION.cff`: `version`, `date-released`.
3. Read the release notes of napari (and of numpy, pandas, scikit-image if a major version came out) for removals that affect SPROUT.
4. Merge to `main`; the tests are green.
5. Rehearse on TestPyPI: Actions → release → Run workflow (a version can be uploaded once, also on TestPyPI: to run it again give e.g. `X.Y.Z.post1`). Then in a new environment:
   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "napari-sprout[all]==X.Y.Z"
   sprout --version
   napari          # Plugins → SPROUT
   ```
6. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`. The workflow builds, tests the wheels on three systems, waits for the approval in Actions, uploads `sprout-seg`, then `napari-sprout`, then makes the GitHub release with the CHANGELOG section.
7. Check https://pypi.org/project/sprout-seg/ and https://pypi.org/project/napari-sprout/, and `pip install "napari-sprout[all]"` in a new environment.

An upload to PyPI cannot be undone or replaced: a broken release is fixed with a new version.
