# Changelog

Versioning convention:

- **Cohort releases** use `vYYYY.MM.N` (e.g. `v2026.04.0`) and are
  tagged in git. One cohort = one major release.
- **Patch numbers** (`v2026.04.0.N`) increment for fixes shipped during
  a cohort's window.
- **Internal iterations** (`v1` through `v18` in the development log
  below) are NOT released to students; they document the build journey.

---

## [2026.04.0] - unreleased (first cohort, target ship date)

First public release for Nebius Academy Module 1 Week 6. Everything
below represents the current homework artifact state for the final
pre-cohort version.

### Added

- Added project `.env` with non-secret defaults for model, dataset, training, and output paths.
- Added `.python-version` pinning the project to Python `3.14.4`.
- Added `pyproject.toml` and `uv.lock` for `uv`-managed dependency setup.
- Added `.gitignore` for Python, Jupyter, local environment, model output, and training artifacts.
- Added `README.md` with project overview, environment setup, run order, and expected outputs.
- Added `TIMY_TRANSFORMER_LM.md` with a Mermaid architecture graph and calculation notes for the tiny Transformer language model.
- Added `docs/HOMEWORK_ASSIGNMENT.md` with the homework task statement and grading checklist.
- Added `docs/DEMO_FLOWS.md` with Mermaid diagrams for prompt tuning and LoRA demo flows.
- Added `docs/RUNNING_DEMOS_AND_HOMEWORK.md` with local Apple Silicon and Nebius Cloud execution paths.
- Added `docs/MNIST_PARALLEL_HPO_DEMO.md` with local and Nebius Cloud instructions for the MNIST parallel HPO demo.
- Added `torchvision` to the `uv` environment for the MNIST demo.

### Changed

- Corrected homework documentation to target `src/tiny_transformer_lm.ipynb` instead of the PEFT demo notebook.
- Added VS Code/Jupyter kernel troubleshooting steps to the homework assignment document.
- Anchored generated `/models/` and `/data/` exclusions in `.gitignore`.
