# KICKOFF.md — how to drive Claude Code

## 0. Drop these at repo root
`CLAUDE.md` (context, read first) · `Design.md` · `GLOSSARY.md` · `config.py` · the module stubs · `requirements.txt`.

## 1. Smoke-test FIRST (on the MI300X, before any build)
```bash
# use AMD's ROCm PyTorch image, e.g. rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
python -c "import torch; print(torch.cuda.get_device_name(0))"   # → AMD Instinct MI300X
export MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=3                   # the real speed win
python -c "import monai, nibabel; print(monai.__version__)"
```
Load ONE NIfTI, run one fwd/bwd of a tiny 3D conv on GPU. Only then build.

## 2. Build order (also in CLAUDE.md §13)
1. `preprocess.py` + cache.
2. `mind.py` → MIND-only submission via `submit.py` = **the floor on the board**.
3. ⭐ `oracle.py` (simulated L2/L3) → real per-level numbers. Wire MIND through it.
4. `encoder.py` + `train.py` (InfoNCE + `augment.py`, big batch). Run 2–3 aug-strength configs in parallel; pick best by oracle.
5. `fuse.py` (per-level w from oracle) + `infer.py` TTA → submit.
6. `veto.py` + retrieve-then-rerank, oracle-gated → submit.

## 3. Suggested first prompts to Claude Code
- "Read CLAUDE.md and Design.md. Implement `preprocess.py` per its docstring and `config.py`. Add a `--dry-run` that processes 2 volumes and prints shapes."
- "Implement `mind.py` (Heinrich 2012 MIND-SSC), then a script that produces a MIND-only `submission.csv` via `submit.py`. Validate the CSV has 377 rows and correct ranking lengths."
- "Implement `oracle.py`: hold out 70 of the 350 pairs, synthesize fake-L2 and fake-L3 (use `augment.py`), and report MRR per level for a given scorer. Test it with the MIND scorer."
- "Implement `encoder.py` + `train.py` (shared 3D backbone, InfoNCE, AMP, batch from config). Train, then report oracle MRR per level."

## 4. THE ONE RULE
Before every Kaggle submit: run `oracle.py`. **Up → keep & submit. Down → revert.** The number decides.

## 5. Secrets
Never commit `kaggle.json` / the API key; never paste it into shared docs. Simplest: upload the CSV to Kaggle by hand.
