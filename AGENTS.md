# Repository Guidelines

## Project Structure & Module Organization

This repository implements MoMask text-to-motion generation in Python/PyTorch. Top-level entry scripts handle major workflows: `gen_t2m.py` for generation, `edit_t2m.py` for temporal inpainting, `train_vq.py`, `train_t2m_transformer.py`, and `train_res_transformer.py` for training, and `eval_t2m_vq.py` / `eval_t2m_trans_res.py` for evaluation. Core packages are organized by concern: `models/` contains VQ and transformer networks, `data/` and `motion_loaders/` define datasets and loaders, `options/` stores CLI argument parsing, `utils/` and `common/` contain shared helpers, and `visualization/` contains rendering utilities. Runtime assets live in `assets/`, `example_data/`, `dataset/`, `checkpoints/`, and `generation/`.

## Build, Test, and Development Commands

- `conda env create -f environment.yml && conda activate momask`: create the tested Python 3.7/PyTorch environment.
- `pip install -r requirements.txt`: alternative pip setup, tested with newer Python versions.
- `bash prepare/download_models.sh`: download pretrained models for generation.
- `bash prepare/download_evaluator.sh && bash prepare/download_glove.sh`: download evaluation dependencies.
- `python gen_t2m.py --gpu_id 0 --ext exp1 --text_prompt "A person is running."`: run a quick text-to-motion generation smoke test.
- `python train_vq.py --name rvq_name --gpu_id 0 --dataset_name t2m --batch_size 256`: start RVQ training; train this before transformer stages.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python import grouping: standard library, third-party libraries, then local modules. Prefer `snake_case` for functions, variables, and CLI option names; use `CamelCase` for model and trainer classes. Keep entry scripts argument-driven through `options/` parsers, and place reusable logic in package modules rather than duplicating it across scripts. Follow existing path conventions such as `./checkpoints/<dataset>/<name>` and `./generation/<ext>`.

## Testing Guidelines

No dedicated test suite is currently included. Validate changes with the smallest relevant workflow: generation for inference changes, `eval_t2m_vq.py` for VQ reconstruction changes, or `eval_t2m_trans_res.py` for transformer changes. When adding tests, place them under a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

Recent commits use short, sentence-case messages such as `Update README.md` and `Fix evaluation bug in train_vq`; keep messages concise and action-oriented. Pull requests should include a clear description, affected scripts or modules, setup/data requirements, and the exact command used for validation. Include generated videos, plots, or metric summaries when behavior changes.

## Security & Configuration Tips

Do not commit downloaded datasets, model checkpoints, credentials, or large generated outputs. Keep machine-specific paths out of source files; pass them through CLI arguments or local configuration instead.
