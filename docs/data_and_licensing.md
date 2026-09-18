# Data, models and licensing

This repository contains original code and documentation (MIT licence, see `LICENSE`)
plus results produced entirely inside the CARLA simulator. It deliberately excludes
everything derived from the NVIDIA PhysicalAI-AV dataset. The table lists each external
component, its licence, and what that means for what is (not) published here.

| Component | Licence | Consequence for this repository |
|---|---|---|
| [Alpamayo-1.5 code](https://github.com/NVlabs/alpamayo1.5) (`alpamayo1_5` package) | Apache-2.0 | Used unmodified as a third-party dependency (cloned at a pinned commit by the notebooks; not vendored). `src/acarla/adapter/lens.py` re-implements the F-Theta camera model's few lines of arithmetic and is checked against the original in an oracle test; attribution in `NOTICE`. |
| [Alpamayo-1.5-10B weights](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | OpenMDW-1.1 | Downloaded at run time by the Kaggle worker. Model outputs on CARLA imagery (trajectories, reasoning text) are published in `results/`. |
| [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) (backbone) | NVIDIA Open Model License, gated | Access must be requested on Hugging Face; the token is read from a Kaggle secret and never stored in this repository. |
| [PhysicalAI-AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | NVIDIA Autonomous Vehicle Dataset License, gated | Section 4.6 prohibits distributing any part of the dataset; section 3 treats benchmarking results on it as confidential. **Not published:** the calibration tables (`golden/m2_reference/*.csv`), the generated rig file `configs/rig_alpamayo.yaml`, per-run `rig_used.json` files, the reference camera frame, the M0/M2 golden files, and the minADE figure of the M0 feasibility run. |
| [CARLA 0.9.16](https://carla.org) | MIT (simulator), assets CC-BY | Recordings, renders and ground truth in `results/` and `docs/img/` are CARLA content. |

## What you need to reproduce the dataset-dependent parts

1. Accept the dataset and backbone licences on Hugging Face and create a read token.
2. Run `notebooks/kaggle_m2_reference_dump.ipynb` (CPU kernel) with that token as the
   Kaggle secret `huggingface`. It writes the calibration tables of the reference clip and
   the upstream loader's tensors to `m2_reference/`.
3. Copy the `calibration_*.csv` files to `golden/m2_reference/` and run
   `python scripts/calibrate_rig.py --calibration-dir golden/m2_reference --out configs/rig_alpamayo.yaml`.
4. `pytest` now also executes the calibration, lens and coordinate oracle tests that are
   skipped in the public checkout.

`configs/rig_provisional.yaml` is an estimated four-camera rig that lets the recorder,
viewer and controller run without the dataset; it must not be mistaken for the
calibrated rig.

## Development history

The public history starts with a curated snapshot. The complete development history
(about 55 commits over two weeks) contains dataset-derived files in almost every revision
and is kept private for that reason; the milestone documents in `docs/` preserve the
chronology, including the failed attempts.
