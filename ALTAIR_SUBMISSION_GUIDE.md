# EBA-FFD v2 on Altair — Submission Guide

Updated per `ALTAIR_COMPLETE_MANUAL.md` — real lessons learned on this same
SVKM cluster. **The most important rule: every path is absolute, and code and
data live in two separate uploaded folders.** Nothing here relies on Jupyter's
working directory.

## 1. What to upload — two SEPARATE folders

```
/data/mpstme-shlocks/EBA-FFD-v2/          <- CODE folder (upload as a zip, unzip)
    run_eba_ffd_altair.ipynb
    config.py, data_loader.py, training_loop.py, client.py, server.py,
    model.py, privacy.py, drift.py, evaluation.py, preprocessing.py,
    ablations.py, baselines.py, scaling.py, reporting.py, ... (all .py files)

/data/mpstme-shlocks/EBA-FFD-data/        <- DATA folder (upload SEPARATELY)
    creditcard.csv
```

Do **not** nest `creditcard.csv` inside the code folder — this manual's
lesson #3 is that separating them and referencing both by absolute path is
what actually works reliably. The code already supports this: `data_loader.py`
reads an `EBA_FFD_DATA_PATH` environment variable (set in the notebook's path
cell) that points at the separate data folder.

If your username or folder names end up different from what's shown above,
that's fine — just update the three path variables in the notebook's second
code cell to match exactly what the Altair **Files** tab shows.

## 2. Login and upload

1. Go to `https://10.126.1.10:4443`, log in.
2. **Files tab** → upload the code zip and the data zip **separately**, then
   unzip each (or unzip via terminal if the portal doesn't auto-unzip).
3. Navigate into each unzipped folder in the Files browser and **copy the
   exact absolute path from the address bar** — this confirms what
   `BASE_PATH` and `DATASET_DIR` must be set to in the notebook.

## 3. Fill in the notebook's path cell

Open `run_eba_ffd_altair.ipynb`'s second code cell (titled "ALTAIR ABSOLUTE
PATHS") and confirm/edit:

```python
ALTAIR_USERNAME = "mpstme-shlocks"
BASE_PATH   = f"/data/{ALTAIR_USERNAME}/EBA-FFD-v2"
DATASET_DIR = f"/data/{ALTAIR_USERNAME}/EBA-FFD-data"
```

This cell **fails loudly with an assertion error** if any path is wrong,
rather than silently proceeding — per the manual's Mistake #6 lesson
("wasted hours debugging file-not-found errors"). Don't skip past a failed
check; fix the path and re-run the cell before continuing.

## 4. Submit the job

In the **Jobs** tab:

1. Click **Jupyter**.
2. Select a PyTorch-capable container image (e.g. `tensorflow_pbs:23.07-tf2-py3`
   — PyTorch is included regardless of the image's name).
3. Set:
   - **Number of Nodes**: 1
   - **Number of GPUs**: 1
   - **Queue**: `workq` (confirm with the lab if a different queue is expected)
   - **Number of Cores**: 8
   - **Amount of Memory (GB)**: 128
   - **Walltime (hrs)**: 48 (full preset is multi-day; reduce to 12 for
     `quick`, 2 for `smoke`)
4. **Job Script**: the absolute path to the notebook, e.g.
   `/data/mpstme-shlocks/EBA-FFD-v2/run_eba_ffd_altair.ipynb`
5. Leave everything else at default.
6. Click **Submit**.

The notebook then runs automatically and unattended — no need to open a
terminal and type commands by hand (this manual's Mistake #5: a manually
typed terminal session dies if the job restarts; a submitted notebook does not).

## 5. Monitor and retrieve results

- Job status moves `Waiting` → `Running`.
- Once running, the notebook executes top to bottom on its own.
- Results land back in the **code** folder (not the data folder):
  - `/data/mpstme-shlocks/EBA-FFD-v2/results/RESULTS.md` — all 5 tables
  - `/data/mpstme-shlocks/EBA-FFD-v2/outputs/plots/` — all figures
  - `/data/mpstme-shlocks/EBA-FFD-v2/outputs/models/` — best model checkpoints
- **Download these before leaving the lab or before the job is deleted** —
  per this manual's Part 9, don't assume the job's storage persists indefinitely.

## Preset options

Notebook's preset cell:

```python
PRESET = "full"  # Options: smoke | quick | full
```

| Preset | Data | Rounds | Seeds | Time on H100 | Use |
|---|---|---|---|---|---|
| `smoke` | 2% | 2 | 1 | ~5 min | Pipeline sanity check only |
| `quick` | 20% | 5 | 2 | ~2-3 h | Trend validation |
| `full` | 100% | 10 | 3 | ~12-24 h | **Reportable — use this** |

## Checklist before you submit

- [ ] Code folder and data folder uploaded **separately** to Altair
- [ ] Confirmed both absolute paths in the Files tab address bar
- [ ] Updated `ALTAIR_USERNAME` / `BASE_PATH` / `DATASET_DIR` in the notebook if they differ from the defaults
- [ ] `PRESET = "full"`
- [ ] Job submitted with **Job Script** = absolute path to the `.ipynb`, not a drag-selected file
- [ ] Walltime ≥ 48h, memory ≥ 128GB, 1 GPU

## Troubleshooting (from lessons learned on this cluster)

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError` in the paths cell | `BASE_PATH`/`DATASET_DIR` don't match Altair's actual layout | Re-check the Files tab address bar, update the two variables |
| `FileNotFoundError` on `creditcard.csv` | Data folder path wrong, or CSV not actually inside it | Verify `DATASET_DIR` contains `creditcard.csv` directly (not nested another level) |
| `ModuleNotFoundError` for `config`/`data_loader`/etc. | `BASE_PATH` doesn't point at the code folder, or `.py` files didn't upload | Re-check `BASE_PATH`; confirm `.py` files are visible in that folder in Files tab |
| `DataLoader will create N workers — not allowed in Jupyter` | Shouldn't happen — this project's code already uses `num_workers=0` everywhere | If it does appear, it means a dependency changed a default; report it, don't silently increase workers |
| Job stuck on `Waiting` | Queue backed up | Normal — wait, or ask the lab |
| Out of memory | Batch size or memory allocation too low for `full` preset | Raise "Amount of Memory (GB)" to 256 |

## When results are ready

1. Download `results/RESULTS.md` and `outputs/plots/` from the **code** folder.
2. These are now GPU-verified, full-scale, reportable numbers — supersede the
   pilot-scale numbers already in the research paper and capstone report.
