# GPU runbook — cleaning the dataset, retraining, and proving the result

Status of the current model, so the steps below have a baseline:

| fact | value | where it comes from |
| --- | --- | --- |
| shipped checkpoint | `ml/runs/classify/runs/disease_classifier-3/weights/best.pt` | `args.yaml` (batch 32, imgsz 224, 50 epochs, seed 42) |
| training-time val top-1 | **0.98821** (epoch 50) | `runs/.../disease_classifier-3/results.csv` |
| deployed-model val top-1 | **95.83%** (480 stratified images, app-parity preprocessing) | `ml/class_error_report.txt` |
| export fidelity | 496/496 argmax agreement, max Δp 4e-6 | `ml/export_parity_report.txt` |
| field photos | 43.8% (7/16), no gate → 87.5% (14/16) at 0.7 | `ml/benchmark_preprocessing.txt` |
| dataset defects | 42.9% of wheat val shares a source photo with train; 693 val files byte-identical to train; 1448 same-class dup groups; 256 cross-class dup groups; `val/Wheat___healthy` is disputed ground truth | `ml/dataset_integrity_report.txt` |

The 98.8% and the 95.8% are both measured on a split that is **not held out**, which is why they are
worth less than they look. Fixing that is the point of step 2.

## 0. What actually needs the friend's GPU

| task | GPU-bound? | run it where | why |
| --- | --- | --- | --- |
| rebuild the split (`rebuild_split.py`) | **no** — sha256 + file copies | this machine | it only reads/writes files; results are deterministic |
| train (`train_disease.py`) | **yes** — but this machine already has a working CUDA GPU (measured below) | either machine; the friend's laptop is only faster | the only step that needs CUDA |
| export to TFLite (`export_tflite_float32.py`) | **no** — PyTorch→ONNX→onnx2tf | this machine | it depends on `C:\Python311` + `ml/.export-libs` (or `C:\Python314` + `ml/.export-libs-314`) which are already installed and proven here |
| parity / accuracy / error audit / integrity audit | no | this machine | pure inference on the CPU, 16–500 images |

Two consequences that change the original plan:

1. **The dataset does not have to move.** Clean it here, then either copy the cleaned split to the
   friend's laptop or, better, only the ~100k cleaned files if they insist on training there.
2. **The export should not happen on the friend's laptop.** That machine has no `.export-libs`
   (~700 MB of TensorFlow/onnx2tf pins). Copying back `best.pt` (~10 MB) and exporting here is
   cheaper and uses a pipeline that is already proven byte-faithful.

**Measured on this machine while writing this runbook** (`ml/train_disease.py --env-only`):

```
python 3.11.9 | ultralytics 8.4.142 | torch 2.5.1+cu121 | cuda ready: True
device: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB VRAM, capability 8.6, 1 device visible)
suggested batch on this device: 16
```

The "no-GPU" assumption about this laptop is wrong: it is the same 6 GB GPU that trained
`disease_classifier-3` at batch 32. So the retrain can run here, tonight,
with `--data data/merged_clean` — the friend's laptop is a speed upgrade, not a prerequisite. If time
on the friend's machine is short or uncertain, doing everything here is the lower-risk path.

## 1. Step 1 — rebuild the split (this machine, CPU only)

```powershell
# read-only report first: hashes every image, writes nothing (a few minutes)
& 'ml\.venv\Scripts\python.exe' ml\rebuild_split.py

# then build the clean split + quarantine tree
& 'ml\.venv\Scripts\python.exe' ml\rebuild_split.py --apply
```

Fast iteration while checking the logic (wheat only, writes to `data/merged_clean_wheat`):

```powershell
& 'ml\.venv\Scripts\python.exe' ml\rebuild_split.py --scope wheat
```

What it does, and what to look for in the output:

| step | effect | read this in the report |
| --- | --- | --- |
| dedupe inside a class | 1 copy kept, the rest quarantined | `same_class_duplicate` count (expect ≈ 2216 wheat + more in other crops) |
| identical pixels, two labels | every copy quarantined | `cross_class_duplicate` (expect ≈ 559 wheat) |
| split along *source photo* boundaries | `<uuid>___leaf_270deg.JPG` can no longer land in val while `<uuid>___leaf.JPG` stays in train | `source photos spanning train and val` must be **0** |
| `val/Wheat___healthy` | quarantined: the source labels them healthy, the trained model calls them yellow rust at ~100% confidence | `mislabeled_val_wheat_healthy` (the first dry run reported 50; the other 20 left as duplicates) |
| class presence | if a class loses val images, whole train photos are promoted to val | `source photos promoted train -> val` |
| class contract | rebuilt class set must equal `backend/models/class_names.json` | `class_names.json contract : OK` |

Exit code is 0 only when the rebuilt split is leak-free, complete for every class, and the class
order still matches the JSON. `--keep-mislabeled-val` disables the `Wheat___healthy` quarantine if you
want to see the difference; `--mode link` hardlinks instead of copying (same drive only).

Do **not** re-run `ml/merge_datasets.py` to "start clean" — it produced this leak
(`split_images()` splits a class's file list 80/20 per file). `merge_datasets.py` is left untouched
as the historical record; `rebuild_split.py` is the replacement for the split stage.

### What the first `--scope wheat` dry run measured (14,154 wheat files)

```
train 8,663   val 2,174   quarantined 3,317   files that changed side of the split 2,233
  same_class_duplicate       2,685
  cross_class_duplicate        582
  mislabeled_val_wheat_healthy  50
VERDICT: leak-free rebuild, safe to train on   (0 leaked digests, 0 shared source photos, 15/15 classes present)
```

The unpleasant but important part: **a large share of the wheat "training data" was duplicate copies of
the same photographs**, so cleaning shrinks some classes hard.

| class | files before | after quarantine | kept |
| --- | --- | --- | --- |
| Wheat___septoria | 1,214 | 865 (71%) | 349 |
| Wheat___smut | 1,415 | 855 (60%) | 560 |
| Wheat___leaf_blight | 912 | 296 (32%) | 616 |
| Wheat___black_rust | 646 | 316 (49%) | 330 |
| Wheat___yellow_rust | 1,371 | 20 (1%) | 1,351 |

Consequences to plan for before the retrain:

- wheat classes are now far more imbalanced than the file counts suggested (330–1,351 kept per class),
  so wheat accuracy may initially *drop* even though the split is finally honest;
- if wheat matters for the demo, add independent wheat images rather than trusting the source set;
- the disputed `val/Wheat___healthy` files are out of the split, and the rebuilt `val/Wheat___healthy`
  consists of 198 images that came from `train/Wheat___healthy` — so **a human should eyeball a sample of
  `ml/data/merged/train/Wheat___healthy` before the retrain**, because that label is now load-bearing.

`ml/audit_leaf_labels.py` was written to automate that label check, and it deliberately refused to
answer on the wheat classes: its own self-test (held-out halves of `Wheat___yellow_rust` vs
`Wheat___mildew`) only separated them 85.3% / 44.7%, which is too weak to judge a third class. Treat a
suppressed verdict as "needs human review", not as "labels are fine".

## 2. Step 2 — what to copy to the friend's laptop

Copy (or zip) only what is needed:

- `ml/data/merged_clean/` — the cleaned split (the whole point)
- `ml/real_world_test/` — the 16 field photos, used by the later verification
- `ml/*.py` — the tooling (`train_disease.py` is the entry point)
- `requirements`/`environment` files, not `.venv` (absolute paths; build a fresh venv there)

Do **not** copy `ml/data/merged/`, `ml/runs/`, `ml/export_build*/`, `ml/.export-libs*/`,
`ml/.venv/` — all regenerable, multi-GB, and `.gitignore`d.

Create the environment there:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# install the CUDA build that matches the laptop's driver, then:
.\.venv\Scripts\python.exe -m pip install ultralytics
```

## 3. Step 3 — environment preflight on the GPU laptop (do this before anything else)

Never assume the other laptop is set up. Run these three, in order:

```powershell
nvidia-smi                                  # driver + GPU name + VRAM + which CUDA it supports
python -V                                   # 3.10-3.12 recommended
& '.\.venv\Scripts\python.exe' ml\train_disease.py --env-only   # on this machine: ml\.venv\Scripts\python.exe
```

`--env-only` prints what training will actually see and exits without touching the dataset:

```
Environment preflight
  python      : 3.12.x (...)
  ultralytics : 8.4.x
  torch       : 2.x.x+cuXXX
  torch cuda  : 12.x
  cuda ready  : True
  device      : NVIDIA GeForce RTX 40xx (8.0 GB VRAM, capability 8.9, 1 device(s) visible)
  suggested batch on this device: 32
```

Stop and fix the environment if any of these is true:

- `torch cuda : not a CUDA build` → `pip install torch --index-url https://download.pytorch.org/whl/cu12x`
- `cuda ready  : False` → wrong torch build, or a driver older than the CUDA build
- `ultralytics : NOT INSTALLED` → `pip install ultralytics`

Batch size is the one setting that must not be inherited blindly. `train_disease.py` passes
Ultralytics `batch=-1` by default, which sizes the batch from *live free VRAM* on that machine;
`suggest_batch()` prints the expectation to compare against:

| total VRAM | expectation | note |
| --- | --- | --- |
| < 4 GB | 8 | also consider `--imgsz 160` |
| 4–6 GB | 16 | the RTX 3050 run used 32; 16 is the safe default at this size |
| 6–8 GB | 32 | matches the run that produced `disease_classifier-3` |
| 8–12 GB | 64 | |
| > 12 GB | 96 | |

## 4. Step 4 — train on the cleaned split

```powershell
& 'ml\.venv\Scripts\python.exe' ml\train_disease.py --data data/merged_clean --name disease_clean
```

On a machine whose venv lives at the repo root (the friend's laptop, step 2) use `.\.venv\Scripts\python.exe` instead; this machine's interpreter is `ml\.venv\Scripts\python.exe`. Both `--data` and `--project` are resolved against `ml/` rather than the current directory, so the run always lands in `ml/runs/disease_clean/` and can be launched from anywhere.

Behaviour changes from the old `train_disease.py`:

- training starts **fresh** (`--resume` is opt-in); the old file hardcoded `resume=...` and would have
  continued the previous run instead of learning from the cleaned data;
- `--batch` defaults to `auto`; `--batch 16` overrides it;
- `--workers 2` if the Windows data loader stalls.

What to expect in `ml/runs/disease_clean/results.csv`:

- every epoch is slower than the run-3 log (fewer duplicates, so the same wall time buys fewer images);
- **val top-1 will drop below 98.8% and that is the fix working**: the old number counted leaked
  images as held-out. A clean run in the low-to-mid 90s over 60 classes is the honest baseline;
- `Wheat___` errors should drop sharply, because the 559 conflicting wheat files and the mislabeled
  `val/Wheat___healthy` images are no longer in the data.

If epoch 1 dies with an OOM: re-run with `--batch 16` (then 8). Use `--imgsz 192` only as a last
resort — the app resizes to 224 and the exporter is validated at 224.

## 5. Step 5 — copy these off the friend's laptop before leaving

| artifact | path | why |
| --- | --- | --- |
| checkpoint | `ml/runs/disease_clean/weights/best.pt` | the model to export |
| training args | `ml/runs/disease_clean/args.yaml` | proof of the exact batch/seed/imgsz used |
| metrics | `ml/runs/disease_clean/results.csv` | per-epoch val top-1 |
| optional | `.../weights/last.pt` | only if a resume may be needed |

Do not copy `ml/data/merged_clean` back — this machine already built the identical split (unless the
split was repaired over there instead).

## 6. Step 6 — export + verify (this machine, CPU only)

```powershell
# a) export float32 NHWC and install it into the app + backend
& 'ml\.venv\Scripts\python.exe' ml\export_tflite_float32.py --checkpoint ml\runs\disease_clean\weights\best.pt --copy

# b) flatbuffer vs checkpoint across every class (default val dir is the OLD leaky one - override it)
& 'ml\.venv\Scripts\python.exe' ml\verify_export_parity.py --checkpoint ml\runs\disease_clean\weights\best.pt --val-dir ml\data\merged_clean\val --val-per-class 20

# c) the shipped file on the 16 field photos + a clean val subset
& 'ml\.venv\Scripts\python.exe' ml\evaluate_real_world.py --val-subset 8

# d) where the remaining errors live (takes a TFLite model, not a checkpoint - so run it after the export)
& 'ml\.venv\Scripts\python.exe' ml\diagnose_class_errors.py --data-dir ml\data\merged_clean --val-per-class 8

# e) confirm the leaks are gone in the split that training used
& 'ml\.venv\Scripts\python.exe' ml\audit_dataset_integrity.py --scope all
```

Two defaults that must be overridden after the rebuild, or the verifiers keep reading the old split:
`verify_export_parity.py --val-dir` and `diagnose_class_errors.py --data-dir` both default to
`ml/data/merged`.

Acceptance gates:

- (a) `class order matches class_names.json` = PASS. The exporter already asserts this (it compares
  the checkpoint's `model.names` with the JSON), and it is the contract that keeps softmax index →
  label mapping correct. `rebuild_split.py` checks the dataset side of the same contract;
- (b) `argmax agreement 100%` and `max probability delta <= 1e-3`;
- (c) field accuracy **should not be expected to jump**: it measures domain shift (43.8% ungated
  before the retrain). What should improve is how many of the 16 photos clear the 0.7 gate;
- (d) wheat confusion pairs shrink — diff against the saved `ml/class_error_report.txt`;
- (e) `VERDICT: clean` for `train` and `val`.

## 7. Step 7 — ship it (this machine)

The 0.7 confidence gate is already implemented, so there is nothing left to build:

- `TFLiteHelper.MIN_CONFIDENCE = 0.70f` — single source of truth for the threshold;
- `ScanActivity.handlePredictionSuccess` shows a "Retake" snackbar below that score instead of
  routing to the result screen (the gate does not name the disease, because the point is that the
  prediction cannot be trusted);
- `backend/main.py` `/predict` adds `low_confidence: true` plus a `message` (still HTTP 200 so
  existing callers keep working).

Remaining manual work:

1. re-run the end-to-end app test with the replaced model asset: scan → result → weather → voice →
   history, and once with a deliberately bad photo to see the retake path;
2. `android/app/build/intermediates/assets/debug/agroveyra_model.tflite` is a stale build copy; the
   next Gradle build regenerates it from `src/main/assets`;
3. grow `ml/real_world_test` beyond 16 photos — with 16, one photo is 6.25% and field accuracy is
   barely measurable.

## 8. Failure playbook

| symptom | cause | fix |
| --- | --- | --- |
| `dataset not found: ...merged_clean` | step 1 not run, or the folder was not copied | `rebuild_split.py --apply` |
| `torch cuda : not a CUDA build` | CPU wheel installed | reinstall the cu12x wheel |
| OOM in epoch 1 | batch too large for that GPU | `--batch 16`, then 8 |
| DataLoader hangs on Windows | too many workers | `--workers 0` (what `disease_classifier-4` used) |
| val top-1 of 99%+ again | the run still points at `data/merged` | check the `data:` field in the run's `args.yaml`; it must be `data/merged_clean` |
| export fails on `C:\Python311\python.exe` | the export interpreter moved | reinstall per the docstring in `export_tflite_float32.py` |
| `REBUILD REJECTED: empty val split (Ultralytics needs every class in both)` | a class ended up with no usable val photos | raise `--min-val-per-class`, or use `--keep-mislabeled-val` to see which class caused it |
