# GPU runbook — cleaning the dataset, retraining, and proving the result

Status of the current model, so the steps below have a baseline:

| fact | value | where it comes from |
| --- | --- | --- |
| shipped checkpoint | `ml/runs/classify/runs/disease_classifier-3/weights/best.pt` | `args.yaml` (batch 32, imgsz 224, 50 epochs, seed 42) |
| training-time val top-1 | **0.98821** (epoch 50) | `runs/.../disease_classifier-3/results.csv` |
| deployed-model val top-1 | **95.83%** (480 stratified images, app-parity preprocessing) | `ml/class_error_report.txt` |
| export fidelity | 496/496 argmax agreement, max Δp 4e-6 | `ml/export_parity_report.txt` |
| field photos | 43.8% (7/16) ungated and unchanged by the rebuilt split; 100% precision (6/16 shown) at the re-tuned 0.95 gate | `ml/real_world_comparison.txt`, `ml/confidence_gate_analysis.txt` (§9) |
| dataset defects | 42.9% of wheat val shares a source photo with train; 693 val files byte-identical to train; 1448 same-class dup groups; 256 cross-class dup groups; `val/Wheat___healthy` is disputed ground truth | `ml/dataset_integrity_report.txt` |
| rebuilt split | `ml/data/merged_clean` — train 75,419 / val 18,957, 3,493 quarantined, 0 leaked digests, 0 shared source photos, 60/60 classes | `ml/dataset_rebuild_report.txt`; re-audited clean by `ml/dataset_integrity_report_clean.txt` |
| class contract | 60 classes across 17 crops; `class_names.json` ordered + identical in backend/Android, 66-entry `treatment_db.json` covers all 60 | `ml/verify_class_contract.py` |

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

### The class contract gate — run this before any training

`rebuild_split.py` checks the dataset side of the class contract inside its own verification, but
nothing checked the rest of it: the app's copy of the list, the treatment text, or the flatbuffer that
actually ships. `ml/verify_class_contract.py` is that gate, and `train_disease.py` now runs it before
epoch 1 (it exits 3 instead of spending GPU hours on a dataset the app would mislabel silently):

```powershell
# after step 1; the default dataset becomes data/merged_clean as soon as it exists
& 'ml\.venv\Scripts\python.exe' ml\verify_class_contract.py --data ml\data\merged_clean --json ml\class_contract_report.json
```

Read-only. It asserts:

- `backend/models/class_names.json` parses as 60 unique names and is already in Python `sorted()` order
  — that order *is* the TFLite output index order, so reordering it silently relabels every prediction;
- those 60 classes still cover **17 crop species** (the 20 raw PlantVillage `Crop___Disease` prefixes,
  with `Cherry_(including_sour)`, `Corn_(maize)` and `Pepper,_bell` merged back into 3 crops);
- `android/app/src/main/assets/class_names.json` is byte-identical to the backend copy;
- every class has a `treatment_db.json` entry (66 keys = 60 classes + 6 kept extras) and both copies
  of that file match;
- the shipped `agroveyra_model.tflite` (backend copy == Android copy) matches the size/sha256 recorded
  in `ml/export_parity_report.txt`, and that report says 60 classes;
- `train/` and `val/` hold exactly those 60 class folders, none of them empty.

Exit code 0 means the contract holds. The "52 disease categories across 17 crops" line in
`GROUP_5_PROJECT.pdf` is **documentation only** — the implementation target is these 60 classes — and
`--expect-count 52` fails the guard by design, which is the self-test that the gate is not a no-op.
`train_disease.py --skip-contract-check` bypasses it; never use that to "make it pass".

Do **not** re-run `ml/merge_datasets.py` to "start clean" — it produced this leak
(`split_images()` splits a class's file list 80/20 per file). `merge_datasets.py` is left untouched
as the historical record; `rebuild_split.py` is the replacement for the split stage.

### What the full-scope dry run measured (97,869 files, 60 classes)

```text
train 75,419   val 18,957   quarantined 3,493   files that changed side of the split 27,810
  same_class_duplicate        2,861
  cross_class_duplicate         582
  mislabeled_val_wheat_healthy   50
VERDICT: leak-free rebuild, safe to train on   (0 leaked digests, 0 shared source photos, 60/60 classes present)
```

27,810 files (28% of the dataset) change side of the split because the old merge split each class's
file list 80/20 *per file*: `<uuid>___leaf_270deg.JPG` could sit in `train` while `<uuid>___leaf.JPG`
sat in `val`. That reshuffle is the leak being removed, not damage. Nothing is deleted — all 3,493
dropped files are preserved under `ml/data/quarantine/<reason>/`, and
`ml/dataset_rebuild_manifest.csv` records the reason and destination of every one of the 97,869 files.
Cleaning hits the wheat classes that were mostly duplicates very hard (`Wheat___septoria` keeps 279 of
its 1,144 train images), so the retrain is finally honest but the wheat class balance is worse than the
old file counts suggested.

### Applied, then re-audited by the *other* tool (12:07:59 → 12:13:37 apply, 12:27:55 audit re-run)

`rebuild_split.py --scope all --apply --mode link` was run for real against the reviewed plan (97,869
destinations, 5m38s), and then `audit_dataset_integrity.py` was pointed at `ml/data/merged_clean`
(12:13:45 → 12:27:55, exit 0). **The two tools agree on every section**, and the audit's numbers match the
0/0 the rebuild script claimed for itself:

| audit section | pre-fix (`merged`) | post-apply (`merged_clean`) |
| --- | --- | --- |
| 1a val images whose source photo is also in train | 450/1,050 wheat (42.9%) | **0/18,957 (0.0%)** |
| 1b byte-identical val images / unique digests | 693 / 486 | **0 / 0** |
| 2 same-class duplicate groups (redundant files) | train 1,448 (2,216), val 297 (309) | **0 (0), 0 (0)** |
| 3 cross-class duplicate groups (affected files) | 256 groups / 559 files | **0 / 0** |
| VERDICT | leaks present | **clean** |

`ml/dataset_integrity_conflicts_clean.csv` is header-only (0 rows). The split sizes are identical in both
tools — **train 75,419 / val 18,957** — and `verify_class_contract.py --data ml/data/merged_clean` lists
those same two numbers and passes **18/18** (60/60 folders in each split, none of them empty), so the
60/60 class count is confirmed independently by the rebuild report, the audit and the contract gate.

### GPU-TASK-2: the retrain on the cleaned split (started 12:33:58)

```powershell
& 'ml\.venv\Scripts\python.exe' -u ml\train_disease.py --data data/merged_clean --name disease_clean --batch 16
# ml\_train_clean.out       live log (launched with -u so it streams instead of block-buffering)
# ml\_train_clean.err       stderr
# ml\_train_clean.pid       PID;  Get-Process -Id (Get-Content ml\_train_clean.pid)
# ml\runs\disease_clean\    args.yaml / results.csv / weights\best.pt
```

It is detached (`Start-Process -WindowStyle Hidden`), so closing the editor or losing this session does not
stop it. Recorded config: YOLO11n-cls, epochs 50, imgsz 224, batch **16** (explicit — the `--env-only`
suggestion for this laptop's 6 GB RTX 3050, not Ultralytics auto-batch), device 0, workers 4, patience 10,
dropout 0.4, seed 42, and **no `--resume`**, so this is a genuinely fresh model rather than a silent
continuation. The dataset scan reports **0 corrupt** images and the in-process contract gate passed before
epoch 1.

One trap worth keeping: do **not** hand `Start-Process -ArgumentList` an absolute `--data` path. PowerShell
joins the array elements without quoting and this repo's path contains a space (`ROHIT RAJESH`), so argparse
rejects `RAJESH\AgriVeyra\...` as a stray argument — exactly how the first launch attempt died seconds after
starting. The relative `--data data/merged_clean` (resolved against `ml/`) has no spaces and is what
`train_disease.py` documents.

Training also writes `ml/data/merged_clean/train.cache` + `val.cache` (Ultralytics' scan index; both scans
reported **0 corrupt** for 75,419 + 18,957 images). Those are new files *inside* the dataset directory, so a
raw file count there becomes 94,378 rather than 94,376 — the image count is unchanged, and every tool here
filters by extension, so the audit's 75,419 / 18,957 are unaffected. The caches can be deleted at any time;
they are rebuilt on the next run.

### Reconciling those numbers with the audit (same defect, two different units)

`rebuild_split.py` reports **files**, `audit_dataset_integrity.py` reports **groups**, and for its section 2
the audit groups by digest across the whole split (`audit_dataset_integrity.py:180`) instead of by
`(class, digest)`. That mismatch is why `ml/_dryrun_reconcile.py` exists: it re-derives the audit's units
from the dry-run manifest plus the sha256 cache (read-only, no re-hashing) so the two can be compared
rather than argued about. Verified on the 97,869-file plan:

| defect | pre-fix audit (wheat scope) | rebuild plan, wheat, audit's own rule | rebuild plan, all 60 classes |
| --- | --- | --- | --- |
| identical train/val images | 693 val files / 486 digests | **693 / 486** (of 1,050 val files) | 753 / 546 (of 17,816) |
| same-class duplicate groups | train 1448 (2216 redundant), val 297 (309) | **1448 / 2216** and **297 / 309** | 1550 / 2322 and 307 / 319 |
| cross-class duplicate groups | 256 groups / 559 train files | **256 / 559** | 256 / 559 |
| worst label pairs | black_rust↔brown_rust 102, leaf_blight↔septoria 97, leaf_blight↔tan_spot 25 | **102 / 97 / 25** | identical |
| photos split across train/val | 450/1,050 wheat val images (42.9%) | — | 8,419 photos / 31,179 images → **0 after the rebuild** |

Every pre-fix audit figure reproduces exactly under the audit's own rule, so the plan is not a different
reading of the same data — only a different counting unit. The rebuild's quarantine counts are larger for
two reasons: they cover all 60 classes, and they also drop a copy whose twin sits in the *opposite* split
(that copy is a leak, not an in-split duplicate).

### Proof for the disputed `Wheat___healthy` label, measured after the rebuild

The rebuild does not *relabel* the 70 disputed `val/Wheat___healthy` files — it removes them: 50 were
flagged `mislabeled_val_wheat_healthy` and 20 were already duplicate copies, so **none of the disputed
files survive in the rebuilt split**. The rebuilt `val/Wheat___healthy` holds **198 images, all of them
taken from `train/Wheat___healthy`** (source label `healthy`); it is the only class whose val folder is
rebuilt entirely from the other split.

That claim was then tested against the deployed flatbuffer instead of trusted
(`diagnose_class_errors.py --data-dir ml/data/merged_clean --val-per-class 198` covers the whole folder):

```text
rebuilt val/Wheat___healthy : 198 images
predicted Wheat___healthy   : 198/198 (100%), mean confidence 0.980
predicted as                : Wheat___healthy x198   (no other class predicted)
whole clean val sample      : 9,872/10,038 = 98.35% top-1   (deployed tflite, app-parity resize)
```

Before the rebuild, 60 of that class's 70 val files were called `Wheat___yellow_rust` at ~100%
confidence. The images the model disagreed with are gone, and the class it now validates on is labelled in
a way the model itself agrees with — the strongest available evidence short of a human re-labelling the
source set. (The old 95.83% figure was measured on a 480-image sample from the leaky split, so the two
percentages are not like-for-like.)

### Why re-running `--apply` is safe, and what it does not touch

`data/merged` is opened read-only, so the source stays byte-identical: no file under it carries the
rebuild day's modification time, and its only non-image entries are Ultralytics' `train.cache` /
`val.cache` (97,869 images + those 2 caches = the 97,871 files counted there). `place()` writes every
destination with `os.link()` and falls back to `shutil.copy2()` only if linking fails, so `merged_clean`
and `quarantine` are hard links, not copies — a sampled file reports `st_size=291,764 st_nlink=2`, and the
94,376 + 3,493 = 97,869 placements cost no extra disk. Re-running `--apply` is idempotent (`place()`
unlinks an existing destination before re-linking it), which is why re-applying the reviewed plan took
5m38s and changed no file content.

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

**Confirmed on this machine: `--batch 16`** — 6.0 GB total / 5.04 GB free on the RTX 3050 6GB
Laptop GPU, so the 6 GB rule applies. Pass it explicitly: the `--batch auto` default sizes the
batch from *live free VRAM* and would pick something larger (32), which would no longer match the
configuration that was reviewed. `--batch 8` is the fallback, and the measurement below says it
will not be needed.

### Environment verified on this machine (2026-09-20)

Nothing needed installing: the venv already carries a CUDA build of torch and a current
Ultralytics.

```
python       : 3.11.9       (ml\.venv\Scripts\python.exe)
torch        : 2.5.1+cu121  (torch.version.cuda 12.1, cudnn 90100)
ultralytics  : 8.4.142
cuda ready   : True         (1 device, driver 616.92)
device       : NVIDIA GeForce RTX 3050 6GB Laptop GPU, capability 8.6, 6.0 GB total
free at probe: 5.04 GB      (a tensor matmul on the GPU succeeded)
```

`ml/_gpu_smoke_check.py` measures one real training step (dict batch -> `model.loss`, fp16
autocast, backward, SGD step - the same path the trainer takes) on synthetic tensors:

| batch | peak allocated | peak reserved |
| --- | --- | --- |
| 16 | 0.21 GB | 0.24 GB |
| 32 | 0.36 GB | 0.40 GB |

Two traps the check had to work around, either of which silently produces a wrong result:

- `model(tensor)` is the *inference* path in Ultralytics (`nn/tasks.py`, `BaseModel.forward`);
  the training step passes a **dict**, so the loss has to come from `model.loss({...})` or
  `model({"img": ..., "cls": ...})`;
- the pretrained checkpoint is loaded with every parameter **frozen** (`utils/torch_utils.py`
  `strip_optimizer` sets `requires_grad=False` when a checkpoint is saved); the classification
  trainer re-enables them at `models/yolo/classify/train.py:90`, so a smoke test must call
  `requires_grad_(True)` first.

For planning only: the last full run (`disease_classifier-3`, batch 32 on the old leaked
`data/merged`) recorded 20,095 s for 50 epochs, about 402 s/epoch. Expect the cleaned 50-epoch run
at batch 16 over 75,419 train images to land in the hours-per-run range; check the first two
epochs of `results.csv` before walking away from all 50.

## 4. Step 4 — train on the cleaned split

```powershell
& 'ml\.venv\Scripts\python.exe' ml\train_disease.py --data data/merged_clean --name disease_clean --batch 16
```

On a machine whose venv lives at the repo root (the friend's laptop, step 2) use `.\.venv\Scripts\python.exe` instead; this machine's interpreter is `ml\.venv\Scripts\python.exe`. Both `--data` and `--project` are resolved against `ml/` rather than the current directory, so the run always lands in `ml/runs/disease_clean/` and can be launched from anywhere.

Behaviour changes from the old `train_disease.py`:

- training starts **fresh** (`--resume` is opt-in); the old file hardcoded `resume=...` and would have
  continued the previous run instead of learning from the cleaned data;
- `--batch` defaults to `auto`; `--batch 16` overrides it;
- `--workers 2` if the Windows data loader stalls;
- it runs `ml/verify_class_contract.py` before epoch 1 and refuses to train (exit 3) if the dataset no
  longer matches the shipped `class_names.json`; `--skip-contract-check` overrides that, and
  `--expect-count` pins a class count other than the shipped 60.

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

The confidence gate is already implemented, so there is nothing left to build:

- `TFLiteHelper.MIN_CONFIDENCE = 0.95f` — single source of truth for the threshold. Re-tuned on
  2026-09-20 for the leak-free model: on the 16 field photos the old 0.70 default showed 11 answers
  and 5 of them were wrong, while 0.95 shows 6 and all 6 are right — and it loses no correct answer
  that 0.70 kept, it only drops wrong ones. See §9 and `ml/confidence_gate_analysis.txt`;
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

## 9. What step 6 actually changed (measured 2026-09-20)

Step 6 rebuilt the split, retrained on it, re-exported and re-scored everything. Full write-up:
`ml/real_world_comparison.txt`; the gate numbers quoted in §7 come from `ml/confidence_gate_analysis.txt`.

| measurement | before (leaky build) | after (`disease_clean`) |
| --- | --- | --- |
| field photos, top-1 | 43.75% (7/16) | **43.75% (7/16) — unchanged** |
| field photos, top-3 | 50.00% | 50.00% — unchanged |
| field photos, answers the app shows at 0.70 | 8 shown, 1 wrong (87.5% precision) | 11 shown, 5 wrong (54.5%) |
| subset of `data/merged/val` (3/class, the same 180 files both runs) | 96.67% | 92.78% |
| subset of `data/merged_clean/val` (honest control, 3/class) | split did not exist | 91.11% |
| flatbuffer vs checkpoint parity | — | 100% argmax agreement, all 60 classes |

Why the leaky-val subset *fell* instead of rising: `--val-subset` samples `sorted()[:3]` per class from
the leaky `data/merged/val` (the same 180 files both runs). The rebuild moved 85 of them into the
model's own training set, 53 into the honest val split, and deleted 42 (duplicates plus the disputed
`Wheat___healthy`). Scored by what the rebuild did to each file: 85/85 correct in train, 52/53 in val,
27/37 same-class duplicates, 1/2 mislabeled, 0/1 cross-class duplicate — every one of the 13 misses is a
deleted file or an honest val photo, none is a file whose path was rebuilt into training. The whole
delta is the `.png` augmented-wheat tail: `.jpg` files 135 → 1 wrong for *both* models, `.png` files
45 → 5 wrong before, 12 after. Removing the leak lowered the old number; the new model is not worse.

Two controls that close off the obvious explanations:

- **resize policy**: `--center-crop` on the honest split scores 92.22% vs 91.11% for the app's
  stretch-resize, and leaves the field photos at exactly 43.75% — preprocessing is not the bottleneck;
- **decode path**: PIL and cv2 decode these files identically (max pixel delta 0), and the missed
  images are not degenerate (they have *higher* pixel std than the correct ones).

The "wheat label fix" was a quarantine, not a relabel — 50 files were moved to
`data/quarantine/mislabeled_wheat_healthy` on the model's own vote, so it cannot be cited as proof the
labels were wrong. Field evidence shows no benefit from it: `healthy_wheat_1` is still correct but its
confidence fell 76.18% → 59.86%, i.e. below the gate.

Verdict, and the reason §7 ships 0.95: **the domain gap dominates.** Deleting 3,493 duplicate or
mislabeled files, re-splitting to zero train/val overlap and retraining leak-free moved the field
number by exactly zero — the errors are cross-crop confusions (rice → `Wheat___leaf_blight`, tomato →
`Cotton___bacterial_blight` at 92%, `yellow_rust` → `brown_rust`) that curated close-up photos cannot
teach away. What the rebuild *did* change is the cost of being wrong: the leak-free model is wrong at
high confidence more often, so the 0.70 gate stopped filtering (54.5% precision) and 0.95 became the
honest threshold. No gate fixes a confidently wrong answer — the same 16 photos scored with center-crop
put `wheat_yellow_rust_2` at 99.70% and wrong — so treat the gate as a UX guard, not a correctness
guarantee.

## 10. Pest model (the second model) - scope, data, and the leak that was found

The architecture always called for two models: `yolo11n-cls` for disease (shipped, §9) and a second
model for pests. This is what the data on this machine actually allows.

**IP102 as downloaded is classification-only.** `ml/data/pest/ip102/` plus the original `ip102.zip`
(3.0 GB) contain `classification/{train,val,test}` with 102 numeric class folders and no annotation
files at all: reading the zip's central directory gives 75,226 entries - 75,222 JPEGs and 4 text
files, **zero XML/COCO/YOLO boxes**. `yolo11n.pt` in detection mode (bounding boxes) is therefore not
trainable on this data, and the app has no `YOLODetector.kt` either, so pest *detection* as a
box-drawing feature does not exist in the codebase. What can be built is a pest *species* classifier -
the same architecture as the disease model.

Integrity (`ml/validate_pest_dataset.py`): all 75,222 IP102 images verified with PIL `verify()` in
88 s - **0 corrupt**, 102 class dirs per split, 0 empty. train 45,095 / val 7,508 / test 22,619.

**Approved scope: 41 classes, 27,551 images** via `ml/build_pest_scoped.py` ->
`ml/data/pest/ip102_scoped/` (train 16,513 / val 2,752 / test 8,286). The full 102-class mapping and
the rejected candidate scopes are in `ml/pest_scope_report.txt` (`ml/pest_scope.py`). In scope:
rice 13, wheat 7, citrus 14, grape 5, corn 1, peach 1. A class qualified if its name names a crop the
app's 60-class contract diagnoses, or if the named species' documented principal host is one of those
crops. Excluded: 26 classes whose host crop the app does not diagnose (beet, alfalfa, flax, cabbage,
mango, olive, legumes), 7 genus-level names with no mappable host, and the 8 in-scope classes with
under 200 images. Readable folder slugs (`rice_leaf_roller`) replace IP102's 0..101 indices, and
`pest_class_names.json` is written in Ultralytics' sorted-folder order so it can be diffed against
`model.names` after training.

**The official IP102 split is not leak-free.** `ml/audit_pest_scoped.py` ->
`ml/pest_scoped_audit.txt`: 0 exact md5 duplicates across splits and 0 filename conflicts, but an
exhaustive cross-split dHash sweep found 361 candidate pairs, and `ml/pest_neardup_verify.py` ->
`ml/pest_neardup_verification.txt` confirmed them by pixel comparison (24x24 grayscale MAE): **300
pairs at MAE 0-2** (the same photo re-encoded), 35 at 2-5, 11 at 5-10 - i.e. **346 pairs involving
497 distinct images are burst-series siblings that IP102 split across train/val/test** (train<->test
228, train<->val 83, val<->test 35). Only 8 pairs exceeded MAE 20, so this is not a hash artefact,
and the two cross-class candidates are among those 8, so there is no evidence of label noise. The
lesson carried over from the disease dataset: the split is checked, not assumed.

**Quarantine plan** (`ml/pest_quarantine_plan.py` -> `ml/pest_quarantine_plan.txt`, applied with
`ml/quarantine_pest_neardup.py --apply`): a model only trains on `train`, so only a *train* image can
leak into an eval split. **242 train images** (1.5% of 16,513; only `spotted_lanternfly` loses
meaningfully at -115) have a near-duplicate in val or test and are removed, which makes 56 val and
163 test images honest. A further **29 val images** are removed for val<->test twins, because val
selects the best epoch and would otherwise bias the final test number. The test split is never
modified. Files move to `ml/data/pest/quarantine/neardup_{train,val}/<slug>/` - moved, not deleted,
the same pattern as the disease dataset's `mislabeled_wheat_healthy` quarantine.

**Known gap for the limitations section.** Pest classification covers rice, wheat, grape and citrus
pests, based on the labeled data available (IP102). Cotton, tomato, pepper and potato - all crops the
disease model diagnoses - have no named or host-specific pest class in IP102; covering them requires
either additional labeled pest datasets or a broader, less precise polyphagous scope (76 classes,
63,207 images, evaluated and rejected as too weak to train on). Expanding pest coverage to those
crops is future work. Two further caveats belong in the same section: IP102 images are close-ups *of
the insect*, not of infested leaves, so the model is only meaningful when the photo is of the pest
itself; and IP102 has no healthy / "no pest present" class, so the pest model can never answer
"nothing here" and needs its own re-tuned confidence gate rather than reusing the disease gate.

**Training (2026-09-21, `ml/runs/pest_classifier_scoped`, run for real on the post-quarantine split):**
45 of 50 epochs ran before `patience=10` stopped it (best epoch 35); 155.9 s/epoch. **val top-1
74.29% / top-5 92.84%; test top-1 73.53% / top-5 92.30%** on all 8,286 held-out images, which were
never used to train or to choose the checkpoint. `model.names` was re-checked against
`pest_class_names.json` after training and is identical. The curve is the expected shape for a nano
model on 41 fine-grained species: top-1 climbs 55.6% (epoch 1) to 74.3% (epoch 35) then plateaus
while train loss keeps falling to 0.43, i.e. patience caught the onset of overfitting instead of us
guessing. Full numbers: `ml/pest_training_report.txt`, `ml/pest_results_curve.csv`.

**Export and verification (`ml/export_pest_tflite.py`, reusing the disease pipeline):** float32 TFLite
with an NHWC `[1,224,224,3]` input and `[1,41]` float32 output, 159 ops, softmax present, **no
quantize/dequantize ops**, class order identical to the json. Parity against the checkpoint on 20
test images: **20/20 top-1 agreement, max |p_tflite - p_torch| = 0.000002**. Installed as
`android/app/src/main/assets/agroveyra_pest_model.tflite` and `backend/models/agroveyra_pest_model.tflite`
(sha256 `e192b07125d741c4...`, 6.06 MiB, byte-identical in all three locations);
`agroveyra_model.tflite` (disease, `6aec2f1afedaf4ec`) is untouched. Report:
`ml/pest_export_report.txt`.

Reading those 20 predictions rather than only the score: 14/20 correct on the sampled images, and
every miss is *semantically adjacent* - `brown_plant_hopper` -> `white_backed_plant_hopper`,
`black_citrus_aphid` <-> `english_grain_aphid`, `florida_red_scale` and `grape_hawk_moth` ->
`spotted_lanternfly`. That is what fine-grained insect classification looks like, not random output.

**Pest confidence gate** (`ml/pest_gate_sweep.py` -> `ml/pest_confidence_gate_analysis.txt`), measured by
scoring all 8,286 test images **once** through the shipped TFLite asset with the app's own preprocessing
and applying the gate post-hoc:

| gate | shown | coverage | wrong shown | precision |
| --- | --- | --- | --- | --- |
| none | 8,286 | 100.00% | 2,313 | 72.09% |
| 0.70 | 6,190 | 74.70% | 936 | 84.88% |
| 0.80 | 5,680 | 68.55% | 690 | 87.85% |
| 0.90 | 5,062 | 61.09% | 443 | 91.25% |
| **0.95** | 4,596 | **55.47%** | 296 | **93.56%** |
| 0.99 | 3,758 | 45.35% | 124 | 96.70% |

Two findings that the training report alone does not show:

1. **The app's preprocessing costs about 1.4 points of top-1.** The same test split scores 73.53% through
   Ultralytics' evaluation transform but **72.09%** through `preprocess_image` (stretch to 224, /255),
   which is what `TFLiteHelper` does. Export parity (20/20 top-1 agreement, max |dp| = 2e-6) rules the
   model out as the cause, so the difference is preprocessing. **72.09% is the number the app produces**,
   and it is the number to quote.
2. **The model is weakest on the crops the app serves most.** Ungated top-1: grape 87.25%, citrus 77.18%,
   peach 76.44%, corn 73.33%, but **rice 59.02% and wheat 56.56%** (n = 2,479 and 914). At the 0.95 gate
   those become 86.12% and 85.91% precision, showing 915 and 369 of them respectively.

0.95 is the consistent choice with the disease gate (93.56% precision, 55.47% coverage in-domain), but the
honest framing differs from the disease model: there, 0.95 showed *only* correct answers on the 16 field
photos; here roughly 6% of shown answers are still wrong, and all of it is in-domain IP102 evidence - no
real farmer photo has ever been through this model.

**Status:** the pest model is trained, exported, verified, gate-measured and installed. Remaining work is
deliberately not started: (1) an app-side entry point - no Kotlin loads the pest asset, so the app behaves
exactly as before; (2) that needs a product decision first, because IP102 images are close-ups *of the
insect*, not of infested leaves, so the UX has to ask for a photo of the pest itself rather than reusing the
leaf-scanning flow.

## 11. Held-out test evaluation of the disease model (measured 2026-09-21)

§9's 97.6% and §7's gate numbers are **validation** numbers: `data/merged_clean/val` was monitored during
training for early stopping. To publish something test-shaped, val was split in half along source-photo
boundaries and one half was treated as held out (`ml/disease_test_evaluation.py` ->
`ml/disease_test_evaluation.txt`, 462 s, exit 0).

| check | result |
| --- | --- |
| split rule | whole augmented families move together (text before `___`), stratified per class, seed 42 |
| TEST half | 9,372 images / 5,777 distinct source families |
| SELECTION half | 9,585 images / 5,805 distinct families; classes with <4 leaves in either half: **0** |
| `best.pt` vs `last.pt` | 236 tensors compared, max abs diff **0** — no epoch was ever chosen on val |
| exact duplicates against all 75,419 train images (md5) | **0** |
| model actually scored | the shipped `agroveyra_model.tflite` (`6aec2f1afedaf4ec`) with the app's preprocessing |
| field photos, same harness | 7/16 = 43.75% top-1 (matches §9 exactly, so the harness agrees with `evaluate_real_world.py`) |

The evaluation report prints 5,779 and 5,809 leaves because it sums the per-class group counts; two families
sit in two class folders each, so the distinct counts are 5,777 and 5,805 (re-checked with the split builder).

Why the split is by source photo and not by file: the images are augmented families of one leaf
(`1547f817-...___FREC_Scab 3194.JPG` / `_90deg` / `_180deg` / `_270deg`). Splitting by file would put two
rotations of the *same leaf* on both sides, which is not an independent sample — the same trap §9 fixed
at the train/val boundary.

**The number to quote: TEST half 9,126/9,372 = 97.38% top-1, 99.64% top-5** (95% CI 97.04–97.72% by
bootstrap over images). Macro precision 93.76% / recall 93.39% / F1 93.44%, weighted F1 97.35%. The
selection half scores 97.62% and the whole val 97.50%, so the held-out half is not an outlier; aggregating
the 1.6 variants per leaf, mean variant accuracy is 95.84% and 95.83% of leaves are fully correct.

Two independent facts make that defensible as a held-out number rather than a re-labelled val number:

1. **No epoch was selected on val.** The training curve peaked at the final epoch, so patience never
   fired — and that is now *verified at the weight level*: `best.pt` and `last.pt` are identical across
   all 236 tensors (max |Δ| = 0). Nothing was chosen by looking at the selection half.
2. **Nothing scored is in train.** Every one of the 18,957 val images was md5-ed against all 75,419 train
   images: **0 exact duplicates**, and the dHash near-duplicate sweep is dissected in §11a below rather
   than quoted raw.

**Where the errors are, by crop (TEST half):**

| crop | n | top-1 | same-crop, wrong disease | wrong crop |
| --- | --- | --- | --- | --- |
| Wheat | 1,083 | **80.15%** | 201 | 14 |
| Rice | 381 | 95.80% | 16 | 0 |
| Cotton | 158 | 98.73% | 0 | 2 |
| Cherry | 324 | 99.07% | 0 | 3 |
| Corn | 707 | 99.43% | 0 | 4 |
| Strawberry | 452 | 99.56% | 0 | 2 |
| Soybean | 253 | 99.60% | 0 | 1 |
| Peach | 446 | 99.78% | 1 | 0 |
| Apple | 963 | 99.90% | 1 | 0 |
| Tomato | 1,908 | 99.95% | 1 | 0 |
| Blueberry, Grape, Orange, Pepper, Potato, Raspberry, Squash | 2,697 | 100.00% | 0 | 0 |

**Wheat is the whole story:** 1,083 of 9,372 test images (11.6%) carry 215 of the 246 errors (87%). All ten
worst classes by recall are wheat — leaf_blight 48.39% (n=62, P 65.22%), black_rust 57.58%, tan_spot 60.56%,
mite 64.94%, common_root_rot 65.52%, aphid 68.97%, smut 75.00%, blast 79.66%, brown_rust 86.78%,
fusarium_head_blight 87.50%; the first non-wheat is `Rice___Bacterial_leaf_blight` at 90.62%. The confusions
stay inside the crop (leaf_blight <-> tan_spot 17 + 5, aphid <-> mite 9, common_root_rot <-> blast 7,
brown_rust <-> black_rust 6), so these are look-alike diseases rather than a broken class. The only repeated
wrong-crop error is `Corn___healthy -> Rice___healthy`, 4 times.

**Healthy vs naming (TEST half):** healthy leaves **2,921/2,933 = 99.59%**, diseased leaves
**6,205/6,439 = 96.37%**. Of the 246 misses, **217 name the wrong disease for the right crop** and 17 pick
the wrong crop entirely. That is the in-domain mirror of the field finding in
`ml/field_failure_analysis.txt`: the model is far better at "this leaf is sick" than at "this is which
disease".

**Gate sweep on the test half** (scored once, gate applied post-hoc, same method as §10's pest table):

| gate | shown | coverage | wrong shown | precision |
| --- | --- | --- | --- | --- |
| none | 9,372 | 100.00% | 246 | 97.38% |
| 0.50 | 9,232 | 98.51% | 157 | 98.30% |
| 0.70 | 9,069 | 96.77% | 80 | 99.12% |
| 0.90 | 8,886 | 94.81% | 33 | 99.63% |
| **0.95** | 8,780 | **93.68%** | 20 | **99.77%** |
| 0.99 | 8,518 | 90.89% | 11 | 99.87% |

In-domain, 0.95 costs only 6.3 points of coverage; on the 16 field photos the same threshold shows 6 of 16
(§9). **93.68% coverage in-domain against 37.5% on real photos, at the same gate** — that one line is the
domain gap, and it is why §7 keeps 0.95 as a precision guard rather than a claim that the gate fixes
anything.

### 11a. The 372 dHash pairs are not leakage (audited 2026-09-21)

The evaluation prints `near-duplicate pairs (dHash<=6, train sample 20,000): 372`. Quoted raw that reads like
a leak. It is not: `ml/disease_neardup_audit.py` -> `ml/disease_neardup_audit.txt` re-derived the same 372
pairs from the same seed and the same 20,000-image train sample (248 s) and then asked the only questions
that matter, from the pixels rather than from a hash.

| question | answer |
| --- | --- |
| do the two sides share a source family (uuid)? | **0 of 372** — no augmented family spans train/held-out, so the rebuild's rule holds |
| same class? | 253 yes, 22 same crop but different disease, 119 different crop |
| same leaf? | 38 pairs at centre MAE <= 5 (37 same-class), 68 "similar" (5–15), 266 merely the same background |
| how much of the TEST half does the strict set touch? | 16 images of 9,372 (0.17%), 16 of 5,777 families |

The 119 different-crop pairs are the tell that a raw dHash count is the wrong instrument here: a
PlantVillage frame is mostly plain background, the background gradient decides the hash, and leaves of
*different* crops land within 6 bits of each other. Those pairs cannot inflate a metric whose labels
disagree, which is why the raw 372 must not be quoted as leakage.

Where the strict 16 come from is more interesting, because it is a real (tiny) hole. The zero-MAE core is
`Cotton___healthy` filed as `h447.jpg` / `h469.jpg` — a second naming scheme with no uuid, so neither md5
nor family grouping can see it. `ml/disease_strict_pair_check.py` re-checked all 16 at 224x224 RGB:

| verdict | images | evidence |
| --- | --- | --- |
| identical copies | **11** | max per-pixel diff **0** on identical dimensions but different bytes (10 cotton-healthy, 1 wheat-blast) |
| burst siblings | 5 | max diff 105–211, MAE 4.6–7.5 (3 wheat-smut, 1 cherry `FREC_Pwd.M 0323`/`0324`, 1 rice `20231006_165800`/`165801`) |

So **11 of 9,372 TEST images (0.12%) have a pixel-identical twin in train**, which md5 cannot catch because
the two files are re-encoded. Control: the same val leaf against six other train leaves of its own class
sits at MAE 42–72 at the same scale, so the 64x64 discriminator is not saturating on white backgrounds.

**What it costs the headline** (TEST half, baseline 9,126/9,372 = 97.38%):

| excluded | images | left | accuracy | delta |
| --- | --- | --- | --- | --- |
| the 11 identical twins | 11 | 9,361 | 9,115/9,361 = **97.37%** | −0.003 |
| the strict set (same class, MAE <= 5) | 16 | 9,356 | 9,110/9,356 = **97.37%** | −0.004 |
| every same-class pair | 118 | 9,254 | 97.34% | −0.03 |
| every pair at all, any class | 171 | 9,201 | 97.33% | −0.05 |

Every excluded image is one the model already got right, so those are leak-free numbers rather than
rounding artefacts: **97.33–97.37% against the published 97.38%**. The train side of that sweep was a
20,000/75,419 = 26.5% sample, which is what the evaluation itself sampled; `--train-sample 75419` runs the
same tool over the whole split. Scaling the strict set by the unsampled share (3.8x, all still correct)
lands at 97.36%, and applying that same worst case to *every* flagged image (645) gives 97.18%, so the
sampling cannot move the conclusion either way.

The exhaustive variant is available but not cheap, and that is the reason the sampled audit is the one
quoted: at 75,419 x 18,957 the same-size bucket comparison is O(n x m) in pure Python and was still
running after five minutes of hamming with all 6.6 GB of images already read (measured: `--train-sample
75419`, stopped at cpu 789 s with no pair list yet, `ml/_dnafull_run.out`). A pigeonhole index over the
64-bit hash would make it interactive without changing the semantics, since the comparison stays inside a
size bucket either way.

Verdict: the split is intact, the hole the design cannot see is 11 byte-different copies of 11 images out
of 9,372, and the in-domain headline is robust in the fourth significant figure. None of it changes §11,
and none of it touches the honest field number of 43.75%.





