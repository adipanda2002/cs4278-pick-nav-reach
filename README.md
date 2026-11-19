# Pick, Navigate, Reach

A reproducible PyBullet project in which a Fetch robot proposes and executes a grasp, carries the object through a generated maze, and approaches the goal. This checkout contains the selected frozen controller package, its simulator assets, and compact evaluation evidence. The evaluation separates 150 development cases from 150 fresh-seed validation trials on the same frozen controller and supplied object set.

## Assessment entry point

Run the autonomous submission with `run_simple.py`, selecting the object and trial seed on the command line. `python pick_nav_reach_env.py` is the preserved starter keyboard demonstration, not the autonomous submission command. The assessment notes below identify the implementation and explain the saved result fields.

## Get the source

Clone the submission repository:

```sh
git clone --depth 1 --branch main https://github.com/adipanda2002/cs4278-pick-nav-reach.git
cd cs4278-pick-nav-reach
```

The frozen submission is tagged [`submission-90-60`](https://github.com/adipanda2002/cs4278-pick-nav-reach/tree/submission-90-60). You can also download its source ZIP from GitHub.

## Quick start

The retained runner was exercised with Python 3.10.0 and the exact versions in [`requirements-reproduction.txt`](requirements-reproduction.txt). A fresh Python 3.10 virtual environment was installed and the foam run and full audit passed on macOS arm64. Other operating systems have not been verified. The runner uses Unix `SIGALRM` for its per-trial wall limit, so use macOS or Linux.

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export LC_ALL=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p runs
python -B run_simple.py --object 14 --seed 0 --output runs/foam-14-0
```

The run is headless by default; the CLI has no GUI switch. Each output path must be new because the runner refuses to overwrite existing results. Choose another path for each run.

To run another supplied object and seed, pass an object ID from `0` through `14` and an integer seed from `0` through `4294967295`, for example:

```sh
python -B run_simple.py --object 5 --seed 60 --output runs/banana-5-seed-60
```

IDs follow the lexicographic order of the directories in `assets/ycb_objects/`: 0 master chef can, 1 sugar box, 2 tomato soup can, 3 mustard bottle, 4 potted meat can, 5 banana, 6 strawberry, 7 apple, 8 bleach cleanser, 9 bowl, 10 mug, 11 plate, 12 large marker, 13 hammer, 14 foam brick.

## Audit a run

The base audit independently checks recorded metrics, source pins, and scene evidence. The full audit also checks the recorded contact witnesses against the URDF and trace. Save each report to a new path.

```sh
python -B -m evaluation.palm_faces043.audit runs/foam-14-0 --output runs/foam-14-0-base-audit.json
python -B verify_run.py runs/foam-14-0 --output runs/foam-14-0-full-audit.json
```

Both commands require a new `--output` path. `verify_run.py` runs the full base audit plus the contact-witness checks and writes a separate JSON file, so simulator diagnostics cannot corrupt the report. Audit an early-contact run in the installation where it was generated: the isolated lift request records absolute asset paths, so moving its evidence to another installation can require path remapping.

## Evaluation

The controller was fixed before the fresh-seed validation. Each cohort contains all 15 supplied objects and ten seeds, with one outcome per object/seed pair. Development and validation results are reported separately.

| Outcome (each out of 150) | Development: seeds 60–69 | Fresh random-seed validation |
|---|---:|---:|
| Strict pickup, calls | 90 (60.0%) | 87 (58.0%) |
| Held delivery, post-pick calls | 60 (40.0%) | 55 (36.7%) |
| Strict pickup, physics ticks | 87 (58.0%) | 85 (56.7%) |
| Held delivery, both tick windows | 57 (38.0%) | 53 (35.3%) |
| Both tick-timed stages | 57 (38.0%) | 53 (35.3%) |

One environment call advances five physics ticks. Pickup is scored within 3,000 calls or 3,000 ticks from reset. Call-timed delivery must occur within 15,000 calls after pickup. The stricter tick delivery measure requires arrival within 15,000 ticks both from reset and after pickup. The same 50 eligible call-boundary pickup observations are required under either clock. All 55 validation deliveries fit the post-pick tick window; two mustard-bottle cases exceed the whole-trial tick window and separate tick pickup limit.

### Development cases

[`results/cases.csv`](results/cases.csv) contains the original 150-case development matrix and trace identities. [`results/summary.json`](results/summary.json) contains its totals and per-object counts. These cases informed development and are not an unseen holdout.

### Fresh-seed validation

Ten distinct seeds were drawn with `secrets.randbelow(2**32)` and recorded before execution, excluding previously recorded seeds. All 150 planned trials completed. Every outcome, including physical failures, was retained. There was no controller tuning, seed replacement or retry. All 150 full audits and separate trace recounts passed; audit PASS establishes evidence consistency rather than task success.

- [`results/validation/cases.csv`](results/validation/cases.csv): all trial outcomes, timing counts and trace hashes.
- [`results/validation/summary.json`](results/validation/summary.json): both-clock totals, per-object/per-seed results and failure categories.
- [`results/validation/seeds.txt`](results/validation/seeds.txt): the ten actual seed inputs, in selection order.
- [`results/validation/protocol.json`](results/validation/protocol.json): a portable export of the preregistered selection and fixed evaluation protocol, with the original protocol hash.
- [`results/validation/audit_receipts.json`](results/validation/audit_receipts.json) and [`results/validation/independent_recounts.json`](results/validation/independent_recounts.json): all 150 audit receipts and independent outcome recounts.
- [`results/validation/SHA256SUMS`](results/validation/SHA256SUMS): checksums of the published validation files.

The fresh seeds produced ten distinct maze/goal configurations, one also present in development. Both cohorts use the supplied objects and the same generator, whose internal maze-generation seed is fixed. They do not test unseen objects, sensor noise, arbitrary maze topologies or a real robot. The controller source manifest and all 226 runtime file hashes are unchanged from validation; this publication adds documentation and evidence only. Full raw traces and original preregistration records are archived locally.

For example, reproduce one validation case with a fresh output path:

```sh
python -B run_simple.py --object 5 --seed 1213611890 --output runs/banana-validation-example
python -B verify_run.py runs/banana-validation-example --output runs/banana-validation-example-audit.json
```

To reproduce the full validation set sequentially after installation:

```sh
while IFS= read -r trial_seed; do
  for object_id in $(seq 0 14); do
    case_dir="runs/validation-${trial_seed}-${object_id}"
    python -B run_simple.py --object "$object_id" --seed "$trial_seed" --output "$case_dir" || break 2
    python -B verify_run.py "$case_dir" --output "${case_dir}-audit.json" || break 2
  done
done < results/validation/seeds.txt
```

A separate protected compatibility set scored 14 pickups and 9 held deliveries by calls, compared with 16 and 11 for its historical reference. Those development regression checks have a different object balance and are not pooled with either 150-case cohort.

## Supplied environment and attribution

The simulator and supplied Fetch/YCB assets originate from the [NUS LinS Lab course starter](https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project). The supplied object models are adapted from the [YCB Benchmarks](https://www.ycbbenchmarks.com/). Third-party Python dependencies are listed in `requirements-reproduction.txt`. No separate project license was supplied with the starter checkout.

## Verification

A fresh installation of the pinned dependencies in an isolated Python 3.10 environment passed the foam demonstration and full audit. The complete 10,725-row trace is identical to the retained demonstration: pickup at 383 calls / 1,915 ticks, held arrival at 2,145 calls / 10,725 ticks. See [`verification/fresh_install.json`](verification/fresh_install.json) and [`verification/fresh_install_audit.json`](verification/fresh_install_audit.json). The frozen runtime file hashes are in [`verification/runtime_sha256.json`](verification/runtime_sha256.json); the controller source manifest is `source.json`.

## Assessment notes

### Inspect the outcome

A trial saves `result.json` (termination, guard, pickup and goal times),
`trace.jsonl.gz` (one observation per physics tick), `scene.json` and
`maze.urdf` (the actual scene), `surface.npz` (points and normals), and source
identities. The full audit writes a separate JSON report, including:

| Field | Interpretation |
|---|---|
| `status` | Whether the saved evidence passes the audit |
| `call_strict_pick` | Qualified pickup within 3,000 environment calls |
| `call_held_delivery` | Held goal arrival within 15,000 calls after pickup |
| `tick_strict_pick` | Qualified pickup within 3,000 physics ticks |
| `both_tick_delivery` | Held delivery within both whole-trial and post-pick 15,000-tick windows |
| `both_timed_stages` | Both tick-qualified pickup and delivery |
| `current_sources_match` | The saved controller sources match this checkout |

An audit `PASS` means the evidence is consistent, not that the robot succeeded.
Read the outcome flags and the recorded termination. A normally completed
runner can exit with code 0 even when a physical task fails; exceptions are
recorded separately in `error.json`. The audit does not assign project marks
or substitute for reviewing the separate rubric criteria.

One environment call advances five physics ticks. The report and saved
evaluation summarize both clocks rather than silently equating them. The
aggregate pickup criterion uses 50 consecutive eligible call-boundary
observations with bilateral contact, no table support, the required rise and
clearance, and no recorded guard. Goal arrival is measured in three dimensions
while held. See the audit and controller for the exact checks.

### Locate the implementation

| Component | Implementation |
|---|---|
| Point-cloud/normal grasp proposals | `simple_controller/geometry.py`, `grasps` |
| Executable grasp selection and approach | `simple_controller/pick.py`, `plan_pick` |
| Numerical IK and collision-checked joint segments | `simple_controller/native.py`, `Native.ik` and `Native.edge` |
| Closure, physical lift, contact checks and task sequencing | `simple_controller/controller.py`, `Controller` |
| Loaded planar navigation | `simple_controller/navigation.py` |
| Translation/yaw fallback search | `simple_controller/se2_planner.py` |
| Simulator action/observation interface | `pick_nav_reach_env.py`, `PickNavReachEnv.step` |
| Independent saved-run verification | `verify_run.py` and `evaluation/` |

The finite IK/path search is not a claim of complete motion planning. The development and fresh-seed validation cohorts remain separate, and neither guarantees success on every object or seed. Required simulator assets are included.
