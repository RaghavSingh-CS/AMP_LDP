# Stage 28 repository and paper-freeze audit

## Recovery and pre-stage state

- Branch: `study2/mil`.
- Source HEAD: `dbd978ffeebe7cee772f8ec693243895be7ff2c8`.
- `main`: `d0601c54f6e9d0f11caac2ebd27f7c86a568a863`.
- Starting tree: clean before Stage 28; after interrupted work, only the two
  intended `docs/paper` drafts were present.
- Stage 27B commit exists at the source HEAD.
- No historical result, model, source, split, or data artifact was modified.

## Frozen verification

| Item | Expected | Read-only result |
|---|---|---|
| Stage 25B execution spec | `006c781899f74f0f62d01282fbf4164ac3ee27d8a61e3f2ea5fcb09b9e445470` | Match |
| Stage 26 protocol | `7207a75dfd31d3254aaa3d4507eb9812b38ff87ecda09abfa62178572c8be09b` | Match |
| Stage 27B preregistration | `ed2457e3a6eaf4b683c3fd8aaf89cf1f1fca63500a68f0427279813533e92702` | Match |
| Frozen Stage 17B FULL_48 | `67ed869858185d8f71d99b3dce8b358c43793178185e5aa50e6997d11b56974b` | Match |
| Frozen Stage 25B DIRECT | `fa11424a6eb5615bc3ce696860a59d992114d891bb9d56bb3423693c172b4a45` | Match |
| Seed 42–46 encoders and latent HGBs | Stage 27A inventory | Stage 27B pre-access audit records all matches |

Study 1 protected artifacts and Study 2 Stage 24A–27B artifacts remain tracked
and unchanged. Stage 28 accessed aggregate reports and metadata only.

## Numerical consistency result

All central Study 1 and Study 2 values specified for Stage 28 match committed
aggregate sources. No raw-value discrepancy was found. Display differences are
rounding only. The paper policy is four decimals for ROC, AP, effects and CI
bounds, and integers for counts.

The audit preserves a critical distinction:

- Study 1 Stage 23B: refitted DEV_TRAIN+DEV_VAL FULL_48, final `0.7925/0.3702`.
- Study 2 Stage 27B comparator: frozen DEV_TRAIN Stage 17B FULL_48, final
  `0.7979/0.3853`.

The two checkpoints answer different governed questions and are not
interchangeable.

## Repository hygiene findings

### MUST FIX BEFORE PUSH

None.

### SHOULD FIX LATER

1. Consider Git LFS or an explicit long-term artifact-retention policy for the
   tracked Stage 21 random-forest checkpoint (`39,249,641` bytes) and large
   row-level diagnostic CSVs. They are below common single-file hosting limits
   and predate Stage 28, so they do not block this documentation commit.
2. Review intentional exact duplicates during a future repository-maintenance
   change: Stage 23B prediction, metric, and reference aliases; repeated frozen
   model copies in historical stage directories; and small Stage 27B naming
   aliases. Do not remove them without first updating manifests and provenance.
3. Add a citable STL-Net source and protocol audit if its numerical context is
   desired in the manuscript. Until then, omit the numbers.

### NO ACTION NEEDED

- `.venv`, `data`, IDE state, Python caches, and generated local cache paths are
  ignored and are not staged.
- No tracked temporary/debug/log/credential file was identified.
- The secret-pattern scan found no API key, password assignment, or private-key
  signature match in the scanned tracked/workspace documentation and source.
- No absolute local Windows, `/Users`, or `/home` path appears in the Stage 28
  paper package.
- Current paper-facing language consistently rejects pristine/untouched claims
  and window-level theft detection. Historical files contain contemporaneous
  “untouched” wording or explicit prohibitions; later governance supplies the
  required disclosure, so historical artifacts should not be rewritten.
- Tracked prediction and checkpoint artifacts were already committed as
  historical reproducibility evidence; Stage 28 adds none.

## Push readiness

**A — SAFE TO PUSH AFTER STAGE 28 COMMIT.**

Reason: Stage 28 adds documentation only; no secret, absolute local path,
untracked required dependency, current false cohort claim, hosting-limit breach,
or unintended historical modification was found. The remaining hygiene items
are non-blocking maintenance recommendations. This assessment authorizes no
push; the user explicitly prohibited pushing in Stage 28.

## Closed-study assertion

Study 1 experimental work is closed. Study 2 experimental work is closed.
No numerical or documentation inconsistency found here authorizes new training,
scoring, prediction generation, cohort access, threshold selection, or rescue
experiments.
