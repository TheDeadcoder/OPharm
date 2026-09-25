# Preregistration amendment 1 (draft for your review)

**Registered document:** `docs/prereg.md` at commit f2ea753 (2026-09-25 00:12 +06). That file stays unedited. This file records every change and clarification made since.

**Status when written (2026-09-25):**
- Held-out skeletons are sealed.
- `configs/locked.yaml` and `prereg.lock` do not exist yet. The lock hash covers `docs/prereg.md`, this file and `configs/locked.yaml`, so no held-out analysis can run before this amendment is committed.

**Observed since registration** (all on development skeletons or reference sets):
- Content directions and the C5 control, re-run on the final prompts.
- The representation analyses (`scripts/11_representation.py`).
- Causal panels C1 to C4.
- A dry run of the confirmatory analysis.

## 1. Deviation

During the completeness check of the grid run, a script printed aggregate label counts over all 6,368 decoded instances, development and held-out together. The counts were not split by any factor, so no confirmatory contrast could be read from them. All later analysis code reads only development rows until unlock.

## 2. Added secondary analysis: the blast-radius probe at a development-selected point

Registered H3 is unchanged. The probe sits at L_blast, `t_post`, and L_blast is chosen by leave-one-form-out AUROC within the training partition.

On development data this rule chose point 25. There the probe reaches 0.682 on the doubly held-out partition, against 0.516 for the n-gram baseline. A decomposition showed that the F1/F2 split does not predict transfer to F3 and F4:
- Class shift barely matters: 0.95 to 0.99 on training forms.
- F4 transfers well. F3, the account-name form, transfers poorly (0.56 to 0.67); F3 is also the weakest form for environment judgment (0.907).
- `t_inst` transfers better than `t_post`.

Added, reported beside the registered analyses and outside the Holm family:
- **The dev-best probe.** The H3 probe, test and baseline at the point that maximizes development doubly held-out AUROC over positions {`t_inst`, `t_post`} and all layers. "Development doubly held-out" means development skeletons of the H3 test classes in the H3 test forms. On development data this is `t_inst`, point 23, with AUROC 0.796. That value is the maximum over 64 candidates and therefore optimistic.
- **Causal readouts.** C1 and C4 read out both probes. C2 ablates `r_blast` taken from both points.
- **Factor probes.** The rollback, target and conjunction probes and the masked-span evaluation are reported at both points.

## 3. Clarifications (points the registered text left open)

- Every estimator is averaged within skeleton first. The bootstrap resamples skeletons within class, and intervals are percentile intervals.
- A one-sided bootstrap p-value is (1 + resamples on the null side of the null value) / (1 + 10,000). The H2 equivalence p-value is the larger of the two one-sided p-values against -0.2 and 0.2.
- H3: the probe and the baseline are fit once on development data. The bootstrap resamples held-out test skeletons, and the better baseline is taken within each resample.
- H4 covers all held-out prompts in cells D,P,A,N and D,P,N,N. `r_blast` is taken at `t_post` and L_steer, the same position and layer as `r_ref`.
- C3 adds two directions at the same norm: `r_blast` with its `r_ref` component removed, and `r_blast` taken at `t_inst`.
- The `r_ask` contrast and the C2 rows use prompts labeled ASK, as registered. Non-executions labeled EXEC_OTHER are excluded.

## 4. Analysis code

- `scripts/14_confirm.py` computes H1 to H5 and the Holm family.
- Scripts 11, 12 and 13 compute the secondary analyses.
- With `--confirm`, every probe and direction is fit on development skeletons and evaluated on held-out skeletons. Layer and position values come from `configs/locked.yaml`.
- The code at the commit that adds this file is the registered analysis.

## 5. Open before commit

- The H4 steering coefficient. The development steering panel will show whether coefficient 1 saturates m(x).
