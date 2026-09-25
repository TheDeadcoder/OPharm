# Preregistration amendment 1

**Registered document:** `docs/prereg.md` at commit f2ea753 (2026-09-25 00:12 +06). That file stays unedited. This file records every change and clarification made since.

**Status:**
- Written on 2026-09-25 and finalized the same day, before unlock.
- Held-out skeletons are sealed.
- The lock hash covers `docs/prereg.md`, this file and `configs/locked.yaml`, so no held-out analysis can run before the final version of this amendment is committed.

**Observed since registration** (all on development skeletons, reference sets or the Qwen3.5-9B validation):
- Content directions and the C5 control, re-run on the final prompts.
- The representation analyses (`scripts/11_representation.py`).
- Causal panels C1 to C4, including the C3 dose ladder (coefficients 0.1 to 2).
- Exploratory panels C6 and C7 (section 4).
- A dry run of the confirmatory analysis. On development data, H4 gives +1.45 [1.32, 1.59], the opposite sign to the prediction.
- Qwen3.5-9B validation (numerics, hook tests, content-harm directions, C5) and the start of its behavioral grid.

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

**Statistics**
- Every estimator, including the secondary ones, is averaged within skeleton first. The bootstrap resamples skeletons within class, 10,000 times, and intervals are percentile intervals. A patching fraction is a ratio of skeleton-averaged means.
- A one-sided bootstrap p-value is (1 + resamples on the null side of the null value) / (1 + 10,000). The H2 equivalence p-value is the larger of the two one-sided p-values against -0.2 and 0.2.
- Holm is applied only when all five p-values exist.

**H3 and H4**
- H3: the probe and the baseline are fit once on development data. The bootstrap resamples held-out test skeletons, and the better baseline is taken within each resample.
- H4 covers all held-out prompts in cells D,P,A,N and D,P,N,N.
  - `r_blast` is taken at `t_post` and L_steer, the same position and layer as `r_ref`.
  - There are 24 random directions per tested direction, each orthogonal to it and matched in norm.

**Secondary analyses**
- C3 adds two directions at the same norm: `r_blast` with its `r_ref` component removed, and `r_blast` taken at `t_inst`.
- The `r_ask` contrast and the C2 rows use prompts labeled ASK, as registered. Non-executions labeled EXEC_OTHER are excluded.
- The behavior secondaries use held-out skeletons only. They include the phrasing-ladder and cue-only effects on m(x):
  - plain note against no note;
  - harm wording against plain note;
  - harm and slang cues against no note.
- Correction before unlock: the behavior report's confirm mode had pooled development and held-out rows. It now reads held-out rows only.

## 4. Analysis code

The code at the commit that finalizes this amendment is the registered analysis. With `--confirm`, every probe and direction is fit on development skeletons and evaluated on held-out skeletons, and layer and position values come from `configs/locked.yaml`.

Primary:
```
uv run python scripts/12_causal.py c3 qwen35_4b --tag grid --confirm --groups DP --coefs 1 --seeds 24 --n 1000 --suffix h4
uv run python scripts/14_confirm.py qwen35_4b --tag grid --confirm
```

Secondary:
```
uv run python scripts/10_pilot_report.py qwen35_4b --tag grid --confirm
uv run python scripts/11_representation.py qwen35_4b --tag grid --confirm
uv run python scripts/12_causal.py c1 qwen35_4b --tag grid --confirm
uv run python scripts/12_causal.py c4 qwen35_4b --tag grid --confirm
uv run python scripts/12_causal.py c2 qwen35_4b --tag grid --confirm
uv run python scripts/12_causal.py c3 qwen35_4b --tag grid --confirm --coefs 0.1,0.25,0.5,1,2
uv run python scripts/13_causal_report.py qwen35_4b --tag grid --confirm
```

Exploratory, not registered:
```
uv run python scripts/12_causal.py c6 qwen35_4b --tag grid --confirm
uv run python scripts/12_causal.py c6 qwen35_4b --tag grid --confirm --dirs r_blast_post --suffix post
uv run python scripts/12_causal.py c7 qwen35_4b --tag grid --confirm
```

## 5. H4 steering coefficient

Coefficient 1, as registered.
- On the development dose ladder, `r_blast` is at least as strong as `r_ref` at every coefficient from 0.1 to 1, so the coefficient does not change the direction of the result.
- Coefficient 1 is near saturation but not at it: `r_ref` gives -11.0 at 1 and -12.4 at 2.
- Keeping the registered value avoids a post hoc choice.

## 6. Qwen3.5-9B

The replication's layer and position values will be set by the registered rules on its development data. They will be locked in a later amendment, before any of its held-out rows are analyzed.
