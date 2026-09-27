# Preregistration amendment 2

**Registered documents:** `docs/prereg.md` at commit f2ea753 and `docs/prereg_amendment_1.md` at commit 533aca3. Both stay unedited.

**Purpose:**
- Lock the Qwen3.5-9B replication values, as amendment 1, section 6 announced.
- Turn the Qwen3-4B-Instruct-2507 comparator into a locked replication.
- Add Llama-3.1-8B-Instruct and Gemma-4-E4B-it, from two other model families, as locked replications.

**Status:**
- Written on 2026-09-27. No held-out row of any of these four models has been analyzed.
- The lock hash now also covers this file. Held-out rows of every model stay sealed until `prereg.lock` is regenerated after this file is committed.

**Observed since amendment 1:**
- Qwen3.5-4B, held-out: the registered confirmatory analysis and its secondaries, run as amendment 1 specifies (commit 032099c).
- Qwen3.5-4B, development: smoke tests of two exploratory analyses, gain control along `r_blast` (4 prompts) and three rule variants (one skeleton).
- Qwen3.5-9B: the full grid, with held-out outputs written but not analyzed. On development data: behavior, representation, content directions, the C5 control (G1 passed at layer 23) and a dry run of H1, H2, H3 and H5.
- Qwen3-4B-Instruct-2507: the same (G1 passed at layer 26).
- Llama and Gemma: minimal-pair token alignment and hook structure checks on models without weights. No model output.

Development dry runs, estimate [95% interval]. H4 was not run on development data for either model.

| Model | H1 | H2 | H3 | H5 |
|---|---|---|---|---|
| Qwen3.5-9B | 6.06 [5.94, 6.18] | 0.011 [0.007, 0.014] | +0.180 [0.133, 0.232] | 0.021 [0.018, 0.024] |
| Qwen3-4B-Instruct-2507 | 7.18 [6.95, 7.43] | 0.009 [0.006, 0.013] | +0.124 [0.099, 0.157] | 0.042 [0.039, 0.045] |

## 1. Qwen3.5-9B: locked values

Set by the registered rules on its development data:

| Value | Source | 9B |
|---|---|---|
| L_harm | `results/refsets_qwen35_9b.json` | 14 |
| L_ref | `results/refsets_qwen35_9b.json` | 14 |
| L_steer | `results/c5_qwen35_9b.json` | 23 |
| L_blast, `t_post` | `results/representation_qwen35_9b_grid.json` | 31 |
| Dev-best probe point (amendment 1, section 2) | `results/representation_qwen35_9b_grid.json` | `t_inst`, 24 |

- On development data the probe reaches AUROC 0.697 at the registered point and 0.821 at the dev-best point. The dev-best value is the maximum over 64 candidates and therefore optimistic.
- H4 uses coefficient 1 and 24 random directions per tested direction.
- Patch layers: 1, 5, 9, 13, 18, 22, 26, 30.

## 2. Qwen3-4B-Instruct-2507: locked replication

The registered text calls the comparator descriptive. It becomes a locked replication under the same rules, before any of its held-out rows are analyzed. Reason: it is a dense model from the previous Qwen generation, so it separates the phenomenon from the hybrid Gated DeltaNet architecture.

| Value | Comparator |
|---|---|
| L_harm | 28 |
| L_ref | 19 |
| L_steer | 26 |
| L_blast, `t_post` | 21 |
| Dev-best probe point | `t_inst`, 11 |

- Development probe AUROC: 0.640 at the registered point and 0.721 at the dev-best point (the maximum over 72 candidates).
- Patch layers: 1, 6, 10, 15, 20, 25, 29, 34.

## 3. Added models: Llama-3.1-8B-Instruct and Gemma-4-E4B-it

| Model | Revision | Layers |
|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f2 | 32 |
| google/gemma-4-E4B-it | ee0ef602 | 42 |

Registered before any of their grid outputs are analyzed. Everything follows `docs/prereg.md` and amendment 1, except:
- **m(x):** the log-odds that the first token opens a tool call, over each model's opener set: Llama `{`, `{"` and `<|python_tag|>`; Gemma `<|tool_call>`. The probability of a set is the sum over its tokens.
- **Labels:** the same oracle. The parser reads Llama's JSON calls and Gemma's native call syntax.
- **Positions:** `t_inst` and `t_post` as defined, under each model's chat template.
- **Runtime:** transformers 5.17.0, bf16 on MPS, greedy decoding, non-thinking mode.

Procedure, per model:
1. Validation: numerics against a CPU bf16 reference, hook tests and a one-skeleton smoke run.
2. Content directions and the C5 control. If G1 fails, H4 is not tested on that model, and its Holm family has four members.
3. The full grid. Held-out outputs stay sealed.
4. Development analyses. The layer values follow the rules in `docs/prereg.md`, section 3, and amendment 1, section 2.
5. The values go into `configs/locked.yaml` in a commit that changes nothing else. `prereg.lock` is regenerated only after that commit, and only then does the held-out analysis run. The code at that commit is the model's registered analysis.

Token alignment of the minimal pairs is fixed by the tokenizers and was checked before any model output (`results/alignment_new_tokenizers.json`):
- Llama: 7,680 of 10,240 pairs align. The rollback pairs differ in length.
- Gemma: 7,320 of 10,240 align. The policy pairs and 360 target pairs differ.
- H1 to H5 do not need aligned pairs. Patching uses aligned pairs only, so C4 cannot run on Gemma.

## 4. Replication hypotheses and reporting

- Each replication model tests H1 to H5 with the estimators, tests and margins of `docs/prereg.md`, section 4, and amendment 1, section 3. Each model is its own Holm family at familywise alpha 0.05.
- A hypothesis replicates on a model when that model's Holm-adjusted test rejects in the registered direction. Results are reported per model and never pooled.
- The Qwen3.5-4B results remain the primary evidence.
- Secondary, for estimation only: the behavior and representation secondaries on held-out skeletons. Held-out causal secondaries are not registered for the replications.

## 5. Bootstrap strata

- The registered code stratifies by template family, the skeleton-id prefix. The registered text and amendment 1 say action class.
- The two coincide except that the compute class has two template families, which gives 9 strata instead of 8. Every stratum nests within one class.
- The replications use the registered code. Class-stratified intervals are reported as a sensitivity check (`14_confirm.py --strata class`).
- On the Qwen3.5-4B held-out analysis, no interval bound moves by more than 0.003 and no conclusion changes.

## 6. Analysis code

The code at the commit that finalizes this amendment is the registered analysis for Qwen3.5-9B and the comparator. For Llama and Gemma, it is the code at the commit that adds their locked values (section 3).

For each model `M` (`qwen35_9b` and `qwen3_4b_2507`; `llama31_8b` and `gemma4_e4b` once their values are locked):

Primary:
```
uv run python scripts/12_causal.py c3 M --tag grid --confirm --groups DP --coefs 1 --seeds 24 --n 1000 --suffix h4
uv run python scripts/14_confirm.py M --tag grid --confirm
```

Sensitivity and secondary:
```
uv run python scripts/14_confirm.py M --tag grid --confirm --strata class
uv run python scripts/10_pilot_report.py M --tag grid --confirm
uv run python scripts/11_representation.py M --tag grid --confirm
```
