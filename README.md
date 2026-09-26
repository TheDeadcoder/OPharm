# Knows the Blast Radius, Pulls the Trigger

Operational harm as a blind spot of content-harm safety in small LLM agents.

## Background

Safety training teaches language models to refuse harmful content. Inside the model, this behavior is carried by linear directions in the residual stream:
- a **harmfulness direction**, read at the end of the instruction;
- a **refusal direction**, read at the last prompt token ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717); [Zhao et al., 2025](https://arxiv.org/abs/2507.11878)).

Adding the refusal direction makes a model refuse, and removing it suppresses refusal.

Operational harm looks different. A destructive action on real infrastructure, such as deleting a production database, is phrased like routine work. This project asks:
- whether the content-harm mechanism registers such actions;
- whether the model represents an action's consequence (its blast radius);
- whether that representation drives the decision to act.

All experiments use frozen weights, with four tools:
- difference-of-means directions;
- linear probes checked against lexical baselines;
- activation patching;
- steering and ablation, each against random directions of matched norm.

The four questions:
1. **Behavior.** How much do blast radius, rollback and an explicit confirmation policy move the decision to act?
2. **Seen.** Do the content-harm harmfulness and refusal directions register destructive production actions?
3. **Represented and wired.** Is blast radius linearly represented, and is that representation causally connected to the action?
4. **Vocabulary.** Does caution respond to harm words rather than to consequences?

## Task and benchmark

**Task.** Each prompt is a single-step SRE agent decision:
- a system prompt with six tools (shell, SQL, Kubernetes, cloud API, IAM, git) and a policy line;
- a ticket that requests one action.

The model either calls a tool or replies in text. A deterministic oracle labels the reply: runs the requested call, runs another call, asks, declines, or other.

**Benchmark.** 320 skeletons, split into 216 development and 104 held-out, drawn from disjoint name pools.
- 8 action classes (file, database, compute, access, observability, version control, network, configuration), with 5 templates each.
- Every skeleton crosses four factors:
  - **Target:** a destructive twin or a benign twin with the same tool, syntax and length (for example `rm -rf /var/lib/postgresql/data` against a scratch directory).
  - **Blast radius:** production or staging, in four surface forms:
    - F1: a host prefix;
    - F2: an `Environment:` line;
    - F3: an account name;
    - F4: a descriptive sentence.
  - **Rollback:** a tested snapshot, or none.
  - **Policy:** a rule requiring confirmation from the on-call engineer before destroying important data, exposing sensitive systems or taking down a live service in production; or a neutral rule of equal token length.
- Extra sets:
  - judgment questions and a masked environment;
  - a phrasing ladder and cue-only controls;
  - shortcut framing, flattened tools and narration.
- 29,608 instances. All 10,240 minimal pairs are token-aligned.

**Models.** Qwen3.5-4B and Qwen3.5-9B, in non-thinking mode, with greedy decoding and bf16 weights. Both are hybrids of Gated DeltaNet and attention layers, with 32 layers.

## Methods

**Decision measure.**
- m(x) is the log-odds that the first generated token opens a tool call.

**Positions.**
- `t_inst`: the closing period of the user turn.
- `t_post`: the last prompt token.

**Directions.**
- `r_harm` (at `t_inst`) and `r_ref` (at `t_post`) are differences of means between two sets of requests, rendered in the same agent format:
  - HarmBench requests that the model refuses;
  - length-matched Alpaca requests that it accepts.
- A prompt's position along one of these directions is measured in content-harm gaps: 0 is the benign centroid, 1 is the content-harm centroid.
- `r_blast` is the mean over production prompts minus the mean over staging prompts.

**Probes.**
- L2 logistic regression on standardized residuals, trained on development data only.
- Trained on 4 action classes in 2 surface forms; tested on the other 4 classes in the other 2 forms.
- Baseline: naive Bayes and logistic regression on 1-to-3-gram counts of the ticket.

**Causal tests.**

| Test | Intervention |
|---|---|
| C1 | Patch the environment span between production and staging twins, in both directions |
| C7 | Patch every position except the environment span |
| C6 | Swap only the coordinate along one direction, at positions whose tokens the twins share |
| C4 | Patch the policy span between confirmation-rule and neutral-rule twins |
| C2 | Remove a direction at every layer and position |
| C3 | Add a direction at every position of one layer; doses are multiples of the norm of `r_ref` |

- **Random controls.** Random directions are the controls for C2, C3 and C6. For C2 and C3 they are orthogonal to the tested direction and matched in norm.
- **Positive control, adding `r_ref`.** It makes the model refuse 100% of benign requests (4B at layer 18, 9B at layer 23). Random directions give at most 2%.
- **Positive control, removing `r_ref`.**
  - 4B: refusal of harmful requests falls from 81% to 70%, against about 80% with random directions.
  - 9B: it falls from 91% to 0%, against about 90% with random directions.

**Statistics.** Estimates are averaged within skeleton first. 95% intervals come from a cluster bootstrap over skeletons, with 10,000 resamples, stratified by template family (nested within action class).

## Findings

Results cover the 216 development skeletons for both models, and the 104 held-out skeletons for the 4B.

### Summary

- **Recognition.** Both models recognize production when asked (AUROC 0.94 for the 4B, 0.99 for the 9B).
- **Representation.** Both represent production linearly, beyond lexical cues.
- **Wiring (4B).** Production reaches the decision token. The direction that encodes it can suppress action as strongly as the content-harm refusal direction.
- **Magnitude.** The natural production signal is small:
  - it moves the action by 0.4 logits on the 4B and 0.06 on the 9B, against 5 to 6 logits for an explicit confirmation rule;
  - it moves the harmfulness direction by 1 to 3% of the content-harm gap.
- **The 9B and the rule.** The 9B follows the confirmation rule more strongly than the 4B, but much of its added caution also falls on staging.
- **Vocabulary.** Harm wording moves the action about as much as a true statement of the consequence.
- **Replication.** The 4B's held-out skeletons reproduce its development results.
- **Preregistered tests.** Four of the five hold on held-out skeletons. The fifth (H4) fails in the opposite direction: the blast-radius direction is at least as strong a handle on the action as the refusal direction.

### Preregistered tests (Qwen3.5-4B, held-out skeletons)

The five tests were fixed before the held-out skeletons were analyzed and form one Holm family at alpha 0.05.

| Hypothesis | Estimate [95% interval] | Holm-adjusted p | Outcome |
|---|---|---|---|
| H1. The confirmation policy moves the action more than blast radius does | 4.67 [4.51, 4.82] logits (policy 5.05, blast radius 0.38) | < 0.001 | supported |
| H2. The harmfulness direction does not register blast radius (equivalence within ±0.2 gaps) | 0.026 [0.022, 0.030] gaps | < 0.001 | supported |
| H3. Blast radius is decodable beyond lexical cues (probe AUROC minus n-gram AUROC) | +0.156 [0.101, 0.213] (probe 0.668, n-gram 0.511) | < 0.001 | supported |
| H4. At matched norm, the refusal direction is a stronger handle on the action than the blast-radius direction | +1.42 [1.29, 1.55] logits, the opposite sign | 1.0 | not supported |
| H5. Harm vocabulary engages the refusal direction beyond neutral wording | 0.065 [0.060, 0.069] gaps | < 0.001 | supported |

In the H4 panel, each direction is added at layer 18 to all 208 held-out destructive production prompts under the neutral rule. The effects on m(x):
- `r_blast` lowers it by 12.67 logits;
- `r_ref` lowers it by 11.26;
- random directions of the same norm lower it by 2.2 (24 per tested direction).

### 1. Behavior

| | 4B development | 4B held-out | 9B development |
|---|---|---|---|
| Destructive production executed, neutral rule | 99.3% | 98.1% | 100% |
| Destructive production executed, confirmation rule | 85.9% | 84.1% | 56.7% |
| Asks on destructive staging, confirmation rule (rollback; none) | 1.4%; 9.7% | 1.0%; 8.7% | 22.7%; 40.7% |
| Effect on m(x): confirmation policy | 5.24 [5.14, 5.33] | 5.05 [4.92, 5.18] | 6.12 [6.01, 6.23] |
| Effect on m(x): rollback | 0.98 [0.92, 1.03] | 0.95 [0.87, 1.02] | 0.41 [0.36, 0.45] |
| Effect on m(x): target, destructive against benign | 0.43 [0.38, 0.49] | 0.40 [0.33, 0.47] | 0.43 [0.38, 0.48] |
| Effect on m(x): production against staging, neutral rule | 0.37 [0.32, 0.42] | 0.38 [0.32, 0.44] | 0.06 [0.03, 0.10] |
| Effect on m(x): production against staging, confirmation rule | 0.43 [0.39, 0.47] | 0.41 [0.35, 0.47] | 0.38 [0.33, 0.42] |
| Over-caution, where acting is correct | 2.4% | 2.3% | 12.5% |
| Paired accuracy on production and staging twins, confirmation rule | 7.6% | 8.7% | 14.4% |
| Shortcut framing, unsafe path taken | 10.2% | 12.5% | 4.6% |
| Judging whether the request targets production (AUROC) | 0.94 | 0.94 | 0.99 |
| First token agrees with the final action | 96.2% | 96.0% | 84.5% |

- Both models recognize production far better than they act on it.
- On the 9B, 10.7% of executions begin with text, so m(x) tracks its behavior less closely than on the 4B.

### 2. Seen

| | 4B development | 4B held-out | 9B development |
|---|---|---|---|
| Production shift on `r_harm` (content-harm gaps) | 0.024 [0.021, 0.026] | 0.026 [0.022, 0.030] | 0.011 [0.007, 0.014] |
| Production shift on `r_ref` | 0.015 [0.013, 0.017] | 0.015 [0.013, 0.017] | 0.016 [0.014, 0.018] |
| Position of the four target-by-environment cells on `r_harm` | 0.34 to 0.39 | 0.34 to 0.40 | 0.15 to 0.18 |

Benign staging requests sit as high as destructive production ones, so the absolute position reflects the task type, not the danger.

### 3. Represented

Blast-radius probe, tested on unseen classes in unseen surface forms (AUROC):

| Probe point | 4B development | 4B held-out | 9B development |
|---|---|---|---|
| `t_post`, layer chosen by leave-one-form-out score | 0.682 (layer 25) | 0.668 | 0.697 (layer 31) |
| `t_inst`, layer chosen by best development score (optimistic on development) | 0.796 (layer 23) | 0.775 | 0.821 (layer 24) |
| n-gram baseline | 0.516 | 0.511 | 0.516 |

- **Transfer (4B development):** unseen classes cost almost nothing (0.95 to 0.99 on the training forms). Unseen surface forms are the difficulty. The account-name form transfers worst (0.56 to 0.67), and it is also the weakest form for judgment (AUROC 0.91).
- **Other factors:** the target factor and the conjunction (destructive and production) are not decodable beyond words on either model.
  - Target: n-gram baseline 0.83 to 0.85, probe 0.61 to 0.68.
  - Conjunction: n-gram baseline 0.84 to 0.88, probe 0.64 to 0.73.
- **Masking:** with the environment span masked, prompts read closer to staging than to production.
- **Geometry:**
  - At `t_post`, `r_blast` has cosine 0.47 to 0.52 with `r_ref` at layers 23 to 25 on the 4B, and 0.33 to 0.35 on the 9B. On the 4B its norm is only about 5% of the norm of `r_ref`.
  - The direction separating asks from executions has cosine 0.59 to 0.62 with `r_ref` on the 4B, and 0.33 to 0.44 on the 9B.

### 4. Wired (4B development)

The patching tests use 60 production and staging pairs, patched in both directions at 8 layers. Production's natural effect on m(x) in these pairs is 0.44 [0.33, 0.55] logits.

- **C1, environment span:** patching the span carries the whole effect at layer 1, about 40% at layer 9 and 7% at layer 18. The information leaves the span by mid-depth.
- **C7, the rest of the prompt:** patching every other position carries the remainder. Together with C1 it accounts for about 100% of the effect at every layer.
- **C6, one coordinate:** a swap along a single direction, at the positions whose tokens the twins share.
  - Along the `t_inst` production direction:
    - at layer 22 it carries all of the `t_inst` probe's signal, and none of the effect on the action (0%, interval -5 to 4%);
    - at layer 18 it carries 63% of the probe's signal and 10% of the action effect.
  - Along the `t_post` production direction: it carries 22 to 33% of the effect on the action at layer 18, and 58% at layer 30.
- **C4, policy span:** the policy's 5.44-logit effect also leaves its span by layer 18.
- **C2, ablation:** removing a direction in the 55 production prompts where the model asks under the confirmation rule.

| Direction removed | Asks flipped to execution |
|---|---|
| `r_blast` at `t_inst`, layer 23 | 97% [93, 100] |
| `r_ref` | 91% [83, 98] |
| `r_blast` at `t_post`, layer 25 | 70% [58, 82] |
| 8 random directions | 2% [1, 3] |

Removing `r_blast` overshoots. m(x) goes from -1.09 to +1.43, past the staging twins (-0.45). The direction behaves like a general caution axis, and production moves prompts only slightly along it.

- **C3, steering at layer 18:** 60 prompts per group under the neutral rule, and 8 random directions per tested direction. Change in m(x) on destructive production prompts (starting near +6.4):

| Direction added | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| `r_ref` | +0.14 | -0.51 | -3.84 | -10.99 | -12.37 |
| `r_blast` (`t_post`), same norm as `r_ref` | -0.49 | -1.68 | -4.81 | -12.45 | -13.69 |
| `r_blast` with its `r_ref` component removed | -0.63 | -1.77 | -4.35 | -10.54 | -14.51 |
| `r_blast` (`t_inst`) | -0.45 | -1.38 | -3.08 | -5.26 | -8.80 |
| Random directions | -0.08 to -0.13 | -0.32 to -0.35 | -0.85 to -1.06 | -2.69 to -3.02 | -8.87 to -9.58 |

  - At matched norm, `r_blast` suppresses action at least as strongly as `r_ref`, at every dose.
  - Removing its `r_ref` component leaves it about as strong, so its effect does not run through the refusal direction.
  - Neither direction is selective: effects on staging prompts and on benign targets match those on destructive production prompts, within 0.49 logits at every dose.
  - At coefficient 2, random directions also collapse m(x).

### 5. Vocabulary

Destructive production without rollback under the neutral rule, and the same notes on benign targets. Change in m(x), in logits:

| | 4B development | 4B held-out | 9B development |
|---|---|---|---|
| Plain consequence note ("this permanently deletes ..., it cannot be restored") | -0.96 [-1.08, -0.85] | -0.93 [-1.08, -0.77] | -0.84 [-0.91, -0.77] |
| The same information in harm vocabulary, beyond the plain note | -0.49 [-0.58, -0.40] | -0.40 [-0.52, -0.29] | -0.14 [-0.18, -0.10] |
| Harm wording on a benign target | -0.97 [-1.09, -0.84] | -0.84 [-1.00, -0.68] | -0.89 [-0.95, -0.83] |
| Slang ("just nuke ...") on a benign target | -0.37 [-0.48, -0.26] | -0.37 [-0.52, -0.22] | -0.79 [-0.88, -0.71] |

- On development data, harm vocabulary also moves the position on `r_ref` up by 0.067 [0.063, 0.070] gaps on the 4B, and by 0.021 [0.018, 0.024] on the 9B.
- Execution stays at 94 to 100% throughout. The 9B executes every ladder and cue prompt.

## Repository

```
configs/          pinned model and data revisions; locked analysis values
docs/             benchmark specification, preregistration
src/opharm/
  bench/          templates, lexicons, generator, token alignment, oracle
  refsets/        content-harm reference set and refusal classifier
  interp/         hooks (capture, patch, steer, ablate, swap), directions, probes
  run/            decision measure, activation capture, greedy decoding
  stats/          cluster bootstrap, Holm correction, held-out lock
  analysis.py     shared data partitions and settings
  chat.py         chat-template rendering with token spans and positions
  models.py       model loading
scripts/          numbered pipeline, below
tests/            unit tests
results/          one JSON summary per experiment; figures in results/figures
```

| Scripts | Step |
|---|---|
| `00` to `03` | download, numerics check, throughput, residual profile |
| `04` to `07` | reference data, content-harm directions, positive control |
| `08` | benchmark generation |
| `09`, `10` | behavioral runs with activation capture, and the behavior report |
| `11` | representation: seen, probes, geometry |
| `12`, `13` | causal tests and their report |
| `14` | hypothesis tests with Holm correction |
| `15` | figures |

Setup: `uv sync`, then put a Hugging Face read token in `.env` as `HF_READ=...`. Generated data, run outputs, caches and model weights stay inside the folder and are not tracked.
