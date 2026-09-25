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

**Model.** Qwen3.5-4B in non-thinking mode, with greedy decoding and bf16 weights. It is a hybrid of Gated DeltaNet and attention layers, with 32 layers in total.

## Methods

**Decision measure.**
- m(x) is the log-odds that the first generated token opens a tool call.
- The first token agrees with the final action on 96.2% of prompts.

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
- L2 logistic regression on standardized residuals.
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
| C3 | Add a direction at layer 18 at every position; doses are multiples of the norm of `r_ref` |

Random directions serve as controls for C2, C3 and C6. For C2 and C3 they are orthogonal to the tested direction and matched in norm. In the positive control, adding `r_ref` at layer 18 makes the model refuse 100% of benign requests; random directions give 0%. Removing it lowers refusal of harmful requests from 81% to 70%, against about 80% with random directions.

**Statistics.** Estimates are averaged within skeleton first. 95% intervals come from a cluster bootstrap over skeletons, stratified by action class, with 10,000 resamples.

## Findings

All results are on the 216 development skeletons.

### Summary

- The model recognizes production.
- It represents production linearly, beyond lexical cues, and carries it to the decision token.
- The direction that encodes production can suppress action as strongly as the content-harm refusal direction.
- But the natural production signal is small:
  - it moves the action by 0.4 logits, against 5.2 for an explicit confirmation rule;
  - it moves the harmfulness direction by 2% of the content-harm gap.
- Harm vocabulary moves the action about as much as a true statement of the consequence.

### 1. Behavior

| Destructive action on production | Neutral rule | Confirmation rule |
|---|---|---|
| Executed as requested | 99.3% | 85.9% (94.4% with rollback, 77.3% without) |

| Effect on m(x), logits | Estimate |
|---|---|
| Confirmation policy | 5.24 [5.14, 5.33] |
| Rollback | 0.98 [0.92, 1.03] |
| Target, destructive against benign | 0.43 [0.38, 0.49] |
| Blast radius, production against staging | 0.37 [0.32, 0.42] (neutral rule); 0.43 [0.39, 0.47] (confirmation rule) |

- **Recognition without caution:** asked whether a request targets a production system, the model answers with AUROC 0.94 (0.91 to 1.00 by form). Yet under the confirmation rule, it asks before a destructive production action only 7 to 21% of the time, depending on form. On staging the rate is 2 to 10%.
- **Other behavior:**
  - over-caution, where acting is correct: 2.4%;
  - paired accuracy on production and staging twins under the confirmation rule: 7.6%;
  - shortcut framing, where the destructive command is only an available shortcut in a benign task: unsafe path taken 10.2% of the time.

### 2. Seen

- **Production shift:** production moves destructive requests by 0.024 [0.021, 0.026] content-harm gaps along `r_harm`, and by 0.015 [0.013, 0.017] along `r_ref`.
- **Absolute position:** all four target-by-environment cells sit at 0.34 to 0.39 on `r_harm`, benign staging included. The absolute position reflects the task type, not the danger.

### 3. Represented

Blast-radius probe, tested on unseen classes in unseen surface forms:

| Position, layer | How the point was chosen | Probe AUROC | n-gram baseline |
|---|---|---|---|
| `t_post`, 25 | Best leave-one-form-out score within the training forms | 0.682 | 0.516 |
| `t_inst`, 23 | Best score on this test partition (optimistic) | 0.796 | 0.516 |

- **Transfer:** unseen classes cost almost nothing (0.95 to 0.99 on the training forms). Unseen surface forms are the difficulty. The account-name form transfers worst (0.56 to 0.67), and it is also the weakest form for judgment (AUROC 0.91).
- **Other factors:** the target factor and the conjunction (destructive and production) are not decodable beyond words.
  - Target: n-gram baseline 0.85, probe 0.62 to 0.64.
  - Conjunction: n-gram baseline 0.88, probe 0.66 to 0.69.
- **Masking:** with the environment span masked, prompts read as staging.
- **Judgment and action:** a production direction taken from the judgment prompts separates production from staging in action prompts, with AUROC up to 0.90.
- **Geometry:**
  - At `t_post`, `r_blast` has cosine 0.49 to 0.52 with `r_ref` at layers 23 to 25. Its norm is only about 5% of the norm of `r_ref`.
  - The direction separating asks from executions has cosine 0.60 to 0.62 with `r_ref`.
  - The tool-schema direction (native JSON tools against the same tools written as prose) is nearly orthogonal to `r_ref` (cosine -0.03 to -0.11).

### 4. Wired

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

- **C3, steering:** this test uses 60 prompts per group under the neutral rule, and 8 random directions per tested direction. Change in m(x) on destructive production prompts (starting near +6.4):

| Direction added at layer 18 | 0.1 | 0.25 | 0.5 | 1 | 2 |
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

These results use destructive production actions without rollback, under the neutral rule.

- **Plain consequence note:** a note stating the consequence plainly ("this permanently deletes ..., it cannot be restored") lowers m(x) by 0.96 [0.85, 1.08].
- **Harm vocabulary:** the same information in harm vocabulary lowers m(x) a further 0.49 [0.40, 0.58]. It also moves the position on `r_ref` up by 0.067 [0.063, 0.070] gaps.
- **Cue-only controls:** on benign targets, harm wording alone lowers m(x) by 0.97 [0.84, 1.10], as much as a true consequence note. Slang ("just nuke ...") lowers it by 0.37 [0.27, 0.48].
- **Execution:** stays at 94 to 100% throughout. All of these effects are small next to the 5.2-logit policy effect.

## Repository

```
configs/          pinned model and data revisions; locked analysis values
docs/             benchmark specification, preregistration, project report
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
results/          one JSON summary per experiment
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

Setup: `uv sync`, then put a Hugging Face read token in `.env` as `HF_READ=...`. Generated data, run outputs, caches and model weights stay inside the folder and are not tracked.
