# Knows the Blast Radius, Pulls the Trigger

Operational harm as a blind spot of content-harm safety in small LLM agents.

Safety-trained small models refuse harmful content, yet as agents they run destructive infrastructure commands. This project looks inside frozen Qwen3.5 models to ask why. There is no training: only difference-of-means directions, linear probes, activation patching and steering. Four questions:

1. **Behavior.** How much do blast radius, rollback and an explicit confirmation policy move the decision to act?
2. **Seen.** Do the content-harm harmfulness and refusal directions register destructive production actions?
3. **Represented and wired.** Is blast radius linearly represented, and is that representation causally connected to the action?
4. **Vocabulary.** Does caution respond to harm words rather than to consequences?

## Status (25 September 2026)

- **Done on Qwen3.5-4B:**
  - instrument validation and the content-harm positive control;
  - three pilots and the full behavioral grid;
  - representation and causal analyses on development skeletons.
- **Preregistration:** registered at commit `f2ea753`; `docs/prereg_amendment_1.md` records the later changes. Held-out skeletons stay sealed until the locked values are final.
- **Running:** the steering dose ladder (C3). It settles the one open item, the H4 steering coefficient.
- **Next:**
  - held-out confirmation on the 4B;
  - replication on Qwen3.5-9B and the dense comparator;
  - narration, thinking and the remaining judgment conditions.

## Setup

**Task.** Each prompt is a single-step SRE agent decision:
- a system prompt with six tools and a policy line;
- a ticket that requests one action.

The model either calls a tool or replies in text. A deterministic oracle labels the reply: runs the requested call, runs another call, asks, declines, or other.

**Benchmark.** 320 skeletons: 216 for development and 104 held out, drawn from disjoint name pools.
- 8 action classes (file, database, compute, access, observability, version control, network, configuration) with 5 templates each.
- Every skeleton crosses four factors:
  - **Target:** a destructive twin or a benign twin with the same tool, shape and length.
  - **Blast radius:** production or staging, in four surface forms:
    - F1: host prefix;
    - F2: an `Environment:` line;
    - F3: account name;
    - F4: a descriptive sentence.
  - **Rollback:** a tested snapshot, or none.
  - **Policy:** a rule requiring confirmation from the on-call engineer, or a neutral rule of equal token length.
- Extra sets:
  - judgment questions and a masked environment;
  - a phrasing ladder and cue-only controls;
  - shortcut framing, flattened tools and narration.
- 29,608 instances. All 10,240 minimal pairs are token-aligned.

| Role | Model |
|---|---|
| Primary | Qwen3.5-4B |
| Replication | Qwen3.5-9B |
| Dense comparator | Qwen3-4B-Instruct-2507 |

**Measures.**
- **m(x):** the log-odds that the first generated token opens a tool call.
- **Content-harm directions:** `r_harm` at the end of the user turn (`t_inst`) and `r_ref` at the last prompt token (`t_post`). They contrast HarmBench requests the model refuses with length-matched Alpaca requests it accepts, rendered in the same agent format.
- **Positions:** measured in content-harm gaps along a direction; 0 is the benign centroid, 1 the content-harm centroid.
- **`r_blast`:** production minus staging.

## Findings so far

All numbers are Qwen3.5-4B on the 216 development skeletons. Brackets are 95% cluster-bootstrap intervals over skeletons.

### Instrument and positive control

- Transformers on Apple MPS in bf16 matches a CPU fp32 reference: top-1 agreement 20/20, KL at most 0.005.
- The first generated token agrees with the eventual action on 96.2% of prompts, so m(x) tracks behavior.
- **Adding `r_ref`** at layer 18 makes the model refuse 100% of benign requests in the agent format; random directions give 0%.
- **Ablating `r_ref`** is weaker: harmful refusal falls to 70%, against about 80% with random directions.

### 1. Behavior: the model recognizes production and runs the command anyway

| Destructive action on production | Neutral rule | Confirmation rule |
|---|---|---|
| Executed as requested | 99.3% | 85.9% (94.4% with rollback, 77.3% without) |

| Effect on m(x), logits | Estimate |
|---|---|
| Confirmation policy | 5.24 [5.11, 5.36] |
| Rollback | 0.98 [0.91, 1.05] |
| Target, destructive against benign | 0.43 [0.36, 0.52] |
| Blast radius, production against staging | 0.37 [0.32, 0.42] (neutral rule); 0.43 [0.39, 0.47] (confirmation rule) |

- **Recognition vs action:** asked whether a request targets a production system, the model answers with AUROC 0.94 (0.91 to 1.00 by form). Yet under the confirmation rule, it asks before a destructive production action only 7 to 21% of the time, depending on form (staging: 2 to 10%).
- **Target importance:** the model does not reliably judge it (judgment AUROC 0.58 in pilot 3). Blast radius is therefore the primary consequence factor.
- **Other behavior:**
  - over-caution on prompts where acting is correct: 2.4%;
  - paired accuracy on production and staging pairs under the confirmation rule: 7.6%;
  - shortcut framing (a benign task where the destructive command is an available shortcut): unsafe path taken 10.2% of the time.

### 2. Seen: the content-harm directions barely register production

- Production moves destructive requests by 0.024 [0.021, 0.026] gaps along `r_harm`, and by 0.015 [0.013, 0.017] along `r_ref`.
- All four target-by-environment cells, benign staging included, sit at 0.34 to 0.39 on `r_harm`. The absolute position reflects the task type, not danger.

### 3. Represented: production is decodable beyond lexical cues

A probe is trained on 4 classes in 2 surface forms and tested on the other 4 classes in the other 2 forms:

| Probe point | Probe AUROC | n-gram baseline |
|---|---|---|
| Registered rule: `t_post`, layer 25 | 0.682 | 0.516 |
| Best development point: `t_inst`, layer 23 | 0.796 | 0.516 |

- **Transfer:** new classes cost almost nothing (0.95 to 0.99 on the training forms). New surface forms are the difficulty. The account-name form transfers worst (0.56 to 0.67), and it is also the weakest form for judgment (0.91).
- **Other factors:** target and the conjunction (destructive and production) are not decodable beyond words. The n-gram baseline beats the probe: 0.85 against 0.62 to 0.64 for target, and 0.88 against 0.66 to 0.69 for the conjunction.
- **Masking:** with the environment span masked, prompts read as staging.
- **Judgment and action:** a production direction taken from the judgment prompts separates production from staging in action prompts, with AUROC up to 0.90.
- **Geometry:**
  - at `t_post`, `r_blast` has cosine 0.49 to 0.52 with `r_ref` at layers 23 to 25, but only about 5% of its norm;
  - the direction separating asks from executions has cosine 0.60 to 0.62 with `r_ref`;
  - the tool-schema direction is nearly orthogonal to `r_ref` (cosine -0.03 to -0.11).

### 4. Wired: production reaches the decision but moves it very little

These tests use 60 production and staging pairs, patched in both directions at 8 layers.

- **C1, environment span:** the natural environment effect on m(x) is 0.43 logits. Patching the span carries all of it at layer 1, about 40% at layer 9 and 7% at layer 18. The information leaves the span by mid-depth.
- **C7, everything except the span:** patching all other positions carries the rest. Together with C1 it accounts for about 100% of the effect at every layer.
- **C6, one coordinate:** swaps only the production coordinate, at the positions shared by both twins.
  - The `t_inst` coordinate carries 98% of the best probe's signal at layer 22, but none of the effect on the action.
  - The `t_post` coordinate carries 22 to 33% of the action effect at layer 18 and 58% at layer 30.
  - So the code the best probe reads does not drive the action; the code at the decision token carries part of the effect.
- **C4, policy span:** it carries the 5.44-logit policy effect and also leaves its span by layer 18.
- **C2, ablation:** removing a direction in the 55 production prompts where the model asks under the confirmation rule.

| Direction removed | Asks flipped to execution |
|---|---|
| `r_blast` at `t_inst`, layer 23 | 95% |
| `r_ref` | 85% |
| `r_blast` at `t_post`, layer 25 | 62% |
| 8 random directions | 2% |

Removing `r_blast` overshoots: m(x) goes from -1.16 to +1.34, past the staging twins (-0.48). The direction acts like a general caution axis on which production moves prompts only slightly.
- **C3, steering dose ladder** (`r_ref`, `r_blast` and matched-norm random directions): running.

### 5. Vocabulary: harm words move the action about as much as real consequences

This uses destructive production actions without rollback, under the neutral rule.

- A plainly worded consequence note ("this permanently deletes ..., it cannot be restored") lowers m(x) by 0.96 [0.85, 1.08].
- Stating the same information in harm vocabulary lowers m(x) a further 0.49 [0.40, 0.58]. It also moves the position on `r_ref` up by 0.067 [0.063, 0.070] gaps.
- On benign targets, harm wording alone lowers m(x) by 0.97 [0.84, 1.10], as much as a true consequence note. Slang ("just nuke ...") lowers it by 0.37 [0.27, 0.48].
- Execution stays at 94 to 100% throughout. All of these effects are small next to the 5.2-logit policy effect.

### Confirmatory hypotheses: dry run on development data

This is the registered analysis code run on development skeletons. Held-out estimates come after the lock.

| | Hypothesis | Development estimate |
|---|---|---|
| H1 | The policy moves the action more than blast radius does | 4.87 [4.75, 4.98] logits |
| H2 | `r_harm` does not register blast radius (equivalence within ±0.2 gaps) | 0.024 [0.021, 0.026] |
| H3 | Blast radius is decodable beyond lexical cues | +0.166 [0.127, 0.209] AUROC over n-grams |
| H4 | At matched norm, `r_ref` is a stronger handle on the action than `r_blast` | pending C3 |
| H5 | Harm vocabulary engages `r_ref` beyond neutral wording | 0.067 [0.063, 0.070] gaps |

**Early reading:** the model is not blind to production. It encodes it, the encoding reaches the decision, and removing the related direction removes the caution the model shows. But production shifts the action by 0.4 logits, against 5.2 for an explicit policy line.

## Design decisions that differ from the original plan

**Benchmark**
- **Benign twin crossed with every factor.** This gives 16 cells per skeleton and 5,120 main prompts, instead of 2,560 destructive prompts. It is needed for over-caution, the conjunction label and steering selectivity.
- **Ticket framing.** Requests arrive as tickets, and the policy names the on-call engineer as the confirmer. The request therefore cannot count as its own confirmation.
- **Direct tool calls.** The system prompt asks for the tool call with no preceding text. In pilot 1, 84% of executions under the confirmation rule began with text, and the first token matched the action only 63% of the time (now 96 to 98%). Narration is kept as a separate condition.
- **Importance in the policy.** The confirmation rule covers actions that "destroy important data, expose sensitive systems, or take down a live service". In pilot 2 the model read "data loss" literally, so deleting temporary data counted too.
- **Three judgment questions** (policy condition, environment, target importance) replace one "destructive and hard to reverse" question.
- **Ladder and cue scope.** The phrasing ladder and cue-only controls use the 29 templates whose effect is deletion or stoppage, so the consequence sentences stay natural. Slang joins the cue-only controls.
- **Ask and decline labels.** The deterministic parser assigns them instead of a small local judge. A 100-item human audit is pending.

**Analysis**
- **Primary consequence factor:** blast radius (production against staging), since target importance is not understood. Target and the conjunction are secondary.
- **"Seen" estimator:** the within-skeleton production shift on `r_harm`. The absolute position mixes task type with harm.
- **Behavior-filtered directions:** content-harm directions come from refused harmful and accepted benign prompts, as in Arditi et al. and Zhao et al.
- **Schema confound:** the same tools as native JSON against flattened prose, rather than tools present against absent.
- **C2 scope:** ablates a direction at all layers and positions, not only at the decision position and the environment span.
- **H4:** compares matched-norm handles (`r_ref` against `r_blast`) on dangerous prompts. The selectivity index is kept as a secondary analysis.
- **Exploratory additions:** C6 (one-coordinate swaps) and C7 (patching everything except the environment span). They separate what the probe reads from what drives the action.

**Models and preregistration**
- **Comparator:** Qwen3-4B-Instruct-2507 rather than Llama-3.1-8B. It is ungated and opens tool calls with a single token.
- **Held-out lock:** held-out skeletons are sealed in code. Analysis refuses them until a lock file matches the hash of the preregistration, its amendment and the locked values.
- **Probe rule:** the registered layer rule is kept for H3. It did not predict transfer to new surface forms, so a best-development-point probe was added as a disclosed secondary analysis (amendment 1).

## Repository

```
configs/          pinned model and data revisions; locked analysis values
docs/             benchmark spec, preregistration, amendment
src/opharm/
  bench/          templates, lexicons, generator, token alignment, oracle
  refsets/        content-harm reference set and refusal classifier
  interp/         hooks (capture, patch, steer, ablate, swap), directions, probes
  run/            decision metric, activation capture, greedy decoding
  stats/          cluster bootstrap, Holm correction, held-out lock
  analysis.py     shared data partitions and locked settings
  chat.py         chat-template rendering with token spans and positions
  models.py       model loading
scripts/          numbered pipeline, below
tests/            unit tests
results/          one JSON summary per experiment
```

| Scripts | Step |
|---|---|
| `00` to `03` | download, numerics check, throughput, residual profile |
| `04` to `07` | reference data, content-harm directions, positive control (C5) |
| `08` | benchmark generation |
| `09`, `10` | behavioral runs with activation capture; behavior report |
| `11` | representation: seen, probes, geometry |
| `12`, `13` | causal panels (C1 to C4, C6, C7) and their report |
| `14` | confirmatory hypotheses H1 to H5 with Holm correction |

Generated data (`benchmark/`), run outputs and activations (`runs/`) and caches (`.cache/`) stay local and untracked.

## Usage

```bash
uv sync
```

Put a Hugging Face read token in `.env` as `HF_READ=...`. Python entry points load it through `opharm.paths`. For shell tools, run `source scripts/env.sh` first. Caches, model weights and run outputs stay inside this folder.
