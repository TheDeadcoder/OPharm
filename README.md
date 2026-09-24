# opharm

Operational harm as a blind spot of content-harm safety in small LLM agents.

## Setup

```bash
uv sync
```

Put a Hugging Face read token in `.env` as `HF_READ=...`. Python entry points pick it up through `opharm.paths`; for shell tools (`hf`, `mlx_lm`), run `source scripts/env.sh` first.

Caches, model weights and run outputs stay inside this folder (`.cache/`, `runs/`), which git ignores.
