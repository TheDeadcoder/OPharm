if [ -n "${BASH_VERSION:-}" ]; then _f="${BASH_SOURCE[0]}"; else _f="$0"; fi
_root="$(cd "$(dirname "$_f")/.." && pwd)"
export HF_HOME="$_root/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$_root/.cache/torch"
export MPLCONFIGDIR="$_root/.cache/matplotlib"
export XDG_CACHE_HOME="$_root/.cache"
export UV_CACHE_DIR="$_root/.cache/uv"
export HF_HUB_DISABLE_TELEMETRY=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$_root/.env" ]; then
  HF_TOKEN="$(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?HF_READ=//p' "$_root/.env" | tail -n 1 | tr -d "\"'")"
  export HF_TOKEN
fi
unset _f _root
