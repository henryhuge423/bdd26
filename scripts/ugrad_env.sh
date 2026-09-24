# Source on ugradx/ugradv before any work:  source /tmp/cgf2604/bdd26/scripts/ugrad_env.sh
# Redirects every cache to local /tmp so nothing lands in the shared 15G NFS home quota.
export UGRAD_ROOT=/tmp/cgf2604
export BDD=$UGRAD_ROOT/bdd26
export PANNUKE_ROOT=$BDD/data/pannuke
export XDG_CACHE_HOME=$UGRAD_ROOT/cache
export UV_CACHE_DIR=$UGRAD_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$UGRAD_ROOT/uv-python UV_PYTHON_BIN_DIR=$UGRAD_ROOT/uv-python/bin UV_TOOL_DIR=$UGRAD_ROOT/uv-tools
export PIP_CACHE_DIR=$UGRAD_ROOT/cache/pip
export HF_HOME=$UGRAD_ROOT/cache/hf
export TORCH_HOME=$UGRAD_ROOT/cache/torch
export WANDB_DIR=$UGRAD_ROOT/cache/wandb WANDB_CACHE_DIR=$UGRAD_ROOT/cache/wandb
export MPLCONFIGDIR=$UGRAD_ROOT/cache/mpl
export TRITON_CACHE_DIR=$UGRAD_ROOT/cache/triton
export CUDA_CACHE_PATH=$UGRAD_ROOT/cache/nv
export TMPDIR=$UGRAD_ROOT/tmp
# ugrad enforces a HARD 16 GB address-space limit per process (limits.conf: @users hard as).
# CUDA+torch already reserve ~9 GB of it, so trim glibc arenas / thread pools to leave ~6.9 GB
# of usable GPU memory per process. Heavy training belongs on the A100 machines.
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_MODULE_LOADING=LAZY
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
[ -f "$UGRAD_ROOT/hg-token.txt" ] && export HF_TOKEN="$(tr -d '\n ' < "$UGRAD_ROOT/hg-token.txt")"
[ -d "$UGRAD_ROOT/venv" ] && source "$UGRAD_ROOT/venv/bin/activate"
cd "$BDD" 2>/dev/null || true
