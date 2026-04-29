#!/usr/bin/env bash
# pi-sync.sh — fast-forward the Pi worktree to origin/main only.
# No scp, no force-push, no manual file copies. If the working tree is
# dirty or has diverged, this script aborts so the operator can resolve.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"

cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "pi-sync: $REPO_DIR is not a git work tree" >&2
    exit 2
fi

dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "pi-sync: working tree is dirty; refusing to sync" >&2
    echo "$dirty" >&2
    exit 3
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "$BRANCH" ]; then
    echo "pi-sync: on branch '$current_branch'; expected '$BRANCH'" >&2
    exit 4
fi

git fetch --prune "$REMOTE"
git pull --ff-only "$REMOTE" "$BRANCH"

head=$(git rev-parse HEAD)
echo "pi-sync: $BRANCH at $head"
