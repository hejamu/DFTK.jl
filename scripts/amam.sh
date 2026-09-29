#!/usr/bin/env bash
# Sync the project to AMAM, run a Julia command through Slurm, sync results back.
# Usage: scripts/amam.sh [-c CPUS] [-m MEM] [-t TIME] -- <julia args...>
# Example: scripts/amam.sh -c 8 -m 16G -- test/runtests.jl
# Runs on the Workstations partition, excluding narvi (ws05, Henrik's box).
set -euo pipefail
CPUS=8; MEM=16G; TIME=02:00:00
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CPUS=$2; shift 2;; -m) MEM=$2; shift 2;; -t) TIME=$2; shift 2;;
    --) shift; break;; *) break;;
  esac
done
HOST=tarvos
REMOTE=WORK/playground/naBEC
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
rsync -a --exclude .git --exclude results "$LOCAL/" "$HOST:$REMOTE/"
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE/results && cd $REMOTE && \
  srun --partition=Workstations --exclude=ws05 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
  --job-name=nabec ~/.juliaup/bin/julia --project=. -t $CPUS $*"
rsync -a "$HOST:$REMOTE/results/" "$LOCAL/results/"
