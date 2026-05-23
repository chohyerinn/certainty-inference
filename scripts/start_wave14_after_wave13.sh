#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
cd "$ROOT" || exit 1
mkdir -p logs

echo "=== wave14 watcher start $(date -Is) ==="

while pgrep -f '[w]ave10_gpu3_runner.sh' >/dev/null || \
      pgrep -f '[w]ave11_gpu3_runner.sh' >/dev/null || \
      pgrep -f '[w]ave12_gpu3_runner.sh' >/dev/null || \
      pgrep -f '[w]ave13_gpu3_runner.sh' >/dev/null; do
  sleep 60
done

if pgrep -f '[w]ave14_gpu3_runner.sh' >/dev/null; then
  echo "=== wave14 already running; watcher done $(date -Is) ==="
  exit 0
fi

echo "=== starting wave14 $(date -Is) ==="
nohup bash scripts/wave14_gpu3_runner.sh > logs/wave14_gpu3_launcher.log 2>&1 < /dev/null &
echo $! > logs/wave14_gpu3.pid
echo "=== wave14 launched pid=$(cat logs/wave14_gpu3.pid) $(date -Is) ==="
