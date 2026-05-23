#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
cd "$ROOT" || exit 1
mkdir -p logs

echo "=== wave11/wave12 watcher start $(date -Is) ==="

while pgrep -f '[w]ave10_gpu3_runner.sh' >/dev/null; do
  sleep 60
done

if pgrep -f '[w]ave11_gpu3_runner.sh' >/dev/null; then
  echo "=== wave11 already running; waiting $(date -Is) ==="
  while pgrep -f '[w]ave11_gpu3_runner.sh' >/dev/null; do
    sleep 60
  done
else
  echo "=== starting wave11 $(date -Is) ==="
  bash scripts/wave11_gpu3_runner.sh
  echo "=== wave11 returned status=$? $(date -Is) ==="
fi

if pgrep -f '[w]ave12_gpu3_runner.sh' >/dev/null; then
  echo "=== wave12 already running; watcher done $(date -Is) ==="
  exit 0
fi

echo "=== starting wave12 $(date -Is) ==="
nohup bash scripts/wave12_gpu3_runner.sh > logs/wave12_gpu3_launcher.log 2>&1 < /dev/null &
echo $! > logs/wave12_gpu3.pid
echo "=== wave12 launched pid=$(cat logs/wave12_gpu3.pid) $(date -Is) ==="
