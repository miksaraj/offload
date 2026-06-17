#!/usr/bin/env bash
# Smoke-tests the offload CLI binary: build it, then exercise every
# subcommand with representative args and check exit codes + output.
#
# Run from the repo root: .claude/skills/run-offload/smoke.sh
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

BIN=target/debug/offload
PASS=0
FAIL=0

check() {
  local desc="$1" expected_exit="$2"
  shift 2
  local out
  out="$("$@" 2>&1)"
  local actual_exit=$?
  if [[ "$actual_exit" -eq "$expected_exit" ]]; then
    echo "ok   - $desc (exit $actual_exit)"
    PASS=$((PASS + 1))
  else
    echo "FAIL - $desc (expected exit $expected_exit, got $actual_exit)"
    echo "       output: $out"
    FAIL=$((FAIL + 1))
  fi
}

contains() {
  local desc="$1" needle="$2"
  shift 2
  local out
  out="$("$@" 2>&1)"
  if [[ "$out" == *"$needle"* ]]; then
    echo "ok   - $desc (output contains '$needle')"
    PASS=$((PASS + 1))
  else
    echo "FAIL - $desc (output missing '$needle')"
    echo "       output: $out"
    FAIL=$((FAIL + 1))
  fi
}

echo "== Building =="
cargo build --workspace --quiet || { echo "FAIL - cargo build"; exit 1; }
echo "ok   - cargo build --workspace"
PASS=$((PASS + 1))

echo
echo "== CLI smoke tests =="
contains "--help lists all subcommands"   "run"        "$BIN" --help
contains "--help lists inspect"           "inspect"    "$BIN" --help
contains "--help lists cache"             "cache"      "$BIN" --help
contains "--help lists models"            "models"     "$BIN" --help
check    "--version exits 0"               0            "$BIN" --version
check    "run with missing input exits 1"  1            "$BIN" run --input /tmp/offload-smoke-does-not-exist.mp4
contains "run with missing input reports it" "input video not found" "$BIN" run --input /tmp/offload-smoke-does-not-exist.mp4
check    "inspect exits 0 (stub)"          0            "$BIN" inspect --input /tmp/offload-smoke-does-not-exist.mp4
check    "cache --clear exits 0 (stub)"    0            "$BIN" cache --clear
check    "models --download exits 0 (stub)" 0           "$BIN" models --download
check    "run with no --input fails to parse" 2          "$BIN" run

echo
echo "== Unit/integration tests =="
if cargo test --workspace --quiet 2>&1 | tail -20; then
  echo "ok   - cargo test --workspace"
  PASS=$((PASS + 1))
else
  echo "FAIL - cargo test --workspace"
  FAIL=$((FAIL + 1))
fi

echo
echo "== Summary: $PASS passed, $FAIL failed =="
[[ "$FAIL" -eq 0 ]]
