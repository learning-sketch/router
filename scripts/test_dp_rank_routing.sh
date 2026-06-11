#!/usr/bin/env bash
#
# test_dp_rank_routing.sh
#
# Observe which vLLM data-parallel (DP) rank the router sends requests to.
#
# The router exposes per-worker Prometheus counters whose `worker` label is the
# DP-aware URL (e.g. http://host:8000@0, http://host:8000@1). By snapshotting
# `vllm_router_processed_requests_total` before/after a burst of requests we can
# see exactly how many requests landed on each DP rank.
#
# It runs two scenarios:
#   1. SAME key      -> with consistent_hash, all requests pin to ONE rank.
#   2. DISTINCT keys -> a different X-Session-ID per request spreads requests
#                       across ranks.
#
# Usage:
#   ./scripts/test_dp_rank_routing.sh
#
# Configure via environment variables (defaults match a local single-node run):
#   ROUTER       Router base URL              (default: http://127.0.0.1:30000)
#   METRICS      Prometheus metrics base URL  (default: http://127.0.0.1:29000)
#   MODEL        Model name; auto-detected from /v1/models if empty
#   ENDPOINT     Inference endpoint           (default: /v1/completions)
#   PROMPT       Prompt text                  (default: "Once upon a time, there was a")
#   N            Requests per scenario        (default: 20)
#   MAX_TOKENS   max_tokens per request       (default: 1)
#   API_KEY      Bearer token, if router auth is enabled (optional)

set -Eeuo pipefail

ROUTER="${ROUTER:-http://127.0.0.1:30000}"
METRICS="${METRICS:-http://127.0.0.1:29000}"
ENDPOINT="${ENDPOINT:-/v1/completions}"
PROMPT="${PROMPT:-Once upon a time, there was a}"
N="${N:-20}"
MAX_TOKENS="${MAX_TOKENS:-1}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-}"

AUTH_ARGS=()
if [[ -n "$API_KEY" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")
fi

log()  { printf '\033[1;34m[*]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[!]\033[0m %s\n' "$*" >&2; }

require() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }
}
require curl
require awk

# ---------------------------------------------------------------------------
# Preflight: router + metrics reachable
# ---------------------------------------------------------------------------
if ! curl -fsS "${AUTH_ARGS[@]}" "${ROUTER}/health" >/dev/null 2>&1; then
  err "Router not reachable at ${ROUTER} (tried ${ROUTER}/health)."
  err "Set ROUTER=... to point at your router."
  exit 1
fi
if ! curl -fsS "${METRICS}/metrics" >/dev/null 2>&1; then
  err "Metrics endpoint not reachable at ${METRICS}/metrics."
  err "Set METRICS=... (default port is 29000)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect model name if not provided
# ---------------------------------------------------------------------------
if [[ -z "$MODEL" ]]; then
  MODEL="$(curl -fsS "${AUTH_ARGS[@]}" "${ROUTER}/v1/models" 2>/dev/null \
    | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
    | head -n1 | sed -E 's/.*"id"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/' || true)"
  if [[ -z "$MODEL" ]]; then
    err "Could not auto-detect model from ${ROUTER}/v1/models. Set MODEL=... explicitly."
    exit 1
  fi
fi
log "Router   : ${ROUTER}"
log "Metrics  : ${METRICS}"
log "Endpoint : ${ENDPOINT}"
log "Model    : ${MODEL}"
log "Requests : ${N} per scenario"
echo

# ---------------------------------------------------------------------------
# Snapshot per-rank counters.
# Prints lines: "<worker_url> <count>" for vllm_router_processed_requests_total.
# ---------------------------------------------------------------------------
snapshot() {
  curl -fsS "${METRICS}/metrics" \
    | awk '
      /^vllm_router_processed_requests_total\{/ {
        line=$0
        # extract worker="..."
        if (match(line, /worker="[^"]*"/)) {
          w=substr(line, RSTART+8, RLENGTH-9)
          # value is the last whitespace-separated field
          n=split(line, a, /[[:space:]]+/)
          gsub(/[^0-9.eE+-]/, "", a[n])
          print w" "a[n]
        }
      }'
}

# Send one inference request. $1 = optional X-Session-ID value.
send_one() {
  local sid="$1"
  local hdr=()
  [[ -n "$sid" ]] && hdr=(-H "X-Session-ID: ${sid}")
  curl -fsS -o /dev/null "${AUTH_ARGS[@]}" "${hdr[@]}" \
    -H "Content-Type: application/json" \
    -X POST "${ROUTER}${ENDPOINT}" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"stream\":false}" \
    || err "request failed (sid='${sid}')"
}

# Print the delta between two snapshots as a per-rank distribution table.
# $1 = before snapshot text, $2 = after snapshot text, $3 = scenario label
report_delta() {
  local before="$1" after="$2" label="$3"
  echo
  ok "Distribution for: ${label}"
  printf '    %-45s %s\n' "WORKER (DP-aware URL)" "REQUESTS"
  printf '    %-45s %s\n' "---------------------------------------------" "--------"
  # Join before/after by worker and print positive deltas.
  awk -v before="$before" -v after="$after" '
    BEGIN {
      n=split(before, bl, "\n")
      for (i=1;i<=n;i++){ if(bl[i]=="")continue; split(bl[i],kv," "); b[kv[1]]=kv[2] }
      m=split(after, al, "\n")
      for (i=1;i<=m;i++){ if(al[i]=="")continue; split(al[i],kv," "); a[kv[1]]=kv[2] }
      total=0
      for (w in a){ d=a[w]-(w in b?b[w]:0); if(d>0){ printf "    %-45s %d\n", w, d; total+=d } }
      if (total==0) print "    (no change detected — check metrics scraping / labels)"
    }'
}

# ---------------------------------------------------------------------------
# Scenario 1: same key (no session header) -> all requests pin to one rank
# ---------------------------------------------------------------------------
log "Scenario 1: SAME key (no X-Session-ID, identical prompt)"
BEFORE="$(snapshot)"
for ((i=0; i<N; i++)); do send_one ""; done
sleep 1
AFTER="$(snapshot)"
report_delta "$BEFORE" "$AFTER" "SAME key  (expect: concentrated on ONE @rank with consistent_hash)"

# ---------------------------------------------------------------------------
# Scenario 2: distinct keys -> requests spread across ranks
# ---------------------------------------------------------------------------
echo
log "Scenario 2: DISTINCT keys (unique X-Session-ID per request)"
BEFORE="$(snapshot)"
for ((i=0; i<N; i++)); do send_one "sess-$(date +%s%N)-${i}"; done
sleep 1
AFTER="$(snapshot)"
report_delta "$BEFORE" "$AFTER" "DISTINCT keys  (expect: spread across @0, @1, ...)"

echo
ok "Done."
echo
log "Tip: you can also watch the live per-rank counters directly:"
echo "    curl -s ${METRICS}/metrics | grep -E 'vllm_router_(processed_requests|policy_decisions)_total'"
log "Or current load per DP-aware worker:"
echo "    curl -s ${ROUTER}/get_loads"
