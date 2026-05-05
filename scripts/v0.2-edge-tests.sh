#!/usr/bin/env bash
# Engram v0.2 W19 Day3 边界测试（22 条）
# Usage: BASE=http://localhost:18080 AUTH="siri-day3-token" bash v0.2-edge-tests.sh
# 输出：每条用例 PASS/FAIL + 实际 status + body
# 维护人：Siri，2026-05-05 14:00 prep

set -u
BASE="${BASE:-http://localhost:18080}"
AUTH="${AUTH:-}"
HDR=(-H "Content-Type: application/json")
[[ -n "$AUTH" ]] && HDR+=(-H "Authorization: Bearer $AUTH")
HDR+=(-H "X-Caller-Type: agent-self")

PASS=0; FAIL=0; TOTAL=0
mkdir -p /tmp/engram-day3-out

run() {
  local id="$1" expect_code="$2" method="$3" path="$4" body="${5:-}"
  TOTAL=$((TOTAL+1))
  local out="/tmp/engram-day3-out/${id}.json"
  local code
  if [[ -n "$body" ]]; then
    code=$(curl -s -o "$out" -w "%{http_code}" -X "$method" "${HDR[@]}" -d "$body" "${BASE}${path}")
  else
    code=$(curl -s -o "$out" -w "%{http_code}" -X "$method" "${HDR[@]}" "${BASE}${path}")
  fi
  if [[ "$code" == "$expect_code" ]]; then
    PASS=$((PASS+1))
    printf "✅ %-10s %3s  %s %s\n" "$id" "$code" "$method" "$path"
  else
    FAIL=$((FAIL+1))
    printf "❌ %-10s expect=%s got=%s  %s %s | body=%s\n" "$id" "$expect_code" "$code" "$method" "$path" "$(head -c 200 "$out")"
  fi
}

# ---- 准备测试数据：建 4 条 memory，分别处于 active / deprecated / archived / 用于 PATCH 测试 ----
echo "==== 准备测试 fixture ===="
ACTIVE_ID=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-fixture-active","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
DEPRECATED_ID=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-fixture-deprecated","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
ARCHIVED_ID=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-fixture-archived","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
PATCH_ID=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-fixture-patch-original","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
echo "ACTIVE_ID=$ACTIVE_ID DEPRECATED_ID=$DEPRECATED_ID ARCHIVED_ID=$ARCHIVED_ID PATCH_ID=$PATCH_ID"
# 把 DEPRECATED_ID 推到 deprecated；ARCHIVED_ID 推到 archived（active→deprecated→archived）
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${DEPRECATED_ID}" >/dev/null
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${ARCHIVED_ID}" >/dev/null
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"archived"}'   "${BASE}/memories/${ARCHIVED_ID}" >/dev/null

# ==== 1. FSM 跨态跳转（7 条） ====
echo "==== FSM ===="
run FSM-01 409 PATCH "/memories/${ARCHIVED_ID}"   '{"lifecycle_status":"active"}'
run FSM-02 409 PATCH "/memories/${ARCHIVED_ID}"   '{"lifecycle_status":"deprecated"}'
run FSM-03 409 PATCH "/memories/${DEPRECATED_ID}" '{"lifecycle_status":"active"}'
run FSM-04 200 PATCH "/memories/${ACTIVE_ID}"     '{"lifecycle_status":"deprecated"}'
run FSM-05 200 PATCH "/memories/${ACTIVE_ID}"     '{"lifecycle_status":"archived"}'
run FSM-06 200 POST  "/memories/${ARCHIVED_ID}/reset" ''
run FSM-07 200 PATCH "/memories/${DEPRECATED_ID}" '{"lifecycle_status":"deprecated"}'   # no-op

# ==== 2. PATCH content 字段拒绝（5 条） ====
echo "==== PATCH ===="
run PATCH-01 400 PATCH "/memories/${PATCH_ID}" '{"content":"hijacked"}'
run PATCH-02 400 PATCH "/memories/${PATCH_ID}" '{"content":"hijacked","tags":["x"]}'
run PATCH-03 200 PATCH "/memories/${PATCH_ID}" '{"tags":["day3-fixture","new-tag"]}'
run PATCH-04 200 PATCH "/memories/${PATCH_ID}" '{"lifecycle_status":"deprecated"}'   # active→deprecated 合法
run PATCH-05 200 PUT   "/memories/${PATCH_ID}" '{"type":"event","content":"day3-fixture-patch-replaced","importance":3,"tags":["day3-fixture"]}'

# 验证 PATCH-01 失败后 content 没污染（验收时人工校验 GET /memories/${PATCH_ID} 的 content）
curl -s "${HDR[@]}" "${BASE}/memories/${PATCH_ID}" > /tmp/engram-day3-out/PATCH-aftercheck.json

# ==== 3. 软删 + search（5 条） ====
echo "==== SEARCH ===="
# 先建一条 archived 专测 search 过滤
SEARCH_ARCHIVED=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-search-needle-archived-XK7Q","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
curl -s "${HDR[@]}" -X DELETE "${BASE}/memories/${SEARCH_ARCHIVED}" >/dev/null   # 软删 → archived
SEARCH_DEPRECATED=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-search-needle-deprecated-XK7Q","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${SEARCH_DEPRECATED}" >/dev/null

# SEARCH-01: archived 默认不召回
res=$(curl -s "${HDR[@]}" -X POST -d '{"query":"day3-search-needle XK7Q","limit":10}' "${BASE}/memories/search")
if ! echo "$res" | grep -q "$SEARCH_ARCHIVED"; then
  PASS=$((PASS+1)); printf "✅ %-10s archived 已被过滤\n" "SEARCH-01"
else
  FAIL=$((FAIL+1)); printf "❌ %-10s archived 仍被召回\n" "SEARCH-01"
fi
TOTAL=$((TOTAL+1))

# SEARCH-02: include_archived=true 召回
res=$(curl -s "${HDR[@]}" -X POST -d "{\"query\":\"day3-search-needle XK7Q\",\"limit\":10,\"include_archived\":true}" "${BASE}/memories/search")
if echo "$res" | grep -q "$SEARCH_ARCHIVED"; then
  PASS=$((PASS+1)); printf "✅ %-10s include_archived=true 召回 archived\n" "SEARCH-02"
else
  FAIL=$((FAIL+1)); printf "❌ %-10s include_archived=true 未召回 archived | body=%s\n" "SEARCH-02" "$(echo "$res" | head -c 200)"
fi
TOTAL=$((TOTAL+1))

run SEARCH-03 200 GET "/memories/${SEARCH_ARCHIVED}"   # 物理保留

# SEARCH-04: deprecated 默认仍召回（Q1 锁定）
res=$(curl -s "${HDR[@]}" -X POST -d '{"query":"day3-search-needle XK7Q","limit":10}' "${BASE}/memories/search")
if echo "$res" | grep -q "$SEARCH_DEPRECATED"; then
  PASS=$((PASS+1)); printf "✅ %-10s deprecated 默认召回\n" "SEARCH-04"
else
  FAIL=$((FAIL+1)); printf "❌ %-10s deprecated 未默认召回 | body=%s\n" "SEARCH-04" "$(echo "$res" | head -c 200)"
fi
TOTAL=$((TOTAL+1))

# SEARCH-05: 性能不退化（archived 大量场景下 latency p95<200ms）—— Day3 review 现场跑，本脚本只校验 200
res_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${HDR[@]}" -d '{"query":"day3-search-needle","limit":50}' "${BASE}/memories/search")
if [[ "$res_code" == "200" ]]; then
  PASS=$((PASS+1)); printf "✅ %-10s search bulk 200（latency 走 perf 测）\n" "SEARCH-05"
else
  FAIL=$((FAIL+1)); printf "❌ %-10s search bulk got=%s\n" "SEARCH-05" "$res_code"
fi
TOTAL=$((TOTAL+1))

# ==== 4. RESET（5 条） ====
echo "==== RESET ===="
# 重新准备
RST_DEPR=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-reset-fixture-deprecated","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${RST_DEPR}" >/dev/null
RST_ARCH=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-reset-fixture-archived","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${RST_ARCH}" >/dev/null
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"archived"}'   "${BASE}/memories/${RST_ARCH}" >/dev/null
RST_ACTIVE=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-reset-fixture-active","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')

run RESET-01 200 POST "/memories/${RST_ARCH}/reset" ''   # archived → active
run RESET-02 409 POST "/memories/${RST_ACTIVE}/reset" '' # active → reset 拒绝（Day1 review 倾向 not_resettable）
run RESET-03 200 POST "/memories/${RST_DEPR}/reset" ''   # deprecated → active（Day1 锁定为允许）

# RESET-04: reset 后 search 默认召回
res=$(curl -s "${HDR[@]}" -X POST -d '{"query":"day3-reset-fixture-archived","limit":10}' "${BASE}/memories/search")
if echo "$res" | grep -q "$RST_ARCH"; then
  PASS=$((PASS+1)); printf "✅ %-10s reset 后默认 search 召回\n" "RESET-04"
else
  FAIL=$((FAIL+1)); printf "❌ %-10s reset 后默认 search 未召回\n" "RESET-04"
fi
TOTAL=$((TOTAL+1))

# RESET-05: agent-self 调用 reset 不限（Day1 锁定）
RST_X=$(curl -s "${HDR[@]}" -X POST -d '{"type":"event","content":"day3-reset-fixture-x","importance":3,"tags":["day3-fixture"]}' "${BASE}/memories" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"deprecated"}' "${BASE}/memories/${RST_X}" >/dev/null
curl -s "${HDR[@]}" -X PATCH -d '{"lifecycle_status":"archived"}'   "${BASE}/memories/${RST_X}" >/dev/null
run RESET-05 200 POST "/memories/${RST_X}/reset" ''

echo
echo "==== 总计：$TOTAL  PASS=$PASS  FAIL=$FAIL ===="
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
