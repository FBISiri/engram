#!/usr/bin/env bash
#
# engram-daily-report.sh — dual-channel observability daily report.
#
# PRIMARY channel : Prometheus text exposition at http://127.0.0.1:8080/metrics
# FALLBACK channel: today's trace file (Go stdouttrace FLAT jsonl), used ONLY for
#                   tail-latency attribution of slow add spans.
#
# Alarms:
#   A1. any single add or embed > 10s
#   A2. embedding cache hit rate < 20%
#   A3. share of searches with top_score < 0.70 > 35%
#
# Exit: 0 = no alarm; 1 = >=1 alarm fired; 2 = could not collect data.

set -u

METRICS_URL="http://127.0.0.1:8080/metrics"
TRACE_DIR="/tmp/siri-state/engram-traces"
TODAY="$(date +%Y-%m-%d)"
TRACE_FILE="${TRACE_DIR}/engram-traces-${TODAY}.jsonl"

SCRAPE="$(curl -s --max-time 10 "$METRICS_URL")"
if [ $? -ne 0 ] || [ -z "$SCRAPE" ]; then
    echo "ERROR: could not scrape ${METRICS_URL} (unreachable or empty)." >&2
    exit 2
fi

# Everything else (parse, assert structure, evaluate alarms) is done in python3.
# The python program itself is fed via the heredoc (python3 -), so the metrics
# blob and paths are passed via argv (a temp file for the scrape).
METRICS_FILE="$(mktemp)"
trap 'rm -f "$METRICS_FILE"' EXIT
printf '%s' "$SCRAPE" > "$METRICS_FILE"

python3 - "$METRICS_FILE" "$TRACE_FILE" "$TODAY" <<'PYEOF'
import sys, re, json, time

metrics = open(sys.argv[1]).read()
trace_file = sys.argv[2]
today = sys.argv[3]

def fail_scrape(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(2)

# ---- parse the text exposition format --------------------------------------
# sample:  name{label="v",...} 123   OR   name 123
line_re = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+\-]+)\s*$')
def parse_labels(s):
    d = {}
    if not s:
        return d
    for k, v in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"', s):
        d[k] = v
    return d

samples = []  # (name, labels_dict, float_value)
for ln in metrics.splitlines():
    ln = ln.strip()
    if not ln or ln.startswith('#'):
        continue
    m = line_re.match(ln)
    if not m:
        continue
    try:
        val = float(m.group(3))
    except ValueError:
        continue
    samples.append((m.group(1), parse_labels(m.group(2)[1:-1] if m.group(2) else ""), val))

def find(name, want=None):
    out = []
    for n, lab, v in samples:
        if n != name:
            continue
        if want:
            if all(lab.get(k) == vv for k, vv in want.items()):
                out.append((lab, v))
        else:
            out.append((lab, v))
    return out

# ---- assert the positive structure we expect (R10) -------------------------
required = [
    "engram_embed_duration_seconds_bucket",
    "engram_embed_cache_hit_total",
    "engram_embed_cache_miss_total",
    "engram_search_top_score_bucket",
]
present = {n for n, _, _ in samples}
missing = [r for r in required if r not in present]
if missing:
    fail_scrape("scrape missing expected metric families: %s" % ", ".join(missing))

# ---- A1 (embed side): embed samples > 10s ----------------------------------
b10 = find("engram_embed_duration_seconds_bucket", {"le": "10"})
binf = find("engram_embed_duration_seconds_bucket", {"le": "+Inf"})
if not b10 or not binf:
    fail_scrape("embed duration histogram missing le=10 or le=+Inf buckets")
embed_gt10 = binf[0][1] - b10[0][1]

# ---- A2: cache hit rate ----------------------------------------------------
hit = find("engram_embed_cache_hit_total")[0][1]
miss = find("engram_embed_cache_miss_total")[0][1]
denom = hit + miss
if denom <= 0:
    fail_scrape("embed cache counters are both zero (no data to evaluate)")
hit_rate = hit / denom

# ---- A3: share of searches with top_score < 0.70 ---------------------------
# The le label nearest 0.70 is a float-rounding artifact (e.g. 0.7000000000000002).
# Discover it per-collection instead of hardcoding a naive "0.70".
tot_lt070 = 0.0
tot_count = 0.0
for lab, cnt in find("engram_search_top_score_count"):
    coll = lab.get("collection")
    tot_count += cnt
    # buckets for this collection
    les = []
    for l, v in find("engram_search_top_score_bucket"):
        if l.get("collection") == coll and l.get("le") != "+Inf":
            try:
                les.append((float(l["le"]), v))
            except (ValueError, KeyError):
                pass
    if not les:
        continue
    # bucket boundary nearest 0.70
    near = min(les, key=lambda t: abs(t[0] - 0.70))
    tot_lt070 += near[1]
if tot_count <= 0:
    fail_scrape("search top_score histogram has zero total count")
share_lt070 = tot_lt070 / tot_count

# ---- A1 (add side): scan trace file for add spans > 10s --------------------
# Manual RFC3339 nanosecond parse (fromisoformat unavailable for these on host).
def parse_rfc3339(s):
    # e.g. 2026-09-14T02:08:32.354582956+08:00
    m = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2}|Z)$', s)
    if not m:
        raise ValueError("bad ts: %r" % s)
    base = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
    frac = float(m.group(2)) if m.group(2) else 0.0
    off = m.group(3)
    if off == "Z":
        off_sec = 0.0
    else:
        sign = 1 if off[0] == '+' else -1
        oh, om = off[1:].split(":")
        off_sec = sign * (int(oh) * 3600 + int(om) * 60)
    # mktime treats input as local time; we only need differences of same-tz
    # timestamps, so the local-tz bias and the offset both cancel in EndTime-StartTime.
    return base + frac - off_sec

def attr_map(attrs):
    d = {}
    for a in attrs or []:
        d[a.get("Key")] = a.get("Value", {}).get("Value")
    return d

trace_exists = False
add_gt10 = 0
slowest = None  # (dur, attrs)
try:
    with open(trace_file) as f:
        trace_exists = True
        for ln in f:
            ln = ln.strip()
            if not ln or '"engram.memory.add"' not in ln:
                continue
            try:
                span = json.loads(ln)
            except ValueError:
                continue
            if span.get("Name") != "engram.memory.add":
                continue
            try:
                dur = parse_rfc3339(span["EndTime"]) - parse_rfc3339(span["StartTime"])
            except (ValueError, KeyError):
                continue
            if dur > 10.0:
                add_gt10 += 1
                if slowest is None or dur > slowest[0]:
                    slowest = (dur, attr_map(span.get("Attributes")))
except FileNotFoundError:
    trace_exists = False

# ---- evaluate alarms -------------------------------------------------------
alarms = []
if embed_gt10 > 0 or add_gt10 > 0:
    parts = []
    if embed_gt10 > 0:
        parts.append("%d embed sample(s) >10s" % int(embed_gt10))
    if add_gt10 > 0:
        parts.append("%d add span(s) >10s" % add_gt10)
    alarms.append("A1 latency: " + "; ".join(parts))
if hit_rate < 0.20:
    alarms.append("A2 cache hit rate %.1f%% < 20%%" % (hit_rate * 100))
if share_lt070 > 0.35:
    alarms.append("A3 low-score search share %.1f%% > 35%%" % (share_lt070 * 100))

# ---- report ----------------------------------------------------------------
print("=== engram daily report (%s) ===" % today)
print("embed samples >10s      : %d" % int(embed_gt10))
print("add spans >10s (trace)  : %d" % add_gt10)
if slowest:
    a = slowest[1]
    print("  slowest add: %.2fs type=%s importance=%s content.length=%s" % (
        slowest[0], a.get("type"), a.get("importance"), a.get("content.length")))
print("cache hit rate          : %.1f%% (hit=%d miss=%d)" % (hit_rate * 100, int(hit), int(miss)))
print("searches top_score<0.70 : %.1f%% (%d/%d)" % (share_lt070 * 100, int(tot_lt070), int(tot_count)))
print("today's trace file      : %s (%s)" % ("present" if trace_exists else "MISSING", trace_file))
print("---")
if alarms:
    for a in alarms:
        print("ALARM: " + a)
    sys.exit(1)
print("no alarm")
sys.exit(0)
PYEOF
rc=$?
exit $rc
