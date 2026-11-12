"""
Exchange segmentation — post-processes a stats CSV produced by main.py to
identify active exchange windows vs resets/circling.

Usage
-----
    python segment.py --stats output/myfight_stats.csv
    python segment.py --stats output/myfight_stats.csv --output output/myfight_exchanges.csv
    python segment.py --stats output/myfight_stats.csv --threshold 20 --min-duration 0.3

Output
------
A CSV with one row per exchange:
    exchange_id, start_frame, start_time_s, end_frame, end_time_s, duration_s, peak_motion
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Default tuning parameters
# ---------------------------------------------------------------------------

# Rolling window for smoothing the activity signal
_SMOOTH_WINDOW_S = 0.5    # seconds

# Auto-threshold: use this percentile of the smoothed signal.
# Higher = fewer but more confident exchanges.
_THRESHOLD_PCTILE = 70

# Clip raw jitter at this value before summing, so tracking-error spikes
# (skeleton jumping 300px in one frame) don't inflate the signal.
_JITTER_CLIP = 80.0       # pixels

# Post-processing
_MIN_DURATION_S = 0.25    # discard bursts shorter than this
_MERGE_GAP_S    = 0.5     # merge consecutive bursts separated by less than this

# Proximity weighting
# When fighters are far apart, gross motion is circling/approach — discount it.
# Proximity is measured from the video distribution:
#   "close" = 25th percentile of observed distance for that signal
#   "far"   = 75th percentile
# At "close" the motion gets full weight; at "far" it gets MIN_PROX_WEIGHT.
_MIN_PROX_WEIGHT = 0.25   # minimum multiplier even when fighters are far apart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_stats(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def _fps_from_rows(rows):
    valid = [(r["frame"], r["time_s"])
             for r in rows if r.get("time_s") and r["time_s"] > 0]
    if len(valid) < 2:
        return 24.0
    (f0, t0), (f1, t1) = valid[0], valid[-1]
    return (f1 - f0) / (t1 - t0) if t1 > t0 else 24.0


def _prox_weight(val, near, far):
    """Linear weight: 1.0 at near, _MIN_PROX_WEIGHT at far, clipped outside."""
    if near is None or far is None or far <= near:
        return 0.5
    t = max(0.0, min(1.0, (val - near) / (far - near)))
    return 1.0 - (1.0 - _MIN_PROX_WEIGHT) * t


def _dist_percentiles(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) not in (None, "")]
    if not vals:
        return None, None
    return _percentile(vals, 25), _percentile(vals, 75)


def _activity_signal(rows, fps):
    """
    Per-frame activity = motion weighted by fighter proximity.

    Motion alone (jitter) picks up footwork at any range.  Weighting by
    proximity means two fighters circling at distance barely registers, while
    the same motion at close range — or with swords near each other — scores
    much higher.

    Proximity hierarchy (most specific available wins per frame):
      sword_dist  — swords crossing = 0 px = definite bind
      wrist_dist  — hands close = likely engaged
      fighter_dist — bodies close = at least in measure
    """
    # Percentile reference distances from the whole video
    sw_near, sw_far = _dist_percentiles(rows, "sword_dist")
    wr_near, wr_far = _dist_percentiles(rows, "wrist_dist")
    fg_near, fg_far = _dist_percentiles(rows, "fighter_dist")

    raw = []
    for r in rows:
        # Motion
        a = min(r.get("a_jitter") or 0.0, _JITTER_CLIP)
        b = min(r.get("b_jitter") or 0.0, _JITTER_CLIP)
        motion = a + b

        # Proximity weight — use most specific available signal
        sd = r.get("sword_dist")
        wd = r.get("wrist_dist")
        fd = r.get("fighter_dist")

        if sd not in (None, ""):
            w = _prox_weight(float(sd), sw_near, sw_far)
            # Crossing swords (sd == 0) always gets full weight regardless
            if float(sd) == 0.0:
                w = 1.0
        elif wd not in (None, ""):
            w = _prox_weight(float(wd), wr_near, wr_far)
        elif fd not in (None, ""):
            w = _prox_weight(float(fd), fg_near, fg_far)
        else:
            w = 0.5   # no spatial data for this frame

        raw.append(motion * w)

    # Smooth with centred rolling mean
    window = max(1, int(round(_SMOOTH_WINDOW_S * fps)))
    half   = window // 2
    n      = len(raw)
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))

    return smoothed


def _percentile(values, p):
    """Percentile without numpy."""
    s = sorted(v for v in values if v is not None)
    if not s:
        return 0.0
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _find_exchanges(rows, signal, fps, threshold, min_duration_s, merge_gap_s):
    min_frames   = max(1, int(round(min_duration_s * fps)))
    merge_frames = max(1, int(round(merge_gap_s    * fps)))

    # Binary active mask
    active = [s >= threshold for s in signal]

    # Collect contiguous active runs (as row indices)
    runs, in_run, start = [], False, 0
    for i, a in enumerate(active):
        if a and not in_run:
            start, in_run = i, True
        elif not a and in_run:
            runs.append([start, i - 1])
            in_run = False
    if in_run:
        runs.append([start, len(active) - 1])

    # Merge close runs
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= merge_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # Filter by minimum length, compute output fields
    exchanges = []
    for si, ei in merged:
        if (ei - si + 1) < min_frames:
            continue
        exchanges.append({
            "exchange_id":  len(exchanges) + 1,
            "start_frame":  int(rows[si]["frame"]),
            "start_time_s": round(rows[si]["time_s"], 3),
            "end_frame":    int(rows[ei]["frame"]),
            "end_time_s":   round(rows[ei]["time_s"], 3),
            "duration_s":   round(rows[ei]["time_s"] - rows[si]["time_s"], 3),
            "peak_motion":  round(max(signal[si:ei + 1]), 2),
        })

    return exchanges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def segment(stats_path, output_path=None, threshold=None,
            min_duration_s=_MIN_DURATION_S, merge_gap_s=_MERGE_GAP_S):
    rows = _load_stats(stats_path)
    if not rows:
        print("No data in stats file.")
        return []

    fps    = _fps_from_rows(rows)
    signal = _activity_signal(rows, fps)

    if threshold is None:
        threshold = _percentile(signal, _THRESHOLD_PCTILE)

    if threshold == 0:
        print("Motion signal is flat — no pose data found or video has no movement.")
        return []

    exchanges = _find_exchanges(rows, signal, fps,
                                threshold, min_duration_s, merge_gap_s)

    # Default output path: replace _stats.csv with _exchanges.csv
    if output_path is None:
        stem = stats_path
        if stem.endswith("_stats.csv"):
            stem = stem[:-len("_stats.csv")]
        elif stem.endswith(".csv"):
            stem = stem[:-4]
        output_path = stem + "_exchanges.csv"

    _fields = ["exchange_id", "start_frame", "start_time_s",
               "end_frame",   "end_time_s",  "duration_s", "peak_motion"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_fields)
        w.writeheader()
        w.writerows(exchanges)

    # Summary
    total_s      = rows[-1]["time_s"] - rows[0]["time_s"]
    exchange_s   = sum(e["duration_s"] for e in exchanges)
    pct_active   = 100 * exchange_s / total_s if total_s else 0

    print(f"\nExchange segmentation")
    print(f"  Input:           {stats_path}")
    print(f"  Output:          {output_path}")
    print(f"  FPS (detected):  {fps:.1f}")
    print(f"  Threshold:       {threshold:.1f}  (auto {_THRESHOLD_PCTILE}th pctile of motion signal)")
    print(f"  Video duration:  {total_s:.1f}s")
    print(f"  Exchanges found: {len(exchanges)}")

    if exchanges:
        durations = [e["duration_s"] for e in exchanges]
        print(f"  Total active:    {exchange_s:.1f}s  ({pct_active:.0f}% of video)")
        print(f"  Avg duration:    {sum(durations) / len(durations):.2f}s")
        print(f"  Min / Max:       {min(durations):.2f}s / {max(durations):.2f}s")
        print()
        print(f"  {'#':>3}  {'start':>8}  {'end':>8}  {'dur':>6}  {'peak':>6}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
        for e in exchanges:
            print(f"  {e['exchange_id']:>3}  "
                  f"{e['start_time_s']:>7.2f}s  "
                  f"{e['end_time_s']:>7.2f}s  "
                  f"{e['duration_s']:>5.2f}s  "
                  f"{e['peak_motion']:>6.1f}")

    return exchanges


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Segment a HEMA stats CSV into exchange windows."
    )
    ap.add_argument("--stats",        required=True,
                    help="Path to *_stats.csv produced by main.py")
    ap.add_argument("--output",       default=None,
                    help="Output CSV path (default: <stem>_exchanges.csv)")
    ap.add_argument("--threshold",    type=float, default=None,
                    help=f"Motion threshold (default: auto {_THRESHOLD_PCTILE}th percentile)")
    ap.add_argument("--min-duration", type=float, default=_MIN_DURATION_S,
                    dest="min_duration",
                    help=f"Min exchange duration in seconds (default: {_MIN_DURATION_S})")
    ap.add_argument("--merge-gap",    type=float, default=_MERGE_GAP_S,
                    dest="merge_gap",
                    help=f"Merge exchanges within this gap in seconds (default: {_MERGE_GAP_S})")
    args = ap.parse_args()

    if not os.path.exists(args.stats):
        print(f"Error: file not found: {args.stats}", file=sys.stderr)
        sys.exit(1)

    segment(
        args.stats,
        output_path    = args.output,
        threshold      = args.threshold,
        min_duration_s = args.min_duration,
        merge_gap_s    = args.merge_gap,
    )


if __name__ == "__main__":
    main()
