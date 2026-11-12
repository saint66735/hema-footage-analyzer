# HEMA Fight Analysis Tool

## Project context

Personal HEMA fight analysis tool. Longsword only, Meyer/Liechtenauer system. Built for personal use — not a general product. I (the project owner) am not proficient in Python; Claude Code handles all implementation.

Footage: single stationary camera, full tournament fights, sometimes at awkward (~45°) angles. One venue has a **mirror wall** that reflects fighters, judges and bystanders — never run a generic detector on the full frame and trust every person it finds.

End goal (MVP): raw fight video in → timestamped timeline of exchanges in relational notation out, e.g.

```
[00:23] Exchange 3
  L: Cut high right — initiated (vor)
  R: Intercept — Indes
  L: Counter in bind
  Outcome: inconclusive
```

MVP success test: run a fight I remember well; the timeline gets the broad strokes right.

Working rule: validate each layer on real footage before building on top of it. Every phase ends with a gate I check.

Full plan document: https://claude.ai/artifact/4dQKjNFETAGFykP2WiPKhq

## Current pipeline (what exists in code)

CLI:

```
python main.py --input <video> [--output <video>] [--max-frames N]
python segment.py --stats output/<name>_stats.csv
```

`main.py`, per frame:
1. `person.py` — YOLOv8 `person` detection + ByteTrack IDs
2. `tracker.py` — follows the two fighters I pick by hand on the seed frame (IOU + colour-histogram re-ID). Manual seed avoids judges, bystanders and mirror reflections.
3. `pose.py` — MediaPipe PoseLandmarker run separately inside each fighter's box
4. `weapon.py` — wrist-anchored blade search (Canny + Hough near the wrists, motion-masked, temporally smoothed). **To be replaced** by a custom model (see Phase 2).
5. `guards.py` — rule-based guard classifier (Vom Tag, Ochs, Pflug, Alber, Langort). Weak first draft, built ahead of plan; kept, reworked in Phase 4.
6. `motion.py` — camera-motion estimate; `overlay.py` — drawing; `video_io.py` — frame I/O
7. `stats.py` — per-frame stats CSV (tracking, pose visibility, jitter, weapon, distances, guards)

`segment.py` — post-processes the stats CSV into exchange windows (first draft).

Git: the folder is a git repo. Footage (`training footage/`), outputs (`output/`) and model weights (`*.pt`, `*.task`) are git-ignored.

GPU: RTX 3060 Ti (8 GB). `.venv` uses the CUDA build of PyTorch.

## Status

- Iteration 1b (person-first pose, manual seed + tracking, wrist-anchored weapon) — signed off 2026-06-23.
- Known issues from the four test clips:
  - Douglas vs Simas: fighter A lost for ~50% of frames.
  - Blade "detected" on ~100% of frames — almost certainly many false positives.
  - Arm/wrist tracking fails regularly, which breaks the wrist-anchored blade search.
  - Guard classifier heavily favours Pflug/Alber; Ochs/Vom Tag rarely appear.

**Current phase: Phase 1 — Detection fixes.**

## Phase plan (agreed 2026-11-12)

Do not build ahead of the current phase without my sign-off.

0. **Housekeeping** — git, this file, `--max-frames`, CUDA PyTorch. *Done.*
1. **Detection fixes** — fix the Douglas vs Simas tracking loss; measure how often arm tracking fails. Gate: both fighters tracked through all four clips.
1.5. **Reference labels** — all four current clips are the reference set. Build a small labelling helper (clicking, not spreadsheets). I mark exchange start/end, initiator, outcome, guards on sampled frames, and each sword's hilt + tip on a few hundred frames. Gate: a labels file per clip + a sword training set.
2. **Custom blade model** — fine-tune a pre-trained YOLO model to find each sword's hilt and tip directly in the frame, not anchored to the arms. Gate: blade found on most held-back frames with few false positives; otherwise label more and retrain.
3. **Exchange segmentation** — tune `segment.py` against my labels; add fighter distance and blade motion alongside body motion. Gate: ~80% of marked exchanges found, few false ones, start times within ~0.5 s.
4. **Guard recognition** — rework `guards.py` against labelled frames, use blade angle, smooth over time, add Lauerhut (club variant: blade against chest, hands at hips). Prefer "unknown" over a forced guard. Gate: right guard on most labelled frames.
5. **Per-exchange analysis** — in order of reliability: initiator → high/low, left/right → cut/thrust → single/compound tempo → relational tags (vor / Indes / counter / void / afterblow / simultaneous / reset). Gate: initiator matches my labels most of the time; other fields carry a confidence and are flagged when unsure.
6. **Timeline + feedback loop** — text timeline; the single question "Are you in this video: left, right, or no?"; for each exchange the tool proposes outcome and tags, I confirm or correct, and corrections are saved and fed back (rule tuning + more blade training data). Gate: the MVP test.

After the MVP: named technique identification → personal profile → full vocabulary system with human-in-the-loop labelling.

## Decisions (2026-11-12)

- Keep the existing guard code as a first draft.
- Guard recognition comes before per-exchange analysis.
- Outcomes come from a feedback loop (tool proposes, I confirm/correct), not from detection alone.
- Blade detection: train a custom model — **now in scope** (the earlier "heuristic only" rule is lifted).
- Reference set: all four clips in `training footage/`.

## Technical notes

- If a frame produces no usable pose (occlusion, blur, bad angle), don't crash — mark it and continue.
- Surface degraded detection visibly (confidence colouring, flags); never hide it.
- Keep modules separable; `weapon.py` stays isolated so the model can be swapped in.
- Performance is secondary to correctness and visibility of detection quality.

A separate vocabulary doc (guards, actions, bind states, tactical states, outcomes, named techniques) is being developed in parallel; relevant from Phase 4 onward.

## Explicit non-goals (whole project)

- No user or opponent profile (until the post-MVP phase)
- No named technique identification (until after the MVP)
- No multi-camera support
- No onboarding beyond the fighter-identity question
- No system/weapon configuration (longsword + Meyer/Liechtenauer is hardcoded)
