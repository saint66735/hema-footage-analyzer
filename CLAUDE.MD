# HEMA Fight Analysis Tool

## Project context

Personal HEMA fight analysis tool. Longsword only, Meyer/Liechtenauer system. Built for personal use — not a general product. I (the project owner) am not proficient in Python; Claude Code handles all implementation.

Footage: single stationary camera, full tournament fights, sometimes at awkward (~45°) angles.

## Current stage: Iteration 1

We are validating whether pose and weapon detection are usable on real footage before building anything further. **Do not build segmentation, classification, notation, the fighter-identity question, or any UI until this iteration is validated against real footage and explicitly signed off.**

### Goal

Take a raw fight video, output a new video file with pose and weapon detection drawn on top, so detection quality can be judged by eye.

### CLI

```
python main.py --input <path_to_video> --output <path_to_output_video>
```

### Pipeline, per frame

1. MediaPipe pose estimation — extract body landmarks for visible people
2. YOLOv8 weapon detection — best-effort longsword/blade detection (no off-the-shelf model is trained on historical swords; poor or absent detection is expected — surface it, don't hide it)
3. Draw pose skeleton over each detected person
4. Draw a box or line over any detected weapon
5. Color-code skeleton by detection confidence (green/yellow/red) so degraded tracking is visible, not silent
6. Write annotated frame to output video

### File structure

```
hema_tool/
├── main.py          # CLI entrypoint, orchestrates the pipeline
├── pose.py          # MediaPipe wrapper — landmark extraction per frame
├── weapon.py        # YOLOv8 wrapper — weapon detection per frame
├── overlay.py       # Draws skeleton + weapon overlay + confidence coloring
├── video_io.py      # Frame read/write via OpenCV
└── requirements.txt
```

Keep modules separable — `weapon.py` in particular will likely need heuristics or fine-tuning later, and that should stay isolated from the rest of the pipeline.

### Technical notes

- No background subtraction needed yet
- If a frame produces no usable pose (occlusion, motion blur, bad angle), don't crash — skip overlay or mark it clearly, and continue
- Performance is not a concern yet — correctness and visibility of detection quality matters more than speed

### Success criteria for this iteration

Watching the output video, I can answer by eye:
- Does the skeleton roughly track each fighter through footwork, cuts, and binds?
- Does the weapon overlay track the blade at all, lose it constantly, or false-positive on other things?
- Does the confidence coloring line up with frames that visibly look wrong?

---

## Iteration 1 results & known issues (as of first real test)

Ran on real tournament footage (26,908 frames). Results:
- Pose detection: missing on 11.7% of frames, otherwise solid on whichever person(s) it locks onto
- **Problem: only finds 1, sometimes 2, people inconsistently.** MediaPipe's standard Pose solution is single-person by design — it isn't reliably finding multiple distinct people because nothing is feeding it separate per-person regions.
- **Problem: weapon detection nearly absent** — 130 detections across 26,908 frames. No stock YOLO model has a sword/blade class; whatever proxy was used isn't working.
- **Environmental hazard discovered: the room has a mirror wall in the background.** It reflects fighters, bystanders, and judges. Any generic person/object detector run on the full frame will detect reflections as if they were real people. This must be accounted for, not just multi-person detection in general.

### Iteration 1b — fixes required

1. **Person detection before pose.** Run YOLOv8's stock `person` class detector on the frame first to get a bounding box per person (this class is well-trained, unlike weapons). Run MediaPipe Pose separately inside each detected box, rather than relying on MediaPipe to find multiple people on its own.
2. **Manual seed-and-track for fighter identification.** On frame 1 (or first valid frame), the user clicks/selects the two bounding boxes that are the real fighters. A simple IOU-based tracker follows those two specific boxes through the rest of the video. This both solves "fighters vs. judges/bystanders" and avoids the mirror-reflection problem, since reflections are never selected in the first place. Do not attempt automatic fighter detection (motion energy, spatial zones, etc.) yet — manual seed is simpler and more reliable for this stage.
3. **Wrist-anchored weapon detection.** Abandon whole-frame weapon search. Instead, anchor to the wrist landmarks already produced by pose, and search a small region radiating from each wrist for a long straight line (e.g. Hough line transform on edges in that local region). This ties weapon detection to something pose already gives reliably and narrows the search space drastically.
4. Fine-tuning a custom-trained sword detection model is explicitly **out of scope** for this iteration — heuristic only.

## Future phases (not in scope yet — do not build ahead of these)

1. Exchange segmentation — detect movement bursts vs resets, timestamp exchanges
2. Per-exchange analysis — initiator, action type, direction, tempo, relational tags (vor/Indes/counter/void/afterblow/simultaneous/reset), outcome
3. Guard recognition library — geometric rules per guard
4. Named technique identification (optional enrichment layer)
5. Personal profile built from analyzed footage
6. Full vocabulary system with human-in-the-loop labeling

A separate vocabulary doc (guards, actions, bind states, tactical states, outcomes, named techniques) is being developed in parallel and will become relevant from Phase 3 onward. Not needed yet.

## Explicit non-goals (full project, not just this iteration)

- No user profile
- No opponent profile
- No named technique identification (until Phase 3)
- No multi-camera support
- No onboarding beyond the eventual fighter-identity question
- No system/weapon configuration (longsword + Meyer/Liechtenauer is hardcoded)