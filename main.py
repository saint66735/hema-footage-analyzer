import argparse
import os
import sys

import cv2

from video_io import VideoReader, VideoWriter
from person import PersonDetector
from pose import PoseDetector
from tracker import FighterTracker
from weapon import WeaponDetector
import guards as guards_mod
from motion import estimate_camera_motion
from overlay import draw_poses, draw_tracker_boxes, draw_weapon_results
from stats import StatsCollector


# ---------------------------------------------------------------------------
# Fighter seed selection
# ---------------------------------------------------------------------------

def _scan_for_seed(reader, person_detector, max_frames=500):
    """Return (frame, boxes, frame_idx) for first frame with ≥2 person detections."""
    print(f"Scanning for seed frame (up to {max_frames} frames) ...")
    for i, frame in enumerate(reader.frames()):
        if i % 60 == 0:
            print(f"\r  scanning frame {i} ...", end="", flush=True)
        boxes = person_detector.detect(frame)
        if len(boxes) >= 2:
            print(f"\r  Found {len(boxes)} people in frame {i}.        ")
            return frame.copy(), boxes, i
        if i >= max_frames:
            break
    return None, [], 0


def select_fighters(reader, person_detector, output_path):
    """
    Display a seed frame with numbered person bounding boxes, prompt the user to
    pick the two that are the actual fighters, and return their boxes.
    """
    frame, boxes, frame_idx = _scan_for_seed(reader, person_detector)

    if frame is None or len(boxes) < 2:
        sys.exit(
            "Error: could not find a frame with 2+ people in the first 500 frames.\n"
            "       Check that fighters enter the frame early in the video."
        )

    # Draw numbered boxes
    annotated = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, str(i), (x1 + 6, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

    # Always save to disk so the user can view it even without a display
    seed_path = os.path.splitext(output_path)[0] + "_seed.jpg"
    cv2.imwrite(seed_path, annotated)
    print(f"Seed frame saved to: {seed_path}")

    # Attempt interactive display (non-fatal if no display available)
    try:
        cv2.imshow("Fighter selection — press any key when ready", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("  (imshow not available — open the seed image file above)")

    print(f"\n{len(boxes)} people detected in frame {frame_idx}. Bounding boxes:")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"  [{i}]  x: {x1}–{x2}   y: {y1}–{y2}")
    print()

    while True:
        raw = input("Enter two fighter indices separated by a space (e.g. '0 2'): ").strip()
        parts = raw.split()
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                if 0 <= a < len(boxes) and 0 <= b < len(boxes) and a != b:
                    print(f"  Seeding tracker: Fighter A = box {a}, Fighter B = box {b}.\n")
                    return boxes[a], boxes[b], frame
            except ValueError:
                pass
        print(f"  Need two different numbers in range 0–{len(boxes) - 1}. Try again.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="HEMA fight analyzer — Iteration 1b: person-tracked pose + wrist-anchored weapon detection"
    )
    p.add_argument("--input",  required=True, help="Path to input video file")
    p.add_argument("--output", default=None,
                   help="Path for output annotated video (default: ./output/<input_stem>_annotated.mp4)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("output", f"{stem}_annotated.mp4")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    try:
        reader = VideoReader(args.input)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Frames: {reader.total_frames}  |  FPS: {reader.fps:.2f}  |  {reader.width}x{reader.height}")
    print()

    person_detector = PersonDetector()
    # One detector per fighter — each maintains independent VIDEO-mode temporal state
    pose_detectors   = [PoseDetector(), PoseDetector()]
    weapon_detectors = [WeaponDetector(), WeaponDetector()]
    guard_trackers   = [guards_mod.GuardTracker(), guards_mod.GuardTracker()]

    # --- Fighter selection ---
    box_a, box_b, seed_frame = select_fighters(reader, person_detector, args.output)
    reader.reset()  # rewind to process every frame from the start
    tracker = FighterTracker(box_a, box_b, seed_frame)

    total     = reader.total_frames or 1
    prev_gray = None
    pose_found = [0, 0]

    stats = StatsCollector(args.input, os.path.dirname(args.output),
                           reader.fps, reader.width, reader.height)

    with reader, VideoWriter(args.output, reader.fps, reader.width, reader.height) as writer:
        for frame_idx, frame in enumerate(reader.frames()):
            if frame_idx % 30 == 0:
                pct = frame_idx / total * 100
                print(f"\r  frame {frame_idx:>6}/{total}  ({pct:5.1f}%)", end="", flush=True)

            # Camera motion
            curr_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cam_motion = estimate_camera_motion(prev_gray, curr_gray) if prev_gray is not None else (0.0, 0.0)
            motion_ref = prev_gray   # keep old gray for weapon frame-diff before overwriting
            prev_gray  = curr_gray

            # Person tracking → fighter boxes
            try:
                track_detections = person_detector.track(frame)
            except Exception as exc:
                print(f"\n  [person] frame {frame_idx}: {exc}")
                track_detections = {}

            fighter_boxes, matched = tracker.update(track_detections, frame)

            # Per-fighter pose (keyed by fighter index)
            timestamp_ms     = int(frame_idx * 1000 / reader.fps)
            per_fighter_lms  = [None, None]
            for i, box in enumerate(fighter_boxes):
                try:
                    lms = pose_detectors[i].detect_in_box(frame, box, timestamp_ms)
                except Exception as exc:
                    print(f"\n  [pose] frame {frame_idx} fighter {i}: {exc}")
                    lms = []
                if lms:
                    per_fighter_lms[i] = lms
                    pose_found[i] += 1

            poses = [lms for lms in per_fighter_lms if lms is not None]

            # Per-fighter weapon detection + guard classification
            weapon_per_fighter = [[], []]
            guard_results      = [None, None]
            fh_f, fw_f         = frame.shape[:2]
            for i, lms in enumerate(per_fighter_lms):
                if lms:
                    try:
                        weapon_per_fighter[i] = weapon_detectors[i].detect(
                            lms, frame, curr_gray, motion_ref, fighter_boxes[i])
                    except Exception as exc:
                        print(f"\n  [weapon] frame {frame_idx} fighter {i}: {exc}")
                    try:
                        # Use fresh (non-stale) weapon line if available
                        w_line = None
                        for det in weapon_per_fighter[i]:
                            if det.get("line") and not det.get("stale"):
                                w_line = det["line"]
                                break
                        guard_results[i] = guard_trackers[i].update(
                            lms, fw_f, fh_f, weapon_line=w_line)
                    except Exception as exc:
                        print(f"\n  [guard] frame {frame_idx} fighter {i}: {exc}")

            all_weapon_results = weapon_per_fighter[0] + weapon_per_fighter[1]

            # Record stats
            stats.record(frame_idx, matched, per_fighter_lms, weapon_per_fighter, cam_motion,
                         fighter_boxes, guard_results)

            # Draw
            draw_tracker_boxes(frame, fighter_boxes, guard_results)
            draw_poses(frame, poses)
            draw_weapon_results(frame, all_weapon_results)
            writer.write(frame)

    for d in pose_detectors:
        d.close()

    stats.write()
    sm = stats.summary()

    print(f"\n\n--- Summary ---")
    print(f"Total frames        : {total}")
    print(f"A matched           : {sm['a_matched_%']}%   B matched    : {sm['b_matched_%']}%")
    print(f"A pose found        : {sm['a_pose_%']}%   B pose found : {sm['b_pose_%']}%")
    print(f"A vis mean          : {sm['a_vis_mean']}     B vis mean   : {sm['b_vis_mean']}")
    print(f"A jitter mean (px)  : {sm['a_jitter_mean']}   B jitter mean: {sm['b_jitter_mean']}")
    print(f"A weapon%           : {sm['a_weapon_%']}%   B weapon%    : {sm['b_weapon_%']}%")
    print(f"Stats CSV           : {stats.path}")
    print(f"Output video        : {args.output}")


if __name__ == "__main__":
    main()
