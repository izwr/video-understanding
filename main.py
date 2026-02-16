import argparse
import json
import sys

from video_processor.processor import (
    extract_clip_frames,
    get_video_info,
    sample_frames,
    sample_frames_by_seconds,
)


def format_context(
    object_inventory: dict[str, int],
    action_results: list,
) -> str:
    """Format detection results into structured text for the LLM prompt."""
    lines: list[str] = ["## Objects Detected"]

    if object_inventory:
        for name, count in sorted(object_inventory.items(), key=lambda x: -x[1]):
            lines.append(f"- {name}: seen in {count} frame(s)")
    else:
        lines.append("- No objects detected")

    lines.append("")
    lines.append("## Actions Detected")
    if action_results:
        for clip_result in action_results:
            for ar in clip_result.results:
                if ar.actions:
                    action_strs = [f"{label} ({conf:.2f})" for label, conf in ar.actions]
                    lines.append(
                        f"- Frame {ar.clip_start_frame}: person at "
                        f"bbox {ar.person_bbox}: {', '.join(action_strs)}"
                    )
    if not any(line.startswith("- Frame") for line in lines):
        lines.append("- No actions detected")

    return "\n".join(lines)


def run_pipeline(video_path: str, verbose: bool = False) -> dict:
    """Run the full 4-stage video analysis pipeline.

    Returns:
        Dict with summary, object_inventory, action_results, and video_info.
    """
    from analyzer.action_detector import ActionDetector
    from analyzer.object_detector import ObjectDetector, build_object_inventory
    from generator.summarizer import VideoSummarizer

    # Stage 0: Video info
    info = get_video_info(video_path)
    if verbose:
        print(f"Video: {info['width']}x{info['height']}, "
              f"{info['fps']:.1f} fps, {info['duration_seconds']:.1f}s, "
              f"{info['total_frames']} frames")

    # Stage 1: Sample frames (1 per second)
    if verbose:
        print("Stage 1: Sampling frames (1/sec)...")
    frames = sample_frames_by_seconds(video_path, every_n_seconds=1.0)
    if verbose:
        print(f"  Sampled {len(frames)} frames")

    # Stage 2: YOLO object detection on sampled frames
    if verbose:
        print("Stage 2: Running YOLO object detection...")
    detector = ObjectDetector()
    frame_detections = detector.detect_frames(frames)
    object_inventory = build_object_inventory(frame_detections)
    if verbose:
        print(f"  Detected {len(object_inventory)} unique object types")
        for name, count in sorted(object_inventory.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name}: {count} frame(s)")

    # Stage 3: SlowFast action classification on person detections
    if verbose:
        print("Stage 3: Running SlowFast action classification...")
    action_detector = ActionDetector()
    action_results = []
    fps = info["fps"]

    for fd in frame_detections:
        persons = fd.person_detections
        if not persons:
            continue

        # Calculate the start frame in the original video for this sampled frame
        original_frame_idx = int(fd.frame_index * fps)
        clip_frames = extract_clip_frames(video_path, original_frame_idx, num_frames=32)
        if len(clip_frames) < 8:
            continue

        person_bboxes = [p.bbox_xyxy for p in persons]
        clip_result = action_detector.classify_clip(
            clip_frames, person_bboxes, clip_start_frame=original_frame_idx
        )
        action_results.append(clip_result)

    if verbose:
        total_actions = sum(
            len(ar.actions)
            for cr in action_results
            for ar in cr.results
        )
        print(f"  Classified actions for {len(action_results)} clips, "
              f"{total_actions} action predictions total")

    # Stage 4: VideoLLAMA3 summarization
    if verbose:
        print("Stage 4: Generating summary with VideoLLaMA3...")
    context_text = format_context(object_inventory, action_results)
    if verbose:
        print(f"  Context:\n{context_text}")

    summarizer = VideoSummarizer()
    summary = summarizer.summarize(video_path, context_text)
    if verbose:
        print(f"\n--- Summary ---\n{summary}\n")

    return {
        "video_info": info,
        "object_inventory": object_inventory,
        "action_results": [
            {
                "clip_start_frame": cr.clip_start_frame,
                "clip_end_frame": cr.clip_end_frame,
                "persons": [
                    {
                        "bbox": ar.person_bbox,
                        "actions": ar.actions,
                    }
                    for ar in cr.results
                ],
            }
            for cr in action_results
        ],
        "context_text": context_text,
        "summary": summary,
    }


def cmd_sample(args: argparse.Namespace) -> None:
    if args.mode == "frame":
        frames = sample_frames(args.video, interval=int(args.interval), max_frames=args.max_frames)
    else:
        frames = sample_frames_by_seconds(
            args.video, every_n_seconds=args.interval, max_frames=args.max_frames
        )
    print(f"Sampled {len(frames)} frames")


def cmd_analyze(args: argparse.Namespace) -> None:
    results = run_pipeline(args.video, verbose=args.verbose)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_json}")
    if not args.verbose:
        print(results["summary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Video understanding pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    # sample subcommand (backward compatible)
    sample_parser = subparsers.add_parser("sample", help="Sample frames from a video file.")
    sample_parser.add_argument("video", help="Path to the input video file.")
    sample_parser.add_argument(
        "-m", "--mode", choices=["frame", "time"], default="time",
        help="Sampling mode: 'frame' (every Nth frame) or 'time' (every N seconds).",
    )
    sample_parser.add_argument(
        "-i", "--interval", type=float, default=1.0,
        help="Interval value: frame count for 'frame' mode, seconds for 'time' mode.",
    )
    sample_parser.add_argument(
        "-n", "--max-frames", type=int, default=None,
        help="Maximum number of frames to sample.",
    )

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Run full video analysis pipeline.")
    analyze_parser.add_argument("video", help="Path to the input video file.")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    analyze_parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save full results as JSON.",
    )

    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
