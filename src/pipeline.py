"""
Main pipeline: Video → YOLO → Quantum Counting → Visualization.

This module integrates all components for end-to-end traffic density
estimation using quantum counting.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from src.vision.grid import make_grid
from src.vision.boxes_to_occupancy import boxes_to_occupancy
from src.vision.visualization import create_visualization
from src.quantum.quantum_counting import (
    quantum_counting,
    compute_classical_count,
)
from src.utils.logging import DensityLogger, FrameLog

# Import video processor (with fallback to mock)
try:
    from src.vision.video_processor import VideoProcessor, YOLO_AVAILABLE, get_video_info
    if not YOLO_AVAILABLE:
        from src.vision.video_processor import MockVideoProcessor as VideoProcessor
        print("Warning: YOLO not available, using mock detections")
except ImportError as e:
    print(f"Error importing video processor: {e}")
    sys.exit(1)


def process_video_with_quantum(
    video_path: str,
    output_path: Optional[str] = None,
    rows: int = 4,
    cols: int = 4,
    precision_qubits: int = 4,
    shots: int = 512,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    show_preview: bool = True,
    confidence_threshold: float = 0.5,
    use_quantum: bool = True,
    device: str = "cuda",
    quantum_every_n: int = 5,
    enable_logging: bool = True,
    log_dir: str = "logs"
):
    """
    Process a video file with quantum traffic density estimation.
    
    Args:
        video_path: Path to input video.
        output_path: Path for output video (None = no save).
        rows: Grid rows (must result in power of 2 total regions).
        cols: Grid columns.
        precision_qubits: QPE precision qubits.
        shots: Quantum measurement shots (lower = faster).
        skip_frames: Frames to skip between processing.
        max_frames: Maximum frames to process.
        show_preview: Whether to show live preview window.
        confidence_threshold: YOLO confidence threshold.
        use_quantum: Whether to run quantum counting (slower).
        device: Device for YOLO ('cpu', 'cuda', 'mps').
        quantum_every_n: Run quantum counting every N frames (1=every frame, 5=every 5th).
        enable_logging: Whether to log results to CSV.
        log_dir: Directory for log files.
    """
    # Validate grid size is power of 2
    N = rows * cols
    if N & (N - 1) != 0:
        print(f"Error: Grid size {rows}x{cols}={N} must be a power of 2")
        sys.exit(1)
    
    # Get video info
    info = get_video_info(video_path)
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration']:.2f}s ({info['frame_count']} frames)")
    
    print(f"\nConfiguration:")
    print(f"  Grid: {rows}x{cols} = {N} regions")
    print(f"  Quantum counting: {'Enabled' if use_quantum else 'Disabled'}")
    if use_quantum:
        print(f"  Precision qubits: {precision_qubits}")
        print(f"  Measurement shots: {shots}")
        print(f"  Quantum every N frames: {quantum_every_n}")
    print(f"  Logging: {'Enabled' if enable_logging else 'Disabled'}")
    
    # Initialize logger
    logger = None
    if enable_logging:
        video_name = Path(video_path).name
        logger = DensityLogger(output_dir=log_dir, video_name=video_name)
        logger.set_config(
            grid=f"{rows}x{cols}",
            total_regions=N,
            quantum_enabled=use_quantum,
            precision_qubits=precision_qubits,
            shots=shots,
            quantum_every_n=quantum_every_n,
            confidence_threshold=confidence_threshold,
            device=device,
            skip_frames=skip_frames
        )
    
    # Initialize video processor
    print(f"\nInitializing YOLO model...")
    processor = VideoProcessor(
        confidence_threshold=confidence_threshold,
        device=device
    )
    
    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            info['fps'] / (skip_frames + 1),
            (info['width'], info['height'])
        )
    
    # Process frames
    print(f"\nProcessing video...")
    print("Press 'q' to quit, 'p' to pause, 's' to save screenshot")
    print("-" * 60)
    
    frame_times = []
    paused = False
    last_quantum_density = None
    last_quantum_count = None
    frames_since_quantum = 0
    target_fps = 30
    frame_duration = 1.0 / target_fps
    
    try:
        for result in processor.process_video(video_path, skip_frames=skip_frames, max_frames=max_frames):
            if paused:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('p'):
                    paused = False
                elif key == ord('q'):
                    break
                continue
            
            start_time = time.time()
            
            # Convert detections to occupancy grid
            occupancy = boxes_to_occupancy(
                result.boxes_xyxy,
                rows, cols,
                info['width'], info['height']
            )
            
            # Classical density (always computed - fast)
            classical_count = compute_classical_count(occupancy)
            classical_density = classical_count / N
            
            # Quantum density (run every N frames for performance)
            quantum_density = last_quantum_density
            quantum_count = last_quantum_count
            quantum_ran_this_frame = False
            
            if use_quantum and frames_since_quantum >= quantum_every_n:
                quantum_count, quantum_density, _ = quantum_counting(
                    occupancy,
                    precision_qubits=precision_qubits,
                    shots=shots
                )
                last_quantum_density = quantum_density
                last_quantum_count = quantum_count
                frames_since_quantum = 0
                quantum_ran_this_frame = True
            else:
                frames_since_quantum += 1
            
            # Create visualization
            labels = [d.class_name for d in result.detections]
            confidences = [d.confidence for d in result.detections]
            
            vis_frame = create_visualization(
                result.frame,
                result.boxes_xyxy,
                occupancy,
                rows, cols,
                classical_density,
                quantum_density=quantum_density,
                quantum_count=quantum_count,
                labels=labels,
                confidences=confidences
            )
            
            # Calculate processing time
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            
            # Log frame data
            if logger:
                timestamp_ms = result.frame_number / info['fps'] * 1000 if info['fps'] > 0 else 0
                logger.log_frame(FrameLog(
                    frame_number=result.frame_number,
                    timestamp_ms=timestamp_ms,
                    num_detections=len(result.detections),
                    classical_count=classical_count,
                    classical_density=classical_density,
                    quantum_count=quantum_count,
                    quantum_density=quantum_density,
                    quantum_ran=quantum_ran_this_frame,
                    processing_time_ms=elapsed * 1000
                ))
            
            # Print progress
            fps_estimate = 1 / elapsed if elapsed > 0 else 0
            print(f"\rFrame {result.frame_number}: "
                  f"Vehicles={len(result.detections)}, "
                  f"Classical={classical_density*100:.1f}%, "
                  f"{'Quantum=' + f'{quantum_density*100:.1f}%' if quantum_density else ''} "
                  f"({fps_estimate:.1f} fps)", end="")
            
            # Show preview
            if show_preview:
                cv2.imshow("Quantum Traffic Density", vis_frame)
                
                # Cap frame rate to 30 FPS for smooth playback
                wait_time = max(1, int((frame_duration - elapsed) * 1000))
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('p'):
                    paused = True
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{result.frame_number}.png"
                    cv2.imwrite(screenshot_path, vis_frame)
                    print(f"\nSaved screenshot: {screenshot_path}")
            
            # Write to output video
            if writer:
                writer.write(vis_frame)
    
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "-" * 60)
    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        print(f"\nProcessed {len(frame_times)} frames")
        print(f"Average processing time: {avg_time*1000:.1f}ms/frame ({1/avg_time:.1f} fps)")
    
    if output_path:
        print(f"Output saved to: {output_path}")
    
    # Save and print logging summary
    if logger:
        summary_path = logger.save_summary()
        logger.print_summary()
        print(f"\nDetailed logs: {logger.csv_path}")
        print(f"Summary report: {summary_path}")


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    rows: int = 4,
    cols: int = 4,
    precision_qubits: int = 4,
    shots: int = 1024,
    confidence_threshold: float = 0.5,
    device: str = "cpu"
):
    """
    Process a single image with quantum traffic density estimation.
    
    Args:
        image_path: Path to input image.
        output_path: Path for output image (None = display only).
        rows: Grid rows.
        cols: Grid columns.
        precision_qubits: QPE precision qubits.
        shots: Quantum measurement shots.
        confidence_threshold: YOLO confidence threshold.
        device: Device for YOLO.
    """
    # Validate grid size
    N = rows * cols
    if N & (N - 1) != 0:
        print(f"Error: Grid size {rows}x{cols}={N} must be a power of 2")
        sys.exit(1)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)
    
    h, w = frame.shape[:2]
    print(f"\nImage: {image_path}")
    print(f"  Resolution: {w}x{h}")
    print(f"  Grid: {rows}x{cols} = {N} regions")
    
    # Initialize processor and detect vehicles
    print(f"\nDetecting vehicles...")
    processor = VideoProcessor(
        confidence_threshold=confidence_threshold,
        device=device
    )
    result = processor.process_frame(frame)
    
    print(f"  Found {len(result.detections)} vehicles")
    
    # Convert to occupancy
    occupancy = boxes_to_occupancy(result.boxes_xyxy, rows, cols, w, h)
    
    # Classical density
    classical_count = compute_classical_count(occupancy)
    classical_density = classical_count / N
    
    # Quantum counting
    print(f"\nRunning quantum counting...")
    quantum_count, quantum_density, counts = quantum_counting(
        occupancy,
        precision_qubits=precision_qubits,
        shots=shots
    )
    
    # Print results
    print(f"\nResults:")
    print(f"  Classical: {classical_count}/{N} regions = {classical_density*100:.1f}%")
    print(f"  Quantum:   {quantum_count}/{N} regions = {quantum_density*100:.1f}%")
    print(f"  Error:     {abs(quantum_count - classical_count)} regions")
    
    # Create visualization
    labels = [d.class_name for d in result.detections]
    confidences = [d.confidence for d in result.detections]
    
    vis_frame = create_visualization(
        frame,
        result.boxes_xyxy,
        occupancy,
        rows, cols,
        classical_density,
        quantum_density=quantum_density,
        quantum_count=quantum_count,
        labels=labels,
        confidences=confidences
    )
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, vis_frame)
        print(f"\nSaved to: {output_path}")
    else:
        print("\nDisplaying result (press any key to close)...")
        cv2.imshow("Quantum Traffic Density", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Traffic Density Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with quantum counting
  python -m src.pipeline --video traffic.mp4

  # Process a single image
  python -m src.pipeline --image intersection.jpg --output result.png

  # Process video without quantum (faster, classical only)
  python -m src.pipeline --video traffic.mp4 --no-quantum

  # Use GPU for YOLO inference
  python -m src.pipeline --video traffic.mp4 --device cuda

  # Custom grid size (must be power of 2)
  python -m src.pipeline --video traffic.mp4 --rows 8 --cols 8
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--image', type=str, help='Path to input image')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview window')
    
    # Grid options
    parser.add_argument('--rows', type=int, default=4, help='Grid rows (default: 4)')
    parser.add_argument('--cols', type=int, default=4, help='Grid columns (default: 4)')
    
    # Quantum options
    parser.add_argument('--no-quantum', action='store_true',
                       help='Disable quantum counting (faster)')
    parser.add_argument('--precision', type=int, default=6,
                       help='QPE precision qubits (default: 6, more=accurate but slower)')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Quantum measurement shots (default: 1024, more=accurate)')
    parser.add_argument('--quantum-every', type=int, default=5,
                       help='Run quantum counting every N frames (default: 5)')
    
    # Processing options
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Frames to skip between processing (default: 0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for YOLO inference (default: cuda)')
    
    # Logging options
    parser.add_argument('--no-log', action='store_true',
                       help='Disable logging to CSV')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for log files (default: logs)')
    
    args = parser.parse_args()
    
    # Validate grid size
    N = args.rows * args.cols
    if N & (N - 1) != 0:
        parser.error(f"Grid size {args.rows}x{args.cols}={N} must be a power of 2")
    
    # Process based on input type
    if args.video:
        process_video_with_quantum(
            video_path=args.video,
            output_path=args.output,
            rows=args.rows,
            cols=args.cols,
            precision_qubits=args.precision,
            shots=args.shots,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            show_preview=not args.no_preview,
            confidence_threshold=args.confidence,
            use_quantum=not args.no_quantum,
            device=args.device,
            quantum_every_n=args.quantum_every,
            enable_logging=not args.no_log,
            log_dir=args.log_dir
        )
    
    elif args.image:
        process_image(
            image_path=args.image,
            output_path=args.output,
            rows=args.rows,
            cols=args.cols,
            precision_qubits=args.precision,
            shots=args.shots,
            confidence_threshold=args.confidence,
            device=args.device
        )


if __name__ == "__main__":
    main()
