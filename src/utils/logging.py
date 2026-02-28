"""
Logging utilities for tracking quantum vs classical density estimates.

Provides CSV logging and summary statistics for analysis.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
import statistics


@dataclass
class FrameLog:
    """Log entry for a single frame."""
    frame_number: int
    timestamp_ms: float
    num_detections: int
    classical_count: int
    classical_density: float
    quantum_count: Optional[int]
    quantum_density: Optional[float]
    quantum_ran: bool  # Whether quantum was computed this frame
    processing_time_ms: float
    # Per-direction fields (None when direction splitting is disabled)
    density_A: Optional[float] = None
    density_B: Optional[float] = None
    count_A: Optional[int] = None
    count_B: Optional[int] = None
    vehicles_A: Optional[int] = None
    vehicles_B: Optional[int] = None
    
    @property
    def error(self) -> Optional[int]:
        """Absolute error in region count."""
        if self.quantum_count is None:
            return None
        return abs(self.quantum_count - self.classical_count)
    
    @property
    def relative_error(self) -> Optional[float]:
        """Relative error as percentage."""
        if self.quantum_count is None or self.classical_count == 0:
            return None
        return abs(self.quantum_count - self.classical_count) / self.classical_count * 100


@dataclass 
class SessionStats:
    """Aggregate statistics for a processing session."""
    total_frames: int = 0
    quantum_frames: int = 0
    avg_classical_density: float = 0.0
    avg_quantum_density: float = 0.0
    avg_error: float = 0.0
    avg_relative_error: float = 0.0
    max_error: int = 0
    min_error: int = 0
    std_error: float = 0.0
    avg_fps: float = 0.0
    total_time_s: float = 0.0
    # Direction stats
    avg_density_A: float = 0.0
    avg_density_B: float = 0.0
    direction_enabled: bool = False


class DensityLogger:
    """
    Logger for tracking quantum vs classical density estimates.
    
    Writes per-frame data to CSV and computes summary statistics.
    """
    
    def __init__(
        self,
        output_dir: str = "logs",
        session_name: Optional[str] = None,
        video_name: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory for log files.
            session_name: Custom session name (default: timestamp).
            video_name: Name of the video being processed.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.video_name = video_name or "unknown"
        
        # Create CSV file
        self.csv_path = self.output_dir / f"density_log_{session_name}.csv"
        self.logs: List[FrameLog] = []
        
        # Initialize CSV with headers
        self._init_csv()
        
        # Config info for the summary
        self.config: Dict[str, Any] = {}
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'frame_number', 'timestamp_ms', 'num_detections',
            'classical_count', 'classical_density',
            'quantum_count', 'quantum_density', 'quantum_ran',
            'error', 'relative_error_pct', 'processing_time_ms',
            'density_A', 'density_B', 'count_A', 'count_B',
            'vehicles_A', 'vehicles_B',
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def set_config(self, **kwargs):
        """Store configuration for summary."""
        self.config.update(kwargs)
    
    def log_frame(self, log: FrameLog):
        """
        Log a single frame's results.
        
        Args:
            log: FrameLog entry.
        """
        self.logs.append(log)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log.frame_number,
                f"{log.timestamp_ms:.2f}",
                log.num_detections,
                log.classical_count,
                f"{log.classical_density:.4f}",
                log.quantum_count if log.quantum_count is not None else "",
                f"{log.quantum_density:.4f}" if log.quantum_density is not None else "",
                log.quantum_ran,
                log.error if log.error is not None else "",
                f"{log.relative_error:.2f}" if log.relative_error is not None else "",
                f"{log.processing_time_ms:.2f}",
                f"{log.density_A:.4f}" if log.density_A is not None else "",
                f"{log.density_B:.4f}" if log.density_B is not None else "",
                log.count_A if log.count_A is not None else "",
                log.count_B if log.count_B is not None else "",
                log.vehicles_A if log.vehicles_A is not None else "",
                log.vehicles_B if log.vehicles_B is not None else "",
            ])
    
    def compute_stats(self) -> SessionStats:
        """Compute aggregate statistics from logged frames."""
        if not self.logs:
            return SessionStats()
        
        stats = SessionStats()
        stats.total_frames = len(self.logs)
        
        # Filter frames where quantum was computed
        quantum_logs = [l for l in self.logs if l.quantum_ran and l.quantum_count is not None]
        stats.quantum_frames = len(quantum_logs)
        
        # Classical stats
        classical_densities = [l.classical_density for l in self.logs]
        stats.avg_classical_density = statistics.mean(classical_densities)
        
        # Quantum stats (only frames where quantum ran)
        if quantum_logs:
            quantum_densities = [l.quantum_density for l in quantum_logs if l.quantum_density is not None]
            if quantum_densities:
                stats.avg_quantum_density = statistics.mean(quantum_densities)
            
            # Error stats
            errors = [l.error for l in quantum_logs if l.error is not None]
            if errors:
                stats.avg_error = statistics.mean(errors)
                stats.max_error = max(errors)
                stats.min_error = min(errors)
                if len(errors) > 1:
                    stats.std_error = statistics.stdev(errors)
            
            rel_errors = [l.relative_error for l in quantum_logs if l.relative_error is not None]
            if rel_errors:
                stats.avg_relative_error = statistics.mean(rel_errors)
        
        # Timing stats
        total_time = sum(l.processing_time_ms for l in self.logs)
        stats.total_time_s = total_time / 1000
        stats.avg_fps = len(self.logs) / stats.total_time_s if stats.total_time_s > 0 else 0

        # Direction stats
        dir_logs_A = [l.density_A for l in self.logs if l.density_A is not None]
        dir_logs_B = [l.density_B for l in self.logs if l.density_B is not None]
        if dir_logs_A:
            stats.direction_enabled = True
            stats.avg_density_A = statistics.mean(dir_logs_A)
        if dir_logs_B:
            stats.direction_enabled = True
            stats.avg_density_B = statistics.mean(dir_logs_B)

        return stats
    
    def save_summary(self) -> str:
        """
        Save a summary report and return its path.
        
        Returns:
            Path to the summary file.
        """
        stats = self.compute_stats()
        summary_path = self.output_dir / f"summary_{self.session_name}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("QUANTUM TRAFFIC DENSITY ESTIMATION - SESSION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Video: {self.video_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            if self.config:
                f.write("Configuration:\n")
                for key, value in self.config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Processing stats
            f.write("Processing Statistics:\n")
            f.write(f"  Total frames processed: {stats.total_frames}\n")
            f.write(f"  Frames with quantum counting: {stats.quantum_frames}\n")
            f.write(f"  Total processing time: {stats.total_time_s:.2f}s\n")
            f.write(f"  Average FPS: {stats.avg_fps:.2f}\n\n")
            
            # Density stats
            f.write("Density Statistics:\n")
            f.write(f"  Average classical density: {stats.avg_classical_density*100:.2f}%\n")
            f.write(f"  Average quantum density: {stats.avg_quantum_density*100:.2f}%\n\n")

            # Direction comparison
            if stats.direction_enabled:
                f.write("Direction Comparison:\n")
                f.write(f"  Average Direction A density: {stats.avg_density_A*100:.2f}%\n")
                f.write(f"  Average Direction B density: {stats.avg_density_B*100:.2f}%\n")
                diff_pp = abs(stats.avg_density_A - stats.avg_density_B) * 100
                denser = "A" if stats.avg_density_A > stats.avg_density_B else "B" if stats.avg_density_B > stats.avg_density_A else "Equal"
                f.write(f"  Density difference: {diff_pp:.2f} percentage points\n")
                f.write(f"  Denser direction: {denser}\n\n")
            
            # Error analysis
            f.write("Quantum Estimation Error Analysis:\n")
            f.write(f"  Average absolute error: {stats.avg_error:.2f} regions\n")
            f.write(f"  Average relative error: {stats.avg_relative_error:.2f}%\n")
            f.write(f"  Min error: {stats.min_error} regions\n")
            f.write(f"  Max error: {stats.max_error} regions\n")
            f.write(f"  Std deviation: {stats.std_error:.2f} regions\n\n")
            
            # Error distribution
            if self.logs:
                quantum_logs = [l for l in self.logs if l.quantum_ran and l.error is not None]
                if quantum_logs:
                    f.write("Error Distribution:\n")
                    error_counts = {}
                    for l in quantum_logs:
                        e = l.error
                        error_counts[e] = error_counts.get(e, 0) + 1
                    
                    for error in sorted(error_counts.keys()):
                        count = error_counts[error]
                        pct = count / len(quantum_logs) * 100
                        bar = "█" * int(pct / 2)
                        f.write(f"  {error:2d} regions: {count:4d} ({pct:5.1f}%) {bar}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"CSV log saved to: {self.csv_path}\n")
            f.write("=" * 70 + "\n")
        
        return str(summary_path)
    
    def print_summary(self):
        """Print summary to console."""
        stats = self.compute_stats()
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total frames: {stats.total_frames}")
        print(f"Quantum frames: {stats.quantum_frames}")
        print(f"Average FPS: {stats.avg_fps:.2f}")
        print(f"\nDensity Comparison:")
        print(f"  Classical avg: {stats.avg_classical_density*100:.2f}%")
        print(f"  Quantum avg:   {stats.avg_quantum_density*100:.2f}%")
        print(f"\nQuantum Error Analysis:")
        print(f"  Mean error: {stats.avg_error:.2f} ± {stats.std_error:.2f} regions")
        print(f"  Mean relative error: {stats.avg_relative_error:.2f}%")
        print(f"  Error range: [{stats.min_error}, {stats.max_error}] regions")
        if stats.direction_enabled:
            print(f"\nDirection Comparison:")
            print(f"  Avg Dir A density: {stats.avg_density_A*100:.2f}%")
            print(f"  Avg Dir B density: {stats.avg_density_B*100:.2f}%")
            diff_pp = abs(stats.avg_density_A - stats.avg_density_B) * 100
            denser = "A" if stats.avg_density_A > stats.avg_density_B else "B" if stats.avg_density_B > stats.avg_density_A else "Equal"
            print(f"  Difference: {diff_pp:.2f}pp (Direction {denser} is denser)")
        print(f"\nLogs saved to: {self.csv_path}")
        print("=" * 60)
