"""
Visualization utilities for traffic density estimation.

Draws grid overlays, bounding boxes, and density metrics on video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.vision.grid import make_grid, index_to_rc


# Color scheme (BGR format for OpenCV)
COLORS = {
    'occupied': (0, 0, 255),      # Red
    'empty': (0, 255, 0),          # Green
    'grid_line': (255, 255, 255),  # White
    'bbox': (255, 165, 0),         # Orange
    'text_bg': (0, 0, 0),          # Black
    'text': (255, 255, 255),       # White
}


def draw_grid_overlay(
    frame,
    rows: int,
    cols: int,
    occupancy: List[int],
    alpha: float = 0.3,
    show_indices: bool = False
) -> np.ndarray:
    """
    Draw a colored grid overlay showing occupied/empty regions.
    
    Args:
        frame: BGR image (numpy array).
        rows: Number of grid rows.
        cols: Number of grid columns.
        occupancy: Binary list (1 = occupied, 0 = empty).
        alpha: Transparency of the overlay (0-1).
        show_indices: Whether to draw region index numbers.
    
    Returns:
        Frame with overlay drawn.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    regions = make_grid(rows, cols, w, h)
    
    for i, (x1, y1, x2, y2) in enumerate(regions):
        color = COLORS['occupied'] if occupancy[i] == 1 else COLORS['empty']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        if show_indices:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(
                overlay, str(i), (cx - 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1
            )
    
    # Blend overlay with original frame
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw grid lines
    cell_w = w / cols
    cell_h = h / rows
    
    for c in range(cols + 1):
        x = int(c * cell_w)
        cv2.line(result, (x, 0), (x, h), COLORS['grid_line'], 1)
    
    for r in range(rows + 1):
        y = int(r * cell_h)
        cv2.line(result, (0, y), (w, y), COLORS['grid_line'], 1)
    
    return result


def draw_bounding_boxes(
    frame,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    color: Tuple[int, int, int] = COLORS['bbox'],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    
    Args:
        frame: BGR image.
        boxes: List of (x1, y1, x2, y2) bounding boxes.
        labels: Optional list of labels for each box.
        confidences: Optional list of confidence scores.
        color: Box color in BGR.
        thickness: Line thickness.
    
    Returns:
        Frame with boxes drawn.
    """
    result = frame.copy()
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if labels or confidences:
            label_parts = []
            if labels and i < len(labels):
                label_parts.append(labels[i])
            if confidences and i < len(confidences):
                label_parts.append(f"{confidences[i]:.2f}")
            
            label = " ".join(label_parts)
            if label:
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1),
                    COLORS['text_bg'], -1
                )
                cv2.putText(
                    result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1
                )
    
    return result


def draw_density_info(
    frame,
    classical_density: float,
    quantum_density: Optional[float] = None,
    classical_count: Optional[int] = None,
    quantum_count: Optional[int] = None,
    total_regions: Optional[int] = None,
    num_detections: Optional[int] = None,
    position: str = "top-left"
) -> np.ndarray:
    """
    Draw density information panel on frame.
    
    Args:
        frame: BGR image.
        classical_density: Classical density (0-1).
        quantum_density: Quantum estimated density (0-1).
        classical_count: Number of occupied regions (classical).
        quantum_count: Number of occupied regions (quantum estimate).
        total_regions: Total number of grid regions.
        num_detections: Number of detected vehicles.
        position: Panel position ("top-left", "top-right", "bottom-left", "bottom-right").
    
    Returns:
        Frame with info panel drawn.
    """
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Build info lines
    lines = [
        f"Classical Density: {classical_density*100:.1f}%"
    ]
    
    if quantum_density is not None:
        lines.append(f"Quantum Density:   {quantum_density*100:.1f}%")
    
    if classical_count is not None and total_regions is not None:
        lines.append(f"Occupied: {classical_count}/{total_regions} regions")
    
    if quantum_count is not None:
        lines.append(f"Quantum Est.: {quantum_count} regions")
    
    if num_detections is not None:
        lines.append(f"Vehicles Detected: {num_detections}")
    
    # Calculate panel size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    padding = 10
    line_height = 25
    
    max_text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    
    panel_w = max_text_width + 2 * padding
    panel_h = len(lines) * line_height + 2 * padding
    
    # Determine position
    if position == "top-left":
        px, py = 10, 10
    elif position == "top-right":
        px, py = w - panel_w - 10, 10
    elif position == "bottom-left":
        px, py = 10, h - panel_h - 10
    else:  # bottom-right
        px, py = w - panel_w - 10, h - panel_h - 10
    
    # Draw semi-transparent background
    overlay = result.copy()
    cv2.rectangle(
        overlay, (px, py), (px + panel_w, py + panel_h),
        COLORS['text_bg'], -1
    )
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
    
    # Draw text
    for i, line in enumerate(lines):
        y = py + padding + (i + 1) * line_height - 5
        cv2.putText(
            result, line, (px + padding, y),
            font, font_scale, COLORS['text'], thickness
        )
    
    return result


def create_visualization(
    frame,
    boxes: List[Tuple[int, int, int, int]],
    occupancy: List[int],
    rows: int,
    cols: int,
    classical_density: float,
    quantum_density: Optional[float] = None,
    quantum_count: Optional[int] = None,
    show_grid: bool = True,
    show_boxes: bool = True,
    show_info: bool = True,
    grid_alpha: float = 0.25,
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None
) -> np.ndarray:
    """
    Create complete visualization with all elements.
    
    Args:
        frame: BGR image.
        boxes: Detected vehicle bounding boxes.
        occupancy: Binary occupancy grid.
        rows: Grid rows.
        cols: Grid columns.
        classical_density: Classical density value.
        quantum_density: Quantum density estimate (optional).
        quantum_count: Quantum region count estimate (optional).
        show_grid: Whether to show grid overlay.
        show_boxes: Whether to show detection boxes.
        show_info: Whether to show info panel.
        grid_alpha: Grid overlay transparency.
        labels: Labels for bounding boxes.
        confidences: Confidence scores for bounding boxes.
    
    Returns:
        Fully annotated frame.
    """
    result = frame.copy()
    
    if show_grid:
        result = draw_grid_overlay(result, rows, cols, occupancy, alpha=grid_alpha)
    
    if show_boxes:
        result = draw_bounding_boxes(
            result, boxes, labels=labels, confidences=confidences
        )
    
    if show_info:
        classical_count = sum(occupancy)
        total_regions = len(occupancy)
        result = draw_density_info(
            result,
            classical_density=classical_density,
            quantum_density=quantum_density,
            classical_count=classical_count,
            quantum_count=quantum_count,
            total_regions=total_regions,
            num_detections=len(boxes)
        )
    
    return result
