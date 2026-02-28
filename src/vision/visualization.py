"""
Visualization utilities for traffic density estimation.

Draws grid overlays, bounding boxes, and density metrics on video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.vision.grid import make_grid, index_to_rc, get_direction_region_indices


# Color scheme (BGR format for OpenCV)
# High-contrast palette: occupied cells are vivid, empty cells are very dark
COLORS = {
    'occupied':      (0, 0, 255),      # Bright red — clearly stands out
    'empty':         (40, 40, 40),      # Near-black — almost invisible tint
    'grid_line':     (200, 200, 200),   # Light grey
    'bbox':          (255, 165, 0),     # Orange
    'text_bg':       (0, 0, 0),         # Black
    'text':          (255, 255, 255),   # White
    'occupied_border': (0, 0, 200),     # Darker red border for occupied
    'dir_a':         (255, 200, 0),     # Bright cyan — Direction A occupied
    'dir_b':         (0, 200, 255),     # Bright orange — Direction B occupied
    'dir_a_empty':   (80, 50, 0),       # Very dark blue — Direction A empty
    'dir_b_empty':   (0, 50, 80),       # Very dark orange — Direction B empty
    'dir_a_border':  (255, 140, 0),     # Cyan border
    'dir_b_border':  (0, 140, 255),     # Orange border
}


def draw_grid_overlay(
    frame,
    rows: int,
    cols: int,
    occupancy: List[int],
    alpha: float = 0.45,
    show_indices: bool = False
) -> np.ndarray:
    """
    Draw a colored grid overlay showing occupied/empty regions.

    Occupied cells are drawn with a vivid red fill + a red border so they
    pop against the video.  Empty cells use a barely visible dark tint so
    the contrast between occupied and empty is immediately obvious.
    
    Args:
        frame: BGR image (numpy array).
        rows: Number of grid rows.
        cols: Number of grid columns.
        occupancy: Binary list (1 = occupied, 0 = empty).
        alpha: Transparency of the overlay (0-1).  Higher = more visible.
        show_indices: Whether to draw region index numbers.
    
    Returns:
        Frame with overlay drawn.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    regions = make_grid(rows, cols, w, h)
    
    # Use different alpha for occupied vs empty to boost contrast
    alpha_occ = min(alpha + 0.15, 1.0)   # occupied gets a stronger tint
    alpha_emp = max(alpha - 0.15, 0.05)   # empty gets a weaker tint

    # Build two separate overlays so we can blend at different strengths
    occ_overlay = frame.copy()
    emp_overlay = frame.copy()

    for i, (x1, y1, x2, y2) in enumerate(regions):
        if occupancy[i] == 1:
            cv2.rectangle(occ_overlay, (x1, y1), (x2, y2), COLORS['occupied'], -1)
        else:
            cv2.rectangle(emp_overlay, (x1, y1), (x2, y2), COLORS['empty'], -1)
        
        if show_indices:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(
                occ_overlay, str(i), (cx - 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1
            )
    
    # Blend occupied overlay (strong)
    result = cv2.addWeighted(occ_overlay, alpha_occ, frame, 1 - alpha_occ, 0)
    # Blend empty overlay on top (subtle)
    result = cv2.addWeighted(emp_overlay, alpha_emp, result, 1 - alpha_emp, 0)

    # Draw a solid border around occupied cells for extra pop
    for i, (x1, y1, x2, y2) in enumerate(regions):
        if occupancy[i] == 1:
            cv2.rectangle(result, (x1, y1), (x2, y2), COLORS['occupied_border'], 3)
    
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


def draw_direction_grid_overlay(
    frame,
    rows: int,
    cols: int,
    occupancy_A: List[int],
    occupancy_B: List[int],
    indices_A: List[int],
    indices_B: List[int],
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw a grid overlay that colour-codes regions by direction and occupancy.

    Direction A regions use blue tones, Direction B regions use orange tones.
    Occupied cells are brighter, empty cells are dimmer.

    Args:
        frame: BGR image.
        rows, cols: Grid dimensions.
        occupancy_A: Full-length occupancy (only indices_A entries matter).
        occupancy_B: Full-length occupancy (only indices_B entries matter).
        indices_A, indices_B: Region indices per direction.
        alpha: Overlay transparency.

    Returns:
        Frame with coloured overlay.
    """
    h, w = frame.shape[:2]
    regions = make_grid(rows, cols, w, h)

    set_A = set(indices_A)
    set_B = set(indices_B)

    alpha_occ = min(alpha + 0.15, 1.0)
    alpha_emp = max(alpha - 0.15, 0.05)

    occ_overlay = frame.copy()
    emp_overlay = frame.copy()

    for i, (x1, y1, x2, y2) in enumerate(regions):
        if i in set_A:
            is_occ = occupancy_A[i] == 1
            color = COLORS['dir_a'] if is_occ else COLORS['dir_a_empty']
        elif i in set_B:
            is_occ = occupancy_B[i] == 1
            color = COLORS['dir_b'] if is_occ else COLORS['dir_b_empty']
        else:
            is_occ = False
            color = COLORS['empty']

        if is_occ:
            cv2.rectangle(occ_overlay, (x1, y1), (x2, y2), color, -1)
        else:
            cv2.rectangle(emp_overlay, (x1, y1), (x2, y2), color, -1)

    # Blend occupied (strong) then empty (subtle)
    result = cv2.addWeighted(occ_overlay, alpha_occ, frame, 1 - alpha_occ, 0)
    result = cv2.addWeighted(emp_overlay, alpha_emp, result, 1 - alpha_emp, 0)

    # Thick coloured border around occupied cells
    for i, (x1, y1, x2, y2) in enumerate(regions):
        if i in set_A and occupancy_A[i] == 1:
            cv2.rectangle(result, (x1, y1), (x2, y2), COLORS['dir_a_border'], 3)
        elif i in set_B and occupancy_B[i] == 1:
            cv2.rectangle(result, (x1, y1), (x2, y2), COLORS['dir_b_border'], 3)

    # Grid lines
    cell_w = w / cols
    cell_h = h / rows
    for c in range(cols + 1):
        x = int(c * cell_w)
        cv2.line(result, (x, 0), (x, h), COLORS['grid_line'], 1)
    for r in range(rows + 1):
        y = int(r * cell_h)
        cv2.line(result, (0, y), (w, y), COLORS['grid_line'], 1)

    # Draw split divider line (thicker)
    if len(indices_A) > 0 and len(indices_B) > 0:
        # Determine split orientation from region indices
        r_a, c_a = index_to_rc(indices_A[-1], cols)
        r_b, c_b = index_to_rc(indices_B[0], cols)
        if c_a < c_b:  # vertical split
            mid_x = int((cols // 2) * cell_w)
            cv2.line(result, (mid_x, 0), (mid_x, h), (0, 255, 255), 3)
        else:  # horizontal split
            mid_y = int((rows // 2) * cell_h)
            cv2.line(result, (0, mid_y), (w, mid_y), (0, 255, 255), 3)

    return result


def draw_direction_comparison(
    frame,
    density_A: float,
    density_B: float,
    count_A: int,
    count_B: int,
    num_vehicles_A: int,
    num_vehicles_B: int,
    regions_A: int,
    regions_B: int,
    position: str = "top-right"
) -> np.ndarray:
    """
    Draw an info panel comparing Direction A and Direction B densities.

    Args:
        frame: BGR image.
        density_A, density_B: Density values (0-1) per direction.
        count_A, count_B: Occupied region counts.
        num_vehicles_A, num_vehicles_B: Vehicle counts per direction.
        regions_A, regions_B: Total region counts per direction.
        position: Panel position.

    Returns:
        Frame with comparison panel drawn.
    """
    result = frame.copy()
    h, w = frame.shape[:2]

    # Determine which direction is denser
    if density_A > density_B:
        comparison = "A is denser"
    elif density_B > density_A:
        comparison = "B is denser"
    else:
        comparison = "Equal density"

    diff = abs(density_A - density_B) * 100

    lines = [
        "--- Direction Comparison ---",
        f"Dir A: {density_A*100:.1f}% ({count_A}/{regions_A} reg, {num_vehicles_A} veh)",
        f"Dir B: {density_B*100:.1f}% ({count_B}/{regions_B} reg, {num_vehicles_B} veh)",
        f"Diff:  {diff:.1f}pp  ({comparison})",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 10
    line_height = 24

    max_text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_w = max_text_width + 2 * padding
    panel_h = len(lines) * line_height + 2 * padding

    if position == "top-left":
        px, py = 10, 10
    elif position == "top-right":
        px, py = w - panel_w - 10, 10
    elif position == "bottom-left":
        px, py = 10, h - panel_h - 10
    else:
        px, py = w - panel_w - 10, h - panel_h - 10

    # Semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  COLORS['text_bg'], -1)
    result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

    # Draw text with colour coding per direction
    for i, line in enumerate(lines):
        y = py + padding + (i + 1) * line_height - 5
        if line.startswith("Dir A"):
            color = COLORS['dir_a']
        elif line.startswith("Dir B"):
            color = COLORS['dir_b']
        else:
            color = COLORS['text']
        cv2.putText(result, line, (px + padding, y),
                    font, font_scale, color, thickness)

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
    confidences: Optional[List[float]] = None,
    direction_data: Optional[dict] = None,
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
        direction_data: Optional dict from directional_occupancy() with keys
            boxes_A, boxes_B, occupancy_A, occupancy_B, indices_A, indices_B,
            count_A, count_B, density_A, density_B.
    
    Returns:
        Fully annotated frame.
    """
    result = frame.copy()

    # --- Grid overlay ---
    if show_grid:
        if direction_data is not None:
            result = draw_direction_grid_overlay(
                result, rows, cols,
                direction_data["occupancy_A"],
                direction_data["occupancy_B"],
                direction_data["indices_A"],
                direction_data["indices_B"],
                alpha=grid_alpha,
            )
        else:
            result = draw_grid_overlay(result, rows, cols, occupancy, alpha=grid_alpha)

    # --- Bounding boxes (colour-coded by direction when available) ---
    if show_boxes:
        if direction_data is not None:
            result = draw_bounding_boxes(
                result, direction_data["boxes_A"],
                color=COLORS['dir_a'], thickness=2,
            )
            result = draw_bounding_boxes(
                result, direction_data["boxes_B"],
                color=COLORS['dir_b'], thickness=2,
            )
        else:
            result = draw_bounding_boxes(
                result, boxes, labels=labels, confidences=confidences
            )

    # --- Info panels ---
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
            num_detections=len(boxes),
            position="top-left",
        )

        # Direction comparison panel (top-right)
        if direction_data is not None:
            result = draw_direction_comparison(
                result,
                density_A=direction_data["density_A"],
                density_B=direction_data["density_B"],
                count_A=direction_data["count_A"],
                count_B=direction_data["count_B"],
                num_vehicles_A=len(direction_data["boxes_A"]),
                num_vehicles_B=len(direction_data["boxes_B"]),
                regions_A=len(direction_data["indices_A"]),
                regions_B=len(direction_data["indices_B"]),
                position="top-right",
            )

    return result
