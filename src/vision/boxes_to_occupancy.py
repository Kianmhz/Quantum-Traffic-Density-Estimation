"""
Convert bounding boxes to a binary occupancy grid.
"""

from typing import List, Tuple, Dict

from src.vision.grid import get_direction_region_indices


def boxes_to_occupancy(
    boxes_xyxy: List[Tuple[int, int, int, int]],
    rows: int,
    cols: int,
    image_w: int,
    image_h: int
) -> List[int]:
    """
    Convert a list of bounding boxes to a binary occupancy grid.

    Each vehicle is assigned to exactly one cell — the cell that contains its
    centre point.  This avoids double-counting vehicles that straddle a cell
    boundary.

    Args:
        boxes_xyxy: List of (x1, y1, x2, y2) bounding boxes (e.g., from YOLO).
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        image_w: Image width in pixels.
        image_h: Image height in pixels.

    Returns:
        List of length rows*cols with 1 for occupied regions, 0 otherwise.
        Ordered row-major (index = r * cols + c).
    """
    N = rows * cols
    occupancy = [0] * N

    cell_w = image_w / cols
    cell_h = image_h / rows

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        c = min(int(cx / cell_w), cols - 1)
        r = min(int(cy / cell_h), rows - 1)

        occupancy[r * cols + c] = 1

    return occupancy


def classify_boxes_by_direction(
    boxes_xyxy: List[Tuple[int, int, int, int]],
    image_w: int,
    image_h: int,
    split: str = "vertical"
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Classify bounding boxes into Direction A and Direction B based on
    the centre point of each box.

    For a vertical split:
        centre_x < image_w / 2  →  Direction A (left half)
        otherwise               →  Direction B (right half)
    For a horizontal split:
        centre_y < image_h / 2  →  Direction A (top half)
        otherwise               →  Direction B (bottom half)

    Args:
        boxes_xyxy: List of (x1, y1, x2, y2) bounding boxes.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
        split: "vertical" or "horizontal".

    Returns:
        (boxes_A, boxes_B) — two lists of bounding boxes.
    """
    boxes_A: List[Tuple[int, int, int, int]] = []
    boxes_B: List[Tuple[int, int, int, int]] = []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if split == "vertical":
            if cx < image_w / 2:
                boxes_A.append(box)
            else:
                boxes_B.append(box)
        else:  # horizontal
            if cy < image_h / 2:
                boxes_A.append(box)
            else:
                boxes_B.append(box)

    return boxes_A, boxes_B


def directional_occupancy(
    boxes_xyxy: List[Tuple[int, int, int, int]],
    rows: int,
    cols: int,
    image_w: int,
    image_h: int,
    split: str = "vertical"
) -> Dict[str, object]:
    """
    Compute per-direction occupancy and density.

    Splits detected vehicles into Direction A / B, builds an occupancy grid
    for each direction (using only that direction's boxes), and calculates
    density over only the grid regions that belong to that direction's half.

    Args:
        boxes_xyxy: All detected bounding boxes.
        rows: Grid rows.
        cols: Grid columns.
        image_w: Image width.
        image_h: Image height.
        split: "vertical" or "horizontal".

    Returns:
        Dict with keys:
            boxes_A, boxes_B          — bounding boxes per direction
            occupancy_A, occupancy_B  — full-length occupancy arrays (only their half is meaningful)
            indices_A, indices_B      — region indices belonging to each direction
            count_A, count_B          — occupied region count per direction
            density_A, density_B      — density per direction (0-1)
    """
    boxes_A, boxes_B = classify_boxes_by_direction(
        boxes_xyxy, image_w, image_h, split
    )

    # Full occupancy using only direction-specific boxes
    occupancy_A = boxes_to_occupancy(boxes_A, rows, cols, image_w, image_h)
    occupancy_B = boxes_to_occupancy(boxes_B, rows, cols, image_w, image_h)

    # Region indices for each half
    indices_A, indices_B = get_direction_region_indices(rows, cols, split)

    # Count occupied cells only in the relevant half
    count_A = sum(occupancy_A[i] for i in indices_A)
    count_B = sum(occupancy_B[i] for i in indices_B)

    n_A = len(indices_A) if indices_A else 1
    n_B = len(indices_B) if indices_B else 1

    return {
        "boxes_A": boxes_A,
        "boxes_B": boxes_B,
        "occupancy_A": occupancy_A,
        "occupancy_B": occupancy_B,
        "indices_A": indices_A,
        "indices_B": indices_B,
        "count_A": count_A,
        "count_B": count_B,
        "density_A": count_A / n_A,
        "density_B": count_B / n_B,
    }
