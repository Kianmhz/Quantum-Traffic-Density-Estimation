"""
Grid utilities for dividing an image into regions.
"""

from typing import List, Tuple


def make_grid(rows: int, cols: int, image_w: int, image_h: int) -> List[Tuple[int, int, int, int]]:
    """
    Divide an image into a grid of rows x cols regions.
    
    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        image_w: Image width in pixels.
        image_h: Image height in pixels.
    
    Returns:
        List of (x1, y1, x2, y2) tuples for each region, ordered row-major
        (i.e., region index = r * cols + c).
    """
    cell_w = image_w / cols
    cell_h = image_h / rows
    
    regions = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            regions.append((x1, y1, x2, y2))
    
    return regions


def region_index(r: int, c: int, cols: int) -> int:
    """
    Convert (row, column) to linear region index.
    
    Args:
        r: Row index (0-based).
        c: Column index (0-based).
        cols: Number of columns in the grid.
    
    Returns:
        Linear index = r * cols + c.
    """
    return r * cols + c


def index_to_rc(i: int, cols: int) -> Tuple[int, int]:
    """
    Convert linear region index to (row, column).
    
    Args:
        i: Linear index.
        cols: Number of columns in the grid.
    
    Returns:
        (row, column) tuple.
    """
    r = i // cols
    c = i % cols
    return (r, c)
