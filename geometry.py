"""
Geometry engine for Christmas tree packing.
Implements polygon operations, rotation, and overlap detection using SAT.
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class Polygon:
    """Represents a polygon with rotation and translation."""
    
    def __init__(self, vertices: np.ndarray):
        """
        Args:
            vertices: Nx2 array of (x, y) coordinates in local space
        """
        self.vertices = np.array(vertices, dtype=np.float64)
        self._centroid = None
        self._edges = None
        self._normals = None
    
    @property
    def centroid(self) -> np.ndarray:
        """Compute polygon centroid."""
        if self._centroid is None:
            self._centroid = np.mean(self.vertices, axis=0)
        return self._centroid
    
    @property
    def edges(self) -> np.ndarray:
        """Get edge vectors."""
        if self._edges is None:
            n = len(self.vertices)
            self._edges = np.array([
                self.vertices[(i + 1) % n] - self.vertices[i]
                for i in range(n)
            ])
        return self._edges
    
    @property
    def normals(self) -> np.ndarray:
        """Get edge normals (perpendicular to edges)."""
        if self._normals is None:
            edges = self.edges
            # Perpendicular vectors (rotate 90 degrees)
            self._normals = np.array([[-e[1], e[0]] for e in edges])
            # Normalize
            norms = np.linalg.norm(self._normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0  # Avoid division by zero
            self._normals = self._normals / norms
        return self._normals
    
    def rotate(self, angle_deg: float) -> 'Polygon':
        """Create a rotated copy of this polygon."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        centroid = self.centroid
        centered_vertices = self.vertices - centroid
        rotated_vertices = (rotation_matrix @ centered_vertices.T).T
        rotated_vertices += centroid
        
        new_poly = Polygon(rotated_vertices)
        return new_poly
    
    def translate(self, dx: float, dy: float) -> 'Polygon':
        """Create a translated copy of this polygon."""
        new_vertices = self.vertices + np.array([dx, dy])
        return Polygon(new_vertices)
    
    def transform(self, x: float, y: float, angle_deg: float) -> 'Polygon':
        """Apply rotation then translation."""
        rotated = self.rotate(angle_deg)
        return rotated.translate(x, y)
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)."""
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])
        return (min_x, min_y, max_x, max_y)
    
    def project_on_axis(self, axis: np.ndarray) -> Tuple[float, float]:
        """
        Project polygon vertices onto an axis.
        Returns (min_projection, max_projection).
        """
        projections = np.dot(self.vertices, axis)
        return (np.min(projections), np.max(projections))


def separating_axis_theorem(poly1: Polygon, poly2: Polygon, 
                            tolerance: float = 1e-9) -> bool:
    """
    Check if two polygons overlap using Separating Axis Theorem.
    Returns True if polygons overlap, False if separated.
    
    Args:
        poly1, poly2: Polygons to test
        tolerance: Numerical tolerance for separation
    
    Returns:
        True if overlapping, False if separated
    """
    # Test normals from poly1
    for normal in poly1.normals:
        min1, max1 = poly1.project_on_axis(normal)
        min2, max2 = poly2.project_on_axis(normal)
        
        if max1 < min2 - tolerance or max2 < min1 - tolerance:
            return False  # Separated
    
    # Test normals from poly2
    for normal in poly2.normals:
        min1, max1 = poly1.project_on_axis(normal)
        min2, max2 = poly2.project_on_axis(normal)
        
        if max1 < min2 - tolerance or max2 < min1 - tolerance:
            return False  # Separated
    
    return True  # No separating axis found, polygons overlap


def check_overlaps(polygons: List[Polygon], 
                   spatial_grid: Optional[dict] = None,
                   grid_size: float = 1.0) -> bool:
    """
    Check if any polygons in the list overlap.
    Uses spatial grid for acceleration if provided.
    
    Args:
        polygons: List of polygons
        spatial_grid: Optional dict mapping grid cell -> list of polygon indices
        grid_size: Size of grid cells
    
    Returns:
        True if any overlap exists
    """
    n = len(polygons)
    
    if spatial_grid is not None:
        # Use spatial grid to reduce checks
        checked_pairs = set()
        for cell_key, indices in spatial_grid.items():
            cell_indices = list(indices)
            for i in range(len(cell_indices)):
                for j in range(i + 1, len(cell_indices)):
                    idx1, idx2 = cell_indices[i], cell_indices[j]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    pair = (idx1, idx2)
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        if separating_axis_theorem(polygons[idx1], polygons[idx2]):
                            return True
    else:
        # Brute force check
        for i in range(n):
            for j in range(i + 1, n):
                if separating_axis_theorem(polygons[i], polygons[j]):
                    return True
    
    return False


def build_spatial_grid(polygons: List[Polygon], grid_size: float) -> dict:
    """
    Build a spatial grid for fast collision detection.
    
    Returns:
        Dict mapping (grid_x, grid_y) -> set of polygon indices
    """
    grid = {}
    
    for idx, poly in enumerate(polygons):
        min_x, min_y, max_x, max_y = poly.get_bounding_box()
        
        # Find all grid cells this polygon touches
        min_gx = int(np.floor(min_x / grid_size))
        max_gx = int(np.floor(max_x / grid_size))
        min_gy = int(np.floor(min_y / grid_size))
        max_gy = int(np.floor(max_y / grid_size))
        
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                key = (gx, gy)
                if key not in grid:
                    grid[key] = set()
                grid[key].add(idx)
    
    return grid


def compute_bounding_square(polygons: List[Polygon]) -> float:
    """
    Compute the side length of the smallest axis-aligned square
    that bounds all polygons.
    
    Returns:
        Side length of bounding square
    """
    if not polygons:
        return 0.0
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for poly in polygons:
        p_min_x, p_min_y, p_max_x, p_max_y = poly.get_bounding_box()
        min_x = min(min_x, p_min_x)
        min_y = min(min_y, p_min_y)
        max_x = max(max_x, p_max_x)
        max_y = max(max_y, p_max_y)
    
    width = max_x - min_x
    height = max_y - min_y
    side_length = max(width, height)
    
    return side_length


def load_tree_polygon(filepath: Optional[str] = None) -> Polygon:
    """
    Load Christmas tree polygon from file.
    If filepath is None, returns a default tree shape.
    
    Default shape: A simple triangular tree with base.
    """
    if filepath:
        try:
            # Try to load from file (CSV format: x,y per line)
            data = np.loadtxt(filepath, delimiter=',')
            return Polygon(data)
        except:
            pass
    
    # Default tree shape: triangle with base
    # This is a placeholder - should be replaced with actual tree data
    vertices = np.array([
        [0.0, 0.0],      # Base left
        [1.0, 0.0],      # Base right
        [0.5, 1.5],     # Top
        [0.0, 0.0]      # Close polygon
    ])
    return Polygon(vertices)

