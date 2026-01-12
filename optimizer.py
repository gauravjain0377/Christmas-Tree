"""
Main optimization logic for tree packing.
Handles initial placement strategies and coordinates optimization.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from geometry import Polygon, load_tree_polygon
from annealing import SimulatedAnnealing


def hexagonal_packing(num_trees: int, spacing: float = 2.0) -> List[Tuple[float, float, float]]:
    """
    Generate initial positions using hexagonal close packing.
    
    Returns:
        List of (x, y, angle) tuples
    """
    positions = []
    
    if num_trees == 0:
        return positions
    
    # Start at origin
    positions.append((0.0, 0.0, 0.0))
    
    if num_trees == 1:
        return positions
    
    # Hexagonal packing parameters
    # Distance between centers in hexagonal packing
    hex_spacing = spacing * math.sqrt(3) / 2
    
    layer = 1
    placed = 1
    
    while placed < num_trees:
        # Number of trees in this layer
        trees_in_layer = min(6 * layer, num_trees - placed)
        
        # Place trees in hexagonal ring
        for i in range(trees_in_layer):
            if placed >= num_trees:
                break
            
            angle = 2 * math.pi * i / trees_in_layer
            radius = layer * hex_spacing
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Try different rotations
            rot_angle = (i * 60) % 360  # Rotate based on position
            
            positions.append((x, y, rot_angle))
            placed += 1
        
        layer += 1
    
    return positions


def spiral_packing(num_trees: int, spacing: float = 2.0) -> List[Tuple[float, float, float]]:
    """
    Generate initial positions using spiral packing.
    
    Returns:
        List of (x, y, angle) tuples
    """
    positions = []
    
    if num_trees == 0:
        return positions
    
    # Golden angle for spiral
    golden_angle = math.pi * (3 - math.sqrt(5))
    
    for i in range(num_trees):
        # Spiral parameters
        radius = spacing * math.sqrt(i) * 0.8
        angle = i * golden_angle
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        # Rotation based on spiral angle
        rot_angle = (math.degrees(angle) * 0.5) % 360
        
        positions.append((x, y, rot_angle))
    
    return positions


def grid_packing(num_trees: int, spacing: float = 2.0) -> List[Tuple[float, float, float]]:
    """
    Generate initial positions using grid packing.
    
    Returns:
        List of (x, y, angle) tuples
    """
    positions = []
    
    if num_trees == 0:
        return positions
    
    cols = int(math.ceil(math.sqrt(num_trees)))
    rows = int(math.ceil(num_trees / cols))
    
    for i in range(num_trees):
        row = i // cols
        col = i % cols
        
        x = col * spacing
        y = row * spacing
        
        # Alternate rotations
        rot_angle = (i * 30) % 360
        
        positions.append((x, y, rot_angle))
    
    return positions


NUM_RESTARTS = 5


def optimize_configuration(base_polygon: Polygon,
                          num_trees: int,
                          num_restarts: int = NUM_RESTARTS,
                          max_iterations: int = 5000,
                          random_seed: Optional[int] = None,
                          verbose: bool = False) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Optimize placement for a single configuration.
    
    Args:
        base_polygon: Base tree polygon
        num_trees: Number of trees to place
        num_restarts: Number of random restarts
        max_iterations: Max iterations per restart
        random_seed: Random seed
        verbose: Print restart progress
    
    Returns:
        (best_positions, best_objective)
    """
    best_positions = None
    best_objective = float('inf')
    
    # Try different initial placement strategies
    strategies = [
        ("hexagonal", hexagonal_packing),
        ("spiral", spiral_packing),
        ("grid", grid_packing)
    ]
    
    total_restarts = len(strategies) * num_restarts
    restart_count = 0
    
    for strategy_name, strategy_func in strategies:
        for restart in range(num_restarts):
            restart_count += 1
            
            # Generate initial positions
            seed = (random_seed + restart_count * 1000) if random_seed is not None else None
            initial_positions = strategy_func(num_trees, spacing=2.0)
            
            # Add small random perturbations
            if seed is not None:
                np.random.seed(seed)
            for i in range(len(initial_positions)):
                x, y, angle = initial_positions[i]
                x += np.random.uniform(-0.1, 0.1)
                y += np.random.uniform(-0.1, 0.1)
                angle += np.random.uniform(-5, 5)
                angle = angle % 360
                initial_positions[i] = (x, y, angle)
            
            # Run simulated annealing
            sa = SimulatedAnnealing(
                base_polygon=base_polygon,
                num_trees=num_trees,
                initial_temp=2.0,
                final_temp=0.01,
                cooling_rate=0.995,
                max_iterations=max_iterations,
                translation_step=0.03,
                rotation_step=2.0,
                random_seed=seed
            )
            
            sa.initialize(initial_positions)
            positions, objective = sa.optimize(verbose=False)
            
            side_length = objective ** 0.5
            
            if verbose:
                print(f"  Restart {restart_count}/{total_restarts} ({strategy_name}) → best side = {side_length:.6f}")
            
            if objective < best_objective:
                best_objective = objective
                best_positions = positions
    
    return (best_positions, best_objective)


def validate_solution(base_polygon: Polygon, 
                     positions: List[Tuple[float, float, float]]) -> bool:
    """
    Validate that solution has no overlaps.
    
    Returns:
        True if valid (no overlaps), False otherwise
    """
    from geometry import separating_axis_theorem
    
    polygons = []
    for x, y, angle in positions:
        poly = base_polygon.transform(x, y, angle)
        polygons.append(poly)
    
    n = len(polygons)
    for i in range(n):
        for j in range(i + 1, n):
            if separating_axis_theorem(polygons[i], polygons[j]):
                return False
    
    return True

