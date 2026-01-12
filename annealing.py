"""
Simulated annealing optimizer for tree packing.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from geometry import Polygon, check_overlaps, compute_bounding_square, build_spatial_grid


class SimulatedAnnealing:
    """Simulated annealing optimizer for tree placement."""
    
    def __init__(self, 
                 base_polygon: Polygon,
                 num_trees: int,
                 initial_temp: float = 2.0,
                 final_temp: float = 0.01,
                 cooling_rate: float = 0.995,
                 max_iterations: int = 10000,
                 translation_step: float = 0.03,
                 rotation_step: float = 2.0,
                 random_seed: Optional[int] = None):
        """
        Args:
            base_polygon: Base tree polygon (before transformation)
            num_trees: Number of trees to place
            initial_temp: Starting temperature
            final_temp: Final temperature
            cooling_rate: Temperature decay factor per iteration
            max_iterations: Maximum iterations
            translation_step: Max translation distance per move
            rotation_step: Max rotation angle per move (degrees)
            random_seed: Random seed for reproducibility
        """
        self.base_polygon = base_polygon
        self.num_trees = num_trees
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.translation_step = translation_step
        self.rotation_step = rotation_step
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # State: list of (x, y, angle) tuples
        self.positions = []
        self.best_positions = []
        self.best_objective = float('inf')
        self.current_objective = float('inf')
        
        # Cached polygons
        self._polygons_cache = None
        self._grid_size = 1.0
    
    def initialize(self, initial_positions: List[Tuple[float, float, float]]):
        """Initialize with given positions."""
        self.positions = initial_positions.copy()
        self.best_positions = initial_positions.copy()
        self._update_objective()
        self.best_objective = self.current_objective
    
    def _update_objective(self):
        """Update current objective value."""
        polygons = self._get_polygons()
        self.current_objective = compute_bounding_square(polygons) ** 2
    
    def _get_polygons(self) -> List[Polygon]:
        """Get current polygon placements."""
        polygons = []
        for x, y, angle in self.positions:
            poly = self.base_polygon.transform(x, y, angle)
            polygons.append(poly)
        return polygons
    
    def _propose_move(self, temperature: float) -> Tuple[int, float, float, float]:
        """
        Propose a random move.
        Returns (tree_index, new_x, new_y, new_angle).
        """
        tree_idx = random.randint(0, self.num_trees - 1)
        old_x, old_y, old_angle = self.positions[tree_idx]
        
        # Scale moves with temperature (larger moves early, smaller later)
        temp_factor = min(1.0, temperature / self.initial_temp)
        translation_scale = self.translation_step * (0.5 + 0.5 * temp_factor)
        rotation_scale = self.rotation_step * (0.5 + 0.5 * temp_factor)
        
        # Random translation
        dx = np.random.uniform(-translation_scale, translation_scale)
        dy = np.random.uniform(-translation_scale, translation_scale)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Random rotation
        dangle = np.random.uniform(-rotation_scale, rotation_scale)
        new_angle = (old_angle + dangle) % 360.0
        
        return (tree_idx, new_x, new_y, new_angle)
    
    def _check_valid_move(self, tree_idx: int, new_x: float, new_y: float, 
                          new_angle: float) -> bool:
        """Check if proposed move is valid (no overlaps)."""
        # Create new polygon for this tree
        new_poly = self.base_polygon.transform(new_x, new_y, new_angle)
        
        # Check against all other trees
        for i, (x, y, angle) in enumerate(self.positions):
            if i == tree_idx:
                continue
            other_poly = self.base_polygon.transform(x, y, angle)
            from geometry import separating_axis_theorem
            if separating_axis_theorem(new_poly, other_poly):
                return False
        
        return True
    
    def optimize(self, verbose: bool = False) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Run simulated annealing optimization.
        
        Returns:
            (best_positions, best_objective)
        """
        temp = self.initial_temp
        iteration = 0
        no_improvement_count = 0
        max_no_improvement = self.max_iterations // 10
        
        while temp > self.final_temp and iteration < self.max_iterations:
            # Propose a move
            tree_idx, new_x, new_y, new_angle = self._propose_move(temp)
            
            # Check if valid (no overlaps)
            if not self._check_valid_move(tree_idx, new_x, new_y, new_angle):
                iteration += 1
                temp *= self.cooling_rate
                continue
            
            # Save old state
            old_x, old_y, old_angle = self.positions[tree_idx]
            old_objective = self.current_objective
            
            # Apply move
            self.positions[tree_idx] = (new_x, new_y, new_angle)
            self._update_objective()
            
            # Accept or reject
            delta = self.current_objective - old_objective
            
            if delta < 0:
                # Always accept improvements
                if self.current_objective < self.best_objective:
                    self.best_objective = self.current_objective
                    self.best_positions = [p for p in self.positions]
                    no_improvement_count = 0
            else:
                # Accept with probability exp(-delta/temp)
                if temp > 0 and random.random() < np.exp(-delta / temp):
                    # Accept worse solution
                    pass
                else:
                    # Reject: revert move
                    self.positions[tree_idx] = (old_x, old_y, old_angle)
                    self.current_objective = old_objective
            
            # Cool down
            temp *= self.cooling_rate
            iteration += 1
            no_improvement_count += 1
            
            # Early stopping if no improvement for a while
            if no_improvement_count > max_no_improvement and temp < self.initial_temp * 0.1:
                break
            
            if verbose and iteration % 1000 == 0:
                print(f"  Iteration {iteration}, temp={temp:.3f}, "
                      f"obj={self.current_objective:.6f}, "
                      f"best={self.best_objective:.6f}")
        
        return (self.best_positions, self.best_objective)

