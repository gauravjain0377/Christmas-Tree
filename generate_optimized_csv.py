"""
Ultra smart optimized CSV generator for Kaggle Christmas Tree Packing Challenge.
Uses sample submission as baseline and applies targeted optimizations.
"""

import numpy as np
import csv
import time
from typing import List, Tuple, Optional
from geometry import Polygon, compute_bounding_square, separating_axis_theorem
from annealing import SimulatedAnnealing


def load_tree_shape() -> Polygon:
    """Load tree shape - optimized for tight packing."""
    vertices = np.array([
        [-0.35, 0.0],      # Base left
        [0.35, 0.0],       # Base right
        [0.2, 0.55],       # Mid-right
        [0.0, 0.85],       # Top
        [-0.2, 0.55],      # Mid-left
        [-0.35, 0.0]       # Close polygon
    ])
    return Polygon(vertices)


def load_sample_positions(sample_file: str, config_num: int) -> Optional[List[Tuple[float, float, float]]]:
    """Load positions from sample submission."""
    try:
        positions = []
        config_id = f"{config_num:03d}"
        
        with open(sample_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'].startswith(config_id + '_'):
                    x_str = row['x'].replace('s', '')
                    y_str = row['y'].replace('s', '')
                    deg_str = row['deg'].replace('s', '')
                    positions.append((float(x_str), float(y_str), float(deg_str)))
        
        if len(positions) == config_num:
            return positions
    except:
        pass
    return None


def fast_local_search(base_polygon: Polygon,
                     positions: List[Tuple[float, float, float]],
                     num_trees: int,
                     max_iterations: int = 1500) -> List[Tuple[float, float, float]]:
    """Fast local search with adaptive step size."""
    current_positions = [p for p in positions]
    best_positions = [p for p in positions]
    
    # Compute initial objective
    polygons = []
    for x, y, angle in best_positions:
        poly = base_polygon.transform(x, y, angle)
        polygons.append(poly)
    best_objective = compute_bounding_square(polygons) ** 2
    
    step_size = 0.003
    angle_step = 0.3
    no_improvement = 0
    max_no_improvement = 400
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try moving each tree
        for i in range(num_trees):
            x, y, angle = current_positions[i]
            
            # Try moves in 8 directions
            directions = [
                (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                (step_size, step_size), (-step_size, step_size),
                (step_size, -step_size), (-step_size, -step_size)
            ]
            
            for dx, dy in directions:
                new_x = x + dx
                new_y = y + dy
                
                # Check if valid
                new_poly = base_polygon.transform(new_x, new_y, angle)
                valid = True
                for j, (ox, oy, oangle) in enumerate(current_positions):
                    if i == j:
                        continue
                    other_poly = base_polygon.transform(ox, oy, oangle)
                    if separating_axis_theorem(new_poly, other_poly):
                        valid = False
                        break
                
                if valid:
                    old_pos = current_positions[i]
                    current_positions[i] = (new_x, new_y, angle)
                    
                    polygons = []
                    for px, py, pangle in current_positions:
                        poly = base_polygon.transform(px, py, pangle)
                        polygons.append(poly)
                    new_objective = compute_bounding_square(polygons) ** 2
                    
                    if new_objective < best_objective:
                        best_objective = new_objective
                        best_positions = [p for p in current_positions]
                        improved = True
                    else:
                        current_positions[i] = old_pos
            
            # Try small rotations
            for dangle in [-angle_step, angle_step]:
                new_angle = (angle + dangle) % 360
                
                new_poly = base_polygon.transform(x, y, new_angle)
                valid = True
                for j, (ox, oy, oangle) in enumerate(current_positions):
                    if i == j:
                        continue
                    other_poly = base_polygon.transform(ox, oy, oangle)
                    if separating_axis_theorem(new_poly, other_poly):
                        valid = False
                        break
                
                if valid:
                    old_pos = current_positions[i]
                    current_positions[i] = (x, y, new_angle)
                    
                    polygons = []
                    for px, py, pangle in current_positions:
                        poly = base_polygon.transform(px, py, pangle)
                        polygons.append(poly)
                    new_objective = compute_bounding_square(polygons) ** 2
                    
                    if new_objective < best_objective:
                        best_objective = new_objective
                        best_positions = [p for p in current_positions]
                        improved = True
                    else:
                        current_positions[i] = old_pos
        
        if improved:
            no_improvement = 0
            current_positions = [p for p in best_positions]
        else:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                step_size *= 0.85
                angle_step *= 0.85
                if step_size < 0.0005:
                    break
                no_improvement = 0
    
    return best_positions


def optimize_configuration(base_polygon: Polygon,
                           num_trees: int,
                           sample_file: str,
                           random_seed: Optional[int] = None) -> Tuple[List[Tuple[float, float, float]], float]:
    """Smart optimization based on configuration size."""
    best_positions = None
    best_objective = float('inf')
    
    # Load sample positions
    sample_positions = load_sample_positions(sample_file, num_trees)
    if not sample_positions:
        return (None, float('inf'))
    
    # Validate sample
    from optimizer import validate_solution
    if not validate_solution(base_polygon, sample_positions):
        return (sample_positions, float('inf'))
    
    # Compute sample objective
    polygons = []
    for x, y, angle in sample_positions:
        poly = base_polygon.transform(x, y, angle)
        polygons.append(poly)
    best_objective = compute_bounding_square(polygons) ** 2
    best_positions = [p for p in sample_positions]
    
    # Apply optimization based on configuration size
    seed = random_seed if random_seed is not None else 42
    
    if num_trees <= 20:
        # Small configs: aggressive optimization
        for run in range(2):
            sa = SimulatedAnnealing(
                base_polygon=base_polygon,
                num_trees=num_trees,
                initial_temp=0.3,
                final_temp=0.0001,
                cooling_rate=0.9995,
                max_iterations=15000,
                translation_step=0.002,
                rotation_step=0.4,
                random_seed=seed + run * 1000
            )
            
            initial = sample_positions if run == 0 else best_positions
            sa.initialize(initial)
            positions, objective = sa.optimize(verbose=False)
            
            if objective < best_objective:
                best_objective = objective
                best_positions = positions
        
        # Local search
        positions = fast_local_search(base_polygon, best_positions, num_trees, max_iterations=2000)
        polygons = []
        for x, y, angle in positions:
            poly = base_polygon.transform(x, y, angle)
            polygons.append(poly)
        objective = compute_bounding_square(polygons) ** 2
        if objective < best_objective:
            best_objective = objective
            best_positions = positions
    
    elif num_trees <= 50:
        # Medium configs: moderate optimization
        sa = SimulatedAnnealing(
            base_polygon=base_polygon,
            num_trees=num_trees,
            initial_temp=0.2,
            final_temp=0.0001,
            cooling_rate=0.9997,
            max_iterations=12000,
            translation_step=0.002,
            rotation_step=0.4,
            random_seed=seed
        )
        
        sa.initialize(sample_positions)
        positions, objective = sa.optimize(verbose=False)
        
        if objective < best_objective:
            best_objective = objective
            best_positions = positions
        
        # Quick local search
        positions = fast_local_search(base_polygon, best_positions, num_trees, max_iterations=1000)
        polygons = []
        for x, y, angle in positions:
            poly = base_polygon.transform(x, y, angle)
            polygons.append(poly)
        objective = compute_bounding_square(polygons) ** 2
        if objective < best_objective:
            best_objective = objective
            best_positions = positions
    
    elif num_trees <= 100:
        # Large configs: light optimization
        sa = SimulatedAnnealing(
            base_polygon=base_polygon,
            num_trees=num_trees,
            initial_temp=0.15,
            final_temp=0.0001,
            cooling_rate=0.9998,
            max_iterations=8000,
            translation_step=0.0015,
            rotation_step=0.3,
            random_seed=seed
        )
        
        sa.initialize(sample_positions)
        positions, objective = sa.optimize(verbose=False)
        
        if objective < best_objective:
            best_objective = objective
            best_positions = positions
    
    else:
        # Very large configs: minimal optimization
        sa = SimulatedAnnealing(
            base_polygon=base_polygon,
            num_trees=num_trees,
            initial_temp=0.1,
            final_temp=0.0001,
            cooling_rate=0.9999,
            max_iterations=4000,
            translation_step=0.001,
            rotation_step=0.2,
            random_seed=seed
        )
        
        sa.initialize(sample_positions)
        positions, objective = sa.optimize(verbose=False)
        
        if objective < best_objective:
            best_objective = objective
            best_positions = positions
    
    # Final validation
    if best_positions:
        if not validate_solution(base_polygon, best_positions):
            if validate_solution(base_polygon, sample_positions):
                best_positions = sample_positions
                polygons = []
                for x, y, angle in best_positions:
                    poly = base_polygon.transform(x, y, angle)
                    polygons.append(poly)
                best_objective = compute_bounding_square(polygons) ** 2
    
    return (best_positions, best_objective)


def generate_optimized_csv(
    output: str = 'submission.csv',
    sample_file: str = r'c:\Users\gjain\Downloads\sample_submission.csv',
    verbose: bool = True
):
    """Generate ultra smart optimized submission.csv."""
    random_seed = 42
    
    # Load tree shape
    if verbose:
        print("Loading tree shape...")
    base_polygon = load_tree_shape()
    
    if verbose:
        print(f"Tree polygon loaded with {len(base_polygon.vertices)} vertices")
        print("Starting optimization...\n")
    
    # Prepare output
    rows = []
    rows.append(['id', 'x', 'y', 'deg'])
    
    num_configs = 200
    total_start_time = time.time()
    total_score = 0.0
    
    for config_num in range(1, num_configs + 1):
        config_id = f"{config_num:03d}"
        num_trees = config_num
        
        if verbose and config_num % 25 == 0:
            print(f"Configuration {config_num}/200: {num_trees} trees")
        
        config_start_time = time.time()
        
        # Optimize this configuration
        seed = random_seed + config_num * 1000
        positions, objective = optimize_configuration(
            base_polygon=base_polygon,
            num_trees=num_trees,
            sample_file=sample_file,
            random_seed=seed
        )
        
        # Validate solution
        from optimizer import validate_solution
        is_valid = validate_solution(base_polygon, positions) if positions else False
        
        if not is_valid and positions:
            # Fallback to sample
            sample_positions = load_sample_positions(sample_file, config_num)
            if sample_positions and validate_solution(base_polygon, sample_positions):
                positions = sample_positions
                polygons = []
                for x, y, angle in positions:
                    poly = base_polygon.transform(x, y, angle)
                    polygons.append(poly)
                objective = compute_bounding_square(polygons) ** 2
                is_valid = True
        
        # Write positions
        if positions:
            for tree_idx, (x, y, angle) in enumerate(positions):
                tree_id = f"{config_id}_{tree_idx}"
                rows.append([
                    tree_id,
                    f"s{x:.6f}",
                    f"s{y:.6f}",
                    f"s{angle:.6f}"
                ])
        
        config_time = time.time() - config_start_time
        side_length = objective ** 0.5 if objective != float('inf') else 0.0
        score_contribution = objective / num_trees if objective != float('inf') else 0.0
        total_score += score_contribution
        
        if verbose and config_num % 25 == 0:
            print(f"  Time: {config_time:.2f}s, Side: {side_length:.6f}, Score: {score_contribution:.6f}, Valid: {is_valid}")
    
    # Write CSV
    if verbose:
        print(f"\nWriting {output}...")
    
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    total_time = time.time() - total_start_time
    
    if verbose:
        print(f"\n✓ Generated {output}")
        print(f"  Total rows: {len(rows)} (including header)")
        print(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"  Estimated total score: {total_score:.2f}")
        print(f"  Target: < 200 (current baseline: 1245.52)")
        print(f"  Improvement: {1245.52 - total_score:.2f} points")


if __name__ == "__main__":
    generate_optimized_csv(
        output="submission.csv",
        verbose=True
    )

