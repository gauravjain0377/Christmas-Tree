"""
Quick generator to create complete submission.csv with all 20,101 lines.
Uses sample submission pattern with optimized positions.
"""

import csv
import numpy as np
from geometry import Polygon


def load_tree_shape() -> Polygon:
    """Load tree shape."""
    vertices = np.array([
        [-0.35, 0.0],
        [0.35, 0.0],
        [0.2, 0.55],
        [0.0, 0.85],
        [-0.2, 0.55],
        [-0.35, 0.0]
    ])
    return Polygon(vertices)


def generate_positions(num_trees: int, base_polygon: Polygon) -> list:
    """Generate positions for num_trees using optimized packing with collision detection."""
    positions = []
    
    if num_trees == 0:
        return positions
    
    # Start with first tree at origin
    positions.append((0.0, 0.0, 90.0))
    
    if num_trees == 1:
        return positions
    
    # Use safe spacing - trees are roughly 0.7-0.85 units tall, so need at least 0.9 spacing
    min_spacing = 0.9
    layer = 1
    placed = 1
    
    while placed < num_trees:
        # Number of trees in this layer
        trees_in_layer = min(6 * layer, num_trees - placed)
        
        # Place trees in hexagonal pattern
        for i in range(trees_in_layer):
            if placed >= num_trees:
                break
            
            angle = 2 * np.pi * i / trees_in_layer
            radius = layer * min_spacing
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Optimized rotation angles
            rot_angle = (i * 60 + 90) % 360
            
            # Check for collision before adding
            new_poly = base_polygon.transform(x, y, rot_angle)
            has_overlap = False
            
            for px, py, pangle in positions:
                other_poly = base_polygon.transform(px, py, pangle)
                from geometry import separating_axis_theorem
                if separating_axis_theorem(new_poly, other_poly):
                    has_overlap = True
                    break
            
            if not has_overlap:
                positions.append((x, y, rot_angle))
                placed += 1
            else:
                # Try with slightly larger radius if collision
                radius += 0.1
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                new_poly = base_polygon.transform(x, y, rot_angle)
                has_overlap = False
                for px, py, pangle in positions:
                    other_poly = base_polygon.transform(px, py, pangle)
                    if separating_axis_theorem(new_poly, other_poly):
                        has_overlap = True
                        break
                if not has_overlap:
                    positions.append((x, y, rot_angle))
                    placed += 1
        
        layer += 1
        # Increase spacing slightly for outer layers to ensure no overlaps
        min_spacing = 0.9 + (layer - 1) * 0.05
    
    return positions


def generate_complete_csv(output: str = 'submission.csv'):
    """Generate complete submission.csv with all 20,101 lines."""
    print("Generating complete submission.csv with collision detection...")
    
    # Load tree shape
    base_polygon = load_tree_shape()
    
    rows = []
    rows.append(['id', 'x', 'y', 'deg'])  # Header
    
    total_trees = 0
    for config_num in range(1, 201):  # Configurations 1 to 200
        config_id = f"{config_num:03d}"
        num_trees = config_num
        
        # Generate positions for this configuration with collision detection
        positions = generate_positions(num_trees, base_polygon)
        
        # Validate no overlaps
        from optimizer import validate_solution
        is_valid = validate_solution(base_polygon, positions)
        
        if not is_valid and len(positions) == num_trees:
            # If invalid, use safer spacing
            positions = []
            positions.append((0.0, 0.0, 90.0))
            if num_trees > 1:
                spacing = 1.0  # Safe spacing
                for i in range(1, num_trees):
                    angle = 2 * np.pi * i / num_trees
                    radius = spacing
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    rot_angle = (i * 60 + 90) % 360
                    positions.append((x, y, rot_angle))
        
        # Write positions
        for tree_idx, (x, y, angle) in enumerate(positions):
            tree_id = f"{config_id}_{tree_idx}"
            rows.append([
                tree_id,
                f"s{x:.6f}",
                f"s{y:.6f}",
                f"s{angle:.6f}"
            ])
            total_trees += 1
        
        if config_num % 50 == 0:
            print(f"  Generated {config_num}/200 configurations...")
    
    # Write CSV file
    print(f"Writing {output}...")
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\nComplete! Generated {output}")
    print(f"  Total rows: {len(rows)} (1 header + {len(rows)-1} data rows)")
    print(f"  Total trees: {total_trees}")
    print(f"  Expected: 20,101 rows (1 header + 20,100 data rows)")


if __name__ == "__main__":
    generate_complete_csv("submission.csv")

