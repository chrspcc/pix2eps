from PIL import Image
import numpy as np
import os
from pix2eps import EdgePathOptimizer

def create_test_image(pattern, name):
    """Create a 3x3 test image from a binary pattern"""
    img = Image.fromarray((np.array(pattern) * 255).astype('uint8'))
    img.save(name)
    return img

def print_paths(paths, pattern, mark_white=True, border=False, border_width=0.0):
    """Pretty print the edge paths and X-marks for visualization"""
    print("\nPaths found:")
    for i, path in enumerate(paths):
        print(f"\nPath {i + 1}:")
        print("Vertices:", ' -> '.join(f"({x:.1f}, {y:.1f})" for x, y in path))
    
    # Calculate grid size based on border width
    padding = int(border_width * 2) if border else 0
    grid_size = 8 + padding
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Calculate offset for the pattern due to border
    offset = int(border_width) if border else 0
    
    # Draw border if enabled
    if border:
        # Draw outer border
        if border_width > 0:
            # Horizontal lines
            for x in range(grid_size):
                grid[0][x] = '-'
                grid[grid_size-1][x] = '-'
            # Vertical lines
            for y in range(grid_size):
                grid[y][0] = '|'
                grid[y][grid_size-1] = '|'
            # Corners
            grid[0][0] = '+'
            grid[0][grid_size-1] = '+'
            grid[grid_size-1][0] = '+'
            grid[grid_size-1][grid_size-1] = '+'
        
        # Draw inner border
        # Horizontal lines
        for x in range(2*offset, grid_size-2*offset):
            grid[2*offset][x] = '-'
            grid[grid_size-2*offset-1][x] = '-'
        # Vertical lines
        for y in range(2*offset, grid_size-2*offset):
            grid[y][2*offset] = '|'
            grid[y][grid_size-2*offset-1] = '|'
        # Corners
        grid[2*offset][2*offset] = '+'
        grid[2*offset][grid_size-2*offset-1] = '+'
        grid[grid_size-2*offset-1][2*offset] = '+'
        grid[grid_size-2*offset-1][grid_size-2*offset-1] = '+'
    
    # Mark pixels with X
    for y in range(len(pattern)):
        for x in range(len(pattern[y])):
            pixel_is_white = pattern[y][x] == 0
            if pixel_is_white == mark_white:
                grid_x = int(x * 2 + 1 + 2*offset)
                grid_y = int(y * 2 + 1 + 2*offset)
                grid[grid_y][grid_x] = 'X'
    
    # Add the paths
    for path in paths:
        # Skip border paths in visualization as we handle them separately
        if border and any(x < 0 or y < 0 for x, y in path):
            continue
            
        # Mark vertices
        for x, y in path:
            grid_x = int(x * 2 + 2*offset)
            grid_y = int(y * 2 + 2*offset)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                if grid[grid_y][grid_x] != '+':  # Don't overwrite border corners
                    grid[grid_y][grid_x] = '+'
    
    print("\nPath visualization (+ = vertex, - or | = edge, X = removal mark):")
    for row in grid:
        print(''.join(row))

# Test patterns (0 = white, 1 = black)
test_patterns = {
    'single_pixel': [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    'horizontal_line': [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    'l_shape': [
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
}

# Create test directory if it doesn't exist
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# Process each test pattern
print("Testing edge path optimization with different patterns (inverted colors):\n")
for pattern_name, pattern in test_patterns.items():
    print(f"\nTesting pattern: {pattern_name}")
    print("Input pattern (# = white, . = black):")
    for row in pattern:
        print(" ".join("." if pixel else "#" for pixel in row))
    
    # Create and save test image
    img_path = f"test_images/{pattern_name}.png"
    img = create_test_image(pattern, img_path)
    
    # Test different configurations
    test_configs = [
        {"desc": "Default (no border, mark white pixels):", "border": False, "width": 1.0, "mark_white": True},
        {"desc": "With border (default width=1):", "border": True, "width": 1.0, "mark_white": True},
        {"desc": "With border width 0:", "border": True, "width": 0.0, "mark_white": True},
        {"desc": "With border width 2:", "border": True, "width": 2.0, "mark_white": True},
        {"desc": "Default border, mark black pixels:", "border": True, "width": 1.0, "mark_white": False}
    ]
    
    for config in test_configs:
        print(f"\n{config['desc']}")
        optimizer = EdgePathOptimizer(img, config["mark_white"], config["border"], config["width"])
        paths = optimizer.optimize_paths()
        print_paths(paths, pattern, config["mark_white"], config["border"], config["width"])

print("\nTest images have been saved in the 'test_images' directory.")
print("You can convert them to EPS using:")
print("python pix2eps.py test_images/*.png  # Default (no border)")
print("python pix2eps.py test_images/*.png --border  # Add border with default width (1)")
print("python pix2eps.py test_images/*.png --border --border-width 0  # Add zero-width border")
print("python pix2eps.py test_images/*.png --border --border-width 2  # Add border with width 2")
print("python pix2eps.py test_images/*.png --border --mark-black  # Default border and mark black pixels") 