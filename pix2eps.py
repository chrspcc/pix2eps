import click
from PIL import Image
import os
from typing import List, Tuple, Set, Dict
from collections import deque

class EdgePathOptimizer:
    def __init__(self, image: Image.Image, mark_white: bool = True, 
                 add_border: bool = False, border_width: float = 1.0,
                 precision: int = 6, output_marks: bool = True,
                 mark_spacing: int = 1):
        # Convert image to RGB first (removing alpha channel) then to black and white
        if image.mode == 'LA':   # If this is a greyscale (luminance) image, convert to RGB
            image = image.convert('RGBA')

        if image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGBA', image.size, (255, 255, 255))
            # Remove alpha channel by compositing on white background
            image = Image.alpha_composite(background, image)

        image = image.convert('RGB')
            
        # Now convert to binary
        self.image = image.convert('1')
        self.width, self.height = self.image.size
        self.visited_edges = set()
        self.edge_graph = {}  # Maps vertices to their connected edges
        self.marked_pixels = []  # Store positions of pixels to be marked with X
        self.mark_white = mark_white  # Whether to mark white (True) or black (False) pixels
        self.add_border = add_border
        self.border_width = border_width
        self.precision = precision
        self.output_marks = output_marks
        self.mark_spacing = max(1, mark_spacing)  # Ensure spacing is at least 1
        
        # Calculate offsets to center the image in the border
        self.offset_x = round(self.border_width if self.add_border else 0, precision)
        self.offset_y = round(self.border_width if self.add_border else 0, precision)

    def round_coord(self, value: float) -> float:
        """Round coordinate to specified precision"""
        return round(value, self.precision)

    def round_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Round both coordinates of a point"""
        return (self.round_coord(point[0]), self.round_coord(point[1]))

    def get_pixel(self, x: int, y: int) -> bool:
        """Return True if pixel is black (foreground)"""
        # Adjust coordinates to account for border offset
        x_img = x - int(self.offset_x)
        y_img = y - int(self.offset_y)
        if 0 <= x_img < self.width and 0 <= y_img < self.height:
            return self.image.getpixel((x_img, y_img)) == 0  # 0 is black in binary mode
        return False

    def find_connected_regions(self) -> List[List[Tuple[int, int]]]:
        """Find connected regions of pixels to be marked"""
        regions = []
        visited = set()
        
        def should_mark_pixel(x: int, y: int) -> bool:
            """Check if pixel should be marked based on color"""
            pixel_is_white = not self.get_pixel(int(x + self.offset_x), int(y + self.offset_y))
            return pixel_is_white == self.mark_white
        
        def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
            """Get valid neighboring pixels"""
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
            return neighbors
        
        # Find connected regions using BFS
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in visited and should_mark_pixel(x, y):
                    # Start new region
                    region = []
                    queue = deque([(x, y)])
                    visited.add((x, y))
                    
                    while queue:
                        px, py = queue.popleft()
                        region.append((px, py))
                        
                        # Check neighbors
                        for nx, ny in get_neighbors(px, py):
                            if (nx, ny) not in visited and should_mark_pixel(nx, ny):
                                queue.append((nx, ny))
                                visited.add((nx, ny))
                    
                    if region:
                        regions.append(region)
        
        return regions

    def collect_marked_pixels(self):
        """Collect positions of pixels that should be marked with slashes"""
        self.marked_pixels = []
        regions = self.find_connected_regions()
        
        for region in regions:
            # Find region bounds
            min_x = min(x for x, _ in region)
            max_x = max(x for x, _ in region)
            min_y = min(y for _, y in region)
            max_y = max(y for _, y in region)
            
            # Create a set of region pixels for quick lookup
            region_pixels = set(region)
            
            # Always mark at least one pixel near the center of the region
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            
            # Find the closest pixel to center that's in the region
            closest_center = min(region, key=lambda p: (p[0] - center_x) ** 2 + (p[1] - center_y) ** 2)
            self.marked_pixels.append(closest_center)
            
            # Add additional marks based on spacing
            if self.mark_spacing > 1:
                for y in range(min_y, max_y + 1, self.mark_spacing):
                    for x in range(min_x, max_x + 1, self.mark_spacing):
                        if (x, y) in region_pixels and (x, y) != closest_center:
                            self.marked_pixels.append((x, y))

    def get_slash_mark_paths(self) -> List[List[Tuple[float, float]]]:
        """Generate slash paths for the center of marked pixels"""
        slash_paths = []
        slash_size = 0.3  # Size of the slash relative to pixel size (30% of pixel)
        
        # Group marks by rows to optimize output
        marks_by_row = {}
        for x, y in self.marked_pixels:
            marks_by_row.setdefault(y, []).append(x)
        
        # Process each row
        for y in sorted(marks_by_row.keys()):
            x_coords = sorted(marks_by_row[y])
            for x in x_coords:
                # Calculate center of pixel with offset
                center_x = self.round_coord(x + 0.5 + self.offset_x)
                center_y = self.round_coord(y + 0.5 + self.offset_y)
                
                # Calculate slash corners
                half_slash = self.round_coord(slash_size / 2)
                
                # Slash
                slash_paths.append([
                    self.round_point((center_x - half_slash, center_y + half_slash)),
                    self.round_point((center_x + half_slash, center_y - half_slash))
                ])
        
        return slash_paths

    def get_pixel_edges(self, x: int, y: int) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the edges (line segments) that make up a pixel's border"""
        edges = []
        if self.get_pixel(x, y):
            # Define the four corners of the pixel
            corners = [
                (x, y),        # top-left
                (x + 1, y),    # top-right
                (x + 1, y + 1),# bottom-right
                (x, y + 1)     # bottom-left
            ]
            
            # Create edges between corners
            for i in range(4):
                edge = (corners[i], corners[(i + 1) % 4])
                edges.append(edge)
        return edges

    def get_border_path(self) -> List[List[Tuple[float, float]]]:
        """Generate the border path around the entire image"""
        if not self.add_border:
            return []

        paths = []
        
        if self.border_width == 0:
            # For zero-width border, create a single path and check for edge conflicts
            border_path = [
                self.round_point((self.offset_x, self.offset_y)),
                self.round_point((self.width + self.offset_x, self.offset_y)),
                self.round_point((self.width + self.offset_x, self.height + self.offset_y)),
                self.round_point((self.offset_x, self.height + self.offset_y)),
                self.round_point((self.offset_x, self.offset_y))  # Close the path
            ]
            
            paths.append(border_path)
        else:
            # For non-zero width border, create two concentric rectangles
            outer_path = [
                (0, 0),
                self.round_point((self.width + 2 * self.border_width, 0)),
                self.round_point((self.width + 2 * self.border_width, self.height + 2 * self.border_width)),
                self.round_point((0, self.height + 2 * self.border_width)),
                (0, 0)  # Close the path
            ]
            
            inner_path = [
                self.round_point((self.offset_x, self.offset_y)),
                self.round_point((self.width + self.offset_x, self.offset_y)),
                self.round_point((self.width + self.offset_x, self.height + self.offset_y)),
                self.round_point((self.offset_x, self.height + self.offset_y)),
                self.round_point((self.offset_x, self.offset_y))  # Close the path
            ]
            
            paths.extend([outer_path, inner_path])
            
        return paths

    def should_include_edge(self, edge: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """Determine if an edge should be included in the final path"""
        (x1, y1), (x2, y2) = edge
        
        # For zero-width border, check if this edge is part of the border
        if self.add_border and self.border_width == 0:
            # Adjust coordinates for border check
            x1_adj = self.round_coord(x1 - self.offset_x)
            y1_adj = self.round_coord(y1 - self.offset_y)
            x2_adj = self.round_coord(x2 - self.offset_x)
            y2_adj = self.round_coord(y2 - self.offset_y)
            
            # If either point is on the border
            if (x1_adj in (0, self.width) or y1_adj in (0, self.height) or 
                x2_adj in (0, self.width) or y2_adj in (0, self.height)):
                # Skip this edge if it's exactly on the border
                if ((x1_adj == x2_adj == 0) or (x1_adj == x2_adj == self.width) or
                    (y1_adj == y2_adj == 0) or (y1_adj == y2_adj == self.height)):
                    return False
        
        # Find the two pixels that share this edge
        if x1 == x2:  # Vertical edge
            left_x = int(x1 - 1)
            right_x = int(x1)
            y = int(min(y1, y2))
            return self.get_pixel(left_x, y) != self.get_pixel(right_x, y)
        else:  # Horizontal edge
            bottom_y = int(y1 - 1)
            top_y = int(y1)
            x = int(min(x1, x2))
            return self.get_pixel(x, bottom_y) != self.get_pixel(x, top_y)

    def build_edge_graph(self):
        """Build a graph of connected edges"""
        self.edge_graph = {}
        
        # Collect all valid edges, accounting for border offset
        for y in range(int(self.offset_y), int(self.height + self.offset_y + 1)):
            for x in range(int(self.offset_x), int(self.width + self.offset_x + 1)):
                # Check horizontal edge
                if x < self.width + self.offset_x:
                    p1 = self.round_point((x, y))
                    p2 = self.round_point((x + 1, y))
                    edge = (p1, p2)
                    if self.should_include_edge(edge):
                        self.edge_graph.setdefault(p1, set()).add(p2)
                        self.edge_graph.setdefault(p2, set()).add(p1)
                
                # Check vertical edge
                if y < self.height + self.offset_y:
                    p1 = self.round_point((x, y))
                    p2 = self.round_point((x, y + 1))
                    edge = (p1, p2)
                    if self.should_include_edge(edge):
                        self.edge_graph.setdefault(p1, set()).add(p2)
                        self.edge_graph.setdefault(p2, set()).add(p1)

    def find_next_start_vertex(self) -> Tuple[float, float] | None:
        """Find the next unvisited vertex that has edges"""
        for vertex in self.edge_graph:
            neighbors = self.edge_graph[vertex]
            if neighbors and any((vertex, n) not in self.visited_edges and (n, vertex) not in self.visited_edges for n in neighbors):
                return vertex
        return None

    def optimize_paths(self) -> List[List[Tuple[float, float]]]:
        """Generate optimized cutting paths following pixel edges"""
        self.build_edge_graph()
        
        # Only collect marked pixels if marks are enabled
        if self.output_marks:
            self.collect_marked_pixels()
        
        # Get all types of paths
        edge_paths = self.optimize_edge_paths()
        x_paths = self.get_slash_mark_paths() if self.output_marks else []
        border_paths = self.get_border_path()
        
        # Combine all paths
        return border_paths + edge_paths + x_paths

    def optimize_edge_paths(self) -> List[List[Tuple[float, float]]]:
        """Generate optimized paths for the edges"""
        paths = []
        
        while True:
            start = self.find_next_start_vertex()
            if start is None:
                break

            current_path = [start]
            current = start
            
            while True:
                # Find an unvisited edge from current vertex
                neighbors = self.edge_graph[current]
                next_vertex = None
                
                for neighbor in neighbors:
                    if (current, neighbor) not in self.visited_edges and (neighbor, current) not in self.visited_edges:
                        next_vertex = neighbor
                        self.visited_edges.add((current, neighbor))
                        break
                
                if next_vertex is None:
                    break
                
                current = next_vertex
                current_path.append(current)
                
                # If we've returned to start, close the path
                if current == start and len(current_path) > 2:
                    break
            
            if len(current_path) > 1:  # Only add paths with actual edges
                # Optimize the path by combining collinear segments
                optimized_path = self.optimize_collinear_segments(current_path)
                paths.append(optimized_path)
        
        return paths

    def optimize_collinear_segments(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Optimize a path by combining collinear segments"""
        if len(path) <= 2:
            return path
        
        optimized = [path[0]]  # Start with first point
        i = 1
        
        while i < len(path):
            # Get current direction
            prev = path[i-1]
            current = path[i]
            
            # Check if horizontal or vertical line
            is_horizontal = abs(current[1] - prev[1]) < 1e-10
            is_vertical = abs(current[0] - prev[0]) < 1e-10
            
            if is_horizontal or is_vertical:
                # Look ahead for more points in same direction
                j = i + 1
                while j < len(path):
                    next_point = path[j]
                    
                    # Check if next segment is in same direction
                    if is_horizontal:
                        if abs(next_point[1] - current[1]) < 1e-10:
                            j += 1
                            continue
                    elif is_vertical:
                        if abs(next_point[0] - current[0]) < 1e-10:
                            j += 1
                            continue
                    break
                
                # Add the last point of the collinear sequence
                optimized.append(path[j-1])
                i = j
            else:
                # Not collinear, add the point as is
                optimized.append(current)
                i += 1
        
        return optimized

def generate_eps(paths: List[List[Tuple[float, float]]], 
                output_file: str,
                border_width: float,
                scale: float = 72.0,
                precision: int = 6):  # Default scale is 72 units (points) per pixel
    """Generate EPS file from optimized edge paths"""
    with open(output_file, 'w') as f:
        # Calculate scaled dimensions
        scaled_width = round((width + 2 * border_width) * scale, precision)
        scaled_height = round((height + 2 * border_width) * scale, precision)
        
        # EPS header with scaled bounding box
        f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
        f.write("%%BoundingBox: 0 0 {0} {1}\n".format(
            int(scaled_width),
            int(scaled_height)
        ))
        f.write("%%EndComments\n")
        
        # PostScript Prolog
        f.write("%%BeginProlog\n")
        f.write("/m systemdict /moveto get def\n")
        f.write("/l systemdict /lineto get def\n")
        f.write("%%EndProlog\n")
        
        # Setup section for page-level commands
        f.write("%%BeginSetup\n")
        f.write("{0} setlinewidth\n".format(round(0.01 * scale, precision)))  # Thin lines for cutting
        f.write("%%EndSetup\n")
        
        # Draw paths with scaled coordinates
        for path in paths:
            if not path:
                continue
                
            f.write("newpath\n")
            # Move to first scaled point
            f.write("{0} {1} m\n".format(
                round(path[0][0] * scale, precision),
                round(scaled_height - (path[0][1] * scale), precision)  # Flip Y coordinate
            ))
            
            # Draw lines to subsequent scaled points
            for point in path[1:]:
                f.write("{0} {1} l\n".format(
                    round(point[0] * scale, precision),
                    round(scaled_height - (point[1] * scale), precision)  # Flip Y coordinate
                ))
            
            f.write("stroke\n")

        f.write("showpage\n")

@click.command()
@click.argument('input_files', nargs=-1, required=True, 
                type=click.Path(exists=True))
@click.option('--output', '-o', help='Output EPS file (optional)')
@click.option('--scale', '-s', default=72.0, type=float,
              help='Scale factor in points per pixel (default: 72.0 for 1 inch per pixel)')
@click.option('--mark-white/--mark-black', default=True,
              help='Choose whether to mark white (default) or black pixels with slashes')
@click.option('--marks/--no-marks', default=True,
              help='Enable or disable slash marks output (default: enabled with spacing of 50 pixels)')
@click.option('--mark-spacing', '-m', default=50, type=int,
              help='Spacing between marks in pixels (default: 50, higher values reduce mark density. A value of 1 would put a slash in each pixel to be removed.)')
@click.option('--border/--no-border', default=False,
              help='Add a border around the entire image')
@click.option('--border-width', '-w', default=1.0, type=float,
              help='Width of the border in pixel units (default: 1.0)')
@click.option('--precision', '-p', default=6, type=int,
              help='Number of decimal places for coordinate rounding (default: 6)')
def convert(input_files, output, scale, mark_white, marks, mark_spacing,
           border, border_width, precision):
    """Convert bitmap/pixel image(s) to EPS for vinyl cutting.
    
    Scale factor determines the size of the output in PostScript points per pixel.
    Common values:
    - 72.0: 1 inch per pixel (default)
    - 28.35: 1 cm per pixel
    """
    for input_file in input_files:
        # Generate output filename if not specified
        if output and len(input_files) == 1:
            output_file = output
        else:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.eps"

        # Process the image
        try:
            with Image.open(input_file) as img:
                global width, height
                width, height = img.size
                
                click.echo(f"Processing {input_file}...")
                optimizer = EdgePathOptimizer(img, mark_white, border, border_width, 
                                           precision, marks, mark_spacing)
                paths = optimizer.optimize_paths()
                
                # Pass the actual border width (0 if no border)
                actual_border_width = border_width if border else 0
                generate_eps(paths, output_file, actual_border_width, scale, precision)
                click.echo(f"Created {output_file}")
                
        except Exception as e:
            click.echo(f"Error processing {input_file}: {str(e)}", err=True)

if __name__ == '__main__':
    convert() 