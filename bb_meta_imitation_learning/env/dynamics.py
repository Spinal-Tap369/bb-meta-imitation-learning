# env\dynamics.py

import numpy
from numba import njit


PI_4 = 0.7853981
PI_2 = 1.5707963
PI = 3.1415926
t_PI = 6.2831852
PI2d = 57.29578

OFFSET_10 = numpy.asarray([0.5, 0.5], dtype=numpy.float32)
OFFSET_01 = numpy.asarray([-0.5, 0.5], dtype=numpy.float32)
OFFSET_m0 = numpy.asarray([-0.5, -0.5], dtype=numpy.float32)
OFFSET_0m = numpy.asarray([0.5, -0.5], dtype=numpy.float32)

# Helper: Nearest point on a line segment.
@njit(cache=True)
def nearest_point(pos, line_1, line_2):
    unit_ori = line_2 - line_1
    edge_norm = numpy.sqrt(numpy.sum(unit_ori * unit_ori))
    unit_ori /= max(1.0e-6, edge_norm)
    dist_1 = numpy.sum((pos - line_1) * unit_ori)
    if dist_1 > edge_norm:
        return numpy.sqrt(numpy.sum((pos - line_2) ** 2)), numpy.copy(line_2)
    elif dist_1 < 0:
        return numpy.sqrt(numpy.sum((pos - line_1) ** 2)), numpy.copy(line_1)
    else:
        line_p = line_1 + dist_1 * unit_ori
        return numpy.sqrt(numpy.sum((pos - line_p) ** 2)), numpy.copy(line_p)

# Basic vector movement (without collision)
@njit(cache=True)
def vector_move(ori, turn_rate, walk_speed, dt):
    # Increase turning and movement speed by applying scaling factors.
    turn_factor = numpy.float32(5.0)   
    move_factor = numpy.float32(6.0)   
    ori += turn_rate * dt * turn_factor
    dx = walk_speed * numpy.cos(ori) * dt * move_factor
    dy = walk_speed * numpy.sin(ori) * dt * move_factor
    return ori, numpy.array([dx, dy], dtype=numpy.float32)

# A helper clamp() function.
@njit(cache=True)
def clamp(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val

# collision resolution using circle-rectangle collision.
@njit(cache=True)
def resolve_collisions(pos, cell_walls, cell_size, agent_radius, margin):
    """
    For a given agent position (pos), check the current cell and neighbors.
    For each wall cell, compute the minimal translation vector required to
    push the agent's circular body (of radius = agent_radius + margin) out.
    """
    cell_i = int(pos[0] / cell_size)
    cell_j = int(pos[1] / cell_size)
    translation = numpy.zeros(2, dtype=numpy.float32)
    collision_found = False
    # Loop over the current cell and its 8 neighbors.
    for i in range(cell_i - 1, cell_i + 2):
        for j in range(cell_j - 1, cell_j + 2):
            # Skip out-of-bound indices.
            if i < 0 or i >= cell_walls.shape[0] or j < 0 or j >= cell_walls.shape[1]:
                continue
            if cell_walls[i, j] > 0:
                # Compute the cell's bounding box.
                left = i * cell_size
                right = (i + 1) * cell_size
                bottom = j * cell_size
                top = (j + 1) * cell_size
                # Clamp the agent's center to the cell's box.
                cx = clamp(pos[0], left, right)
                cy = clamp(pos[1], bottom, top)
                diff = numpy.array([pos[0] - cx, pos[1] - cy], dtype=numpy.float32)
                dist = numpy.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
                threshold = agent_radius + margin
                if dist < threshold:
                    collision_found = True
                    penetration = threshold - dist
                    if dist > 0:
                        translation += (diff / dist) * penetration
                    else:
                        # If exactly overlapping, choose an arbitrary direction.
                        translation += numpy.array([penetration, 0], dtype=numpy.float32)
    return translation, collision_found

# collision resolution combined with movement.
@njit(cache=True)
def vector_move_with_collision(ori, pos, turn_rate, walk_speed, dt, cell_walls, cell_size, col_dist):
    """
    Moves the agent using vector_move() and then resolves collisions.
    The agent is modeled as a circle with radius = agent_radius.
    If any overlap with wall cells is detected, a minimal translation
    is computed (via resolve_collisions()) and applied.
    
    Parameters:
      - ori: current orientation (float32)
      - pos: current position (a sequence of 2 numbers; will be converted to float32 array)
      - turn_rate, walk_speed: controls
      - dt: timestep
      - cell_walls: 2D array (nonzero means wall)
      - cell_size: physical size of one cell
      - col_dist: additional collision margin
      
    Returns:
      - new orientation (ori)
      - new position (float32 array of length 2)
      - collision_occurred: True if any collision was resolved.
    """
    # Ensure pos is a numpy array of float32.
    pos = numpy.asarray(pos, dtype=numpy.float32)
    
    # Define the agent radius (tweak as needed) and margin.
    agent_radius = numpy.float32(0.22 * cell_size)
    margin = numpy.float32(col_dist)
    
    # Compute the intended movement.
    ori, offset = vector_move(ori, turn_rate, walk_speed, dt)
    new_pos = pos + offset  # intended new position
    
    collision_occurred = False
    max_iterations = 10
    for _ in range(max_iterations):
        translation, collided = resolve_collisions(new_pos, cell_walls, cell_size, agent_radius, margin)
        if collided:
            collision_occurred = True
            new_pos = new_pos + translation
        else:
            break
    return ori, new_pos, collision_occurred