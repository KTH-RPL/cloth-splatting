import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circular_trajectory0(start_pos, radius, angle, velocity, time_step=0.1):
    """
    Generate points along a circular trajectory in 3D based on velocity and time step.
    
    :param start_pos: Tuple of (x, y, z) as starting position of the center of the circle.
    :param radius: The radius of the circle.
    :param angle: The total angle in radians to sweep the circle.
    :param velocity: Desired velocity along the circle.
    :param time_step: Time interval between steps.
    :return: List of positions (x, y, z) in the circle.
    """
    arc_length = radius * angle  # Length of the circular arc
    step_length = velocity * time_step  # Distance covered in each step
    n_steps = int(np.ceil(arc_length / step_length))  # Total number of steps
    angles = np.linspace(0, angle, n_steps)
    
    return [(start_pos[0] + radius * np.cos(a), start_pos[1], start_pos[2] + radius * np.sin(a)) for a in angles]

def generate_circular_trajectory(start_pos, radius, angle, velocity, tilt, time_step=0.1):
    """
    Generate points along a tilted circular trajectory in 3D based on velocity, tilt, and time step.
    
    :param start_pos: Tuple of (x, y, z) as the starting position of the center of the circle.
    :param radius: The radius of the circle.
    :param angle: The total angle in radians to sweep the circle.
    :param velocity: Desired velocity along the circle.
    :param tilt: Tilt angle in radians from the original plane.
    :param time_step: Time interval between steps.
    :return: List of positions (x, y, z) in the tilted circle.
    """
    arc_length = radius * angle  # Length of the circular arc
    step_length = velocity * time_step  # Distance covered in each step
    n_steps = int(np.ceil(arc_length / step_length))  # Total number of steps
    angles = np.linspace(0, angle, n_steps)
    
    # Rotation matrix about the x-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    
    points = []
    for a in angles:
        # Point in the original x-z plane
        point = np.array([start_pos[0] + radius * np.cos(a), start_pos[1], start_pos[2] + radius * np.sin(a)])
        # Apply rotation
        rotated_point = Rx @ (point - np.array(start_pos)) + np.array(start_pos)
        points.append(tuple(rotated_point))
    
    return points

def compute_actions_from_trajectory(trajectory):
    """
    Compute the displacements required to follow a 3D trajectory.
    
    :param trajectory: List of positions (x, y, z) defining the trajectory.
    :return: List of delta actions (dx, dy, dz) to follow the trajectory.
    """
    actions = []
    for i in range(1, len(trajectory)):
        current_pos = trajectory[i - 1]
        next_pos = trajectory[i]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        dz = next_pos[2] - current_pos[2]
        actions.append((dx, dy, dz))
    return actions

def visualize_trajectory_and_actions(trajectory, actions):
    """
    Plot the 3D trajectory and actions for visual analysis.
    
    :param trajectory: List of positions (x, y, z) defining the trajectory.
    :param actions: List of delta actions (dx, dy, dz) to follow the trajectory.
    """
    # Extract coordinates for plotting
    xs, ys, zs = zip(*trajectory)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, 'bo-', label='Trajectory Points')  # Trajectory points
    
    # Add action vectors to the plot
    for (x, y, z), (dx, dy, dz) in zip(trajectory[:-1], actions):
        ax.quiver(x, y, z, dx, dy, dz, color='r', length=np.linalg.norm([dx, dy, dz]), normalize=True)
    
    ax.set_title('3D Circular Trajectory with Action Segments')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.show()
    
    
def visualize_multiple_trajectories(trajectories_and_actions):
    """
    Plot multiple 3D trajectories and actions for visual analysis.
    
    :param trajectories_and_actions: List of tuples, each containing a trajectory and its actions.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Different colors for different trajectories

    for index, (trajectory, actions) in enumerate(trajectories_and_actions):
        color = colors[index % len(colors)]  # Cycle through colors
        xs, ys, zs = zip(*trajectory)
        ax.plot(xs, ys, zs, marker='o', linestyle='-', color=color, label=f'Trajectory {index + 1}')
        
        if actions:  # Plot actions if available
            for (x, y, z), (dx, dy, dz) in zip(trajectory[:-1], actions):
                ax.quiver(x, y, z, dx, dy, dz, color=color, length=np.linalg.norm([dx, dy, dz]) * 0.1, normalize=True)
                
    # set axis limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    
    ax.set_title('Multiple 3D Trajectories with Action Segments')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Z coordinate')
    ax.grid(True)
    ax.legend()
    plt.show()
    

def rotate_point_around_axis(pt, axis, theta, origin):
    """
    Rotate a single point around an axis by angle theta (in radians), where the axis passes through the origin.
    """
    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    return np.dot(rotation_matrix, pt - origin) + origin

def bezier_quadratic(P0, P1, P2, t):
    """
    Calculate position on a quadratic Bézier curve at parameter t.
    """
    return (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2

def generate_bezier_trajectory(start, end, height, tilt, velocity, dt=0.1):
    """
    Generate a trajectory using a quadratic Bézier curve with the control point rotated around the start-end axis.
    """
    # Calculate the unrotated control point
    peak = (start + end) / 2 + np.array([0, 0, height])
    control = 2 * peak - 0.5 * (start + end)  # Calculate the control point
    
    # Rotate only the control point around the axis defined by the start and end points
    axis = end - start
    rotated_control = rotate_point_around_axis(control, axis, tilt, (start + end) / 2)
    
    # Sample curve based on velocity
    length = np.linalg.norm(end - start)
    num_points = int(length / velocity / dt)
    t_values = np.linspace(0, 1, num_points)
    trajectory = [bezier_quadratic(start, rotated_control, end, t) for t in t_values]
    
    return np.array(trajectory)

def get_action_traj(pick, place, height, tilt, velocity, dt=0.01, sim_data=False):
    if sim_data:
        pick = pick[[0, 2, 1]]
        place = place[[0, 2, 1]]
    trajectory = generate_bezier_trajectory(pick, place, height, tilt, velocity, dt=dt)
    if sim_data:
        trajectory[:, [1, 2]] = trajectory[:, [2, 1]]
    actions = compute_actions_from_trajectory(trajectory)
    
    return np.asarray(trajectory), np.asarray(actions)
    

if __name__ == "__main__":
    
    # ###################################### 
    
    # # Parameters for the 3D trajectory
    # start_pos = (0, 0, 0)  # Adjust as needed
    # radius = 1.0  # Adjust radius as needed
    # angle = np.pi * 0.9  # Total angle of circular path
    # velocity = 1.  # Desired velocity in units per second
    # time_step = 0.1  # Time interval between steps in seconds
    # tilt = 0

    # # Generate and visualize the trajectory
    # trajectory = generate_circular_trajectory(start_pos, radius, angle, velocity, tilt, time_step)
    # actions = compute_actions_from_trajectory(trajectory)
    # visualize_trajectory_and_actions(trajectory, actions)
    
    # # Example trajectories and actions for visualization
    # params = [
    #     ((0, 0, 0), 1.0, np.pi * 0.9, 0.05, np.radians(30), 0.1),
    #     ((0, 0, 0), 1.5, np.pi *0.9, 0.07, np.radians(45), 0.1),
    #     ((0, 0, 0), 1.2, np.pi, 0.9, np.radians(60), 0.1),
    #     ((0, 0, 0), 1.0, np.pi * 0.9, 0.05, np.radians(15), 0.1)
    # ]

    # trajectories_and_actions = []
    # for start_pos, radius, angle, velocity, tilt, time_step in params:
    #     trajectory = generate_circular_trajectory(start_pos, radius, angle, velocity, tilt, time_step)
    #     actions = compute_actions_from_trajectory(trajectory)
    #     trajectories_and_actions.append((trajectory, actions))

    # visualize_multiple_trajectories(trajectories_and_actions)
    
    ######################################
    
    # Example Usage
    # start = np.array([0, 0, 0])
    # end = np.array([1, 0, 0])
    # height = 0.5
    # tilt = np.pi / 4  # 45 degrees
    # velocity = 1.0  # Units per second
    # trajectory = generate_bezier_trajectory(start, end, height, tilt, velocity, dt=0.1)
    # actions = compute_actions_from_trajectory(trajectory)
    # visualize_trajectory_and_actions(trajectory, actions)
    # Plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Bézier Trajectory')
    # ax.legend()
    # plt.show()
    
    
    start = np.array([0, 0, 0])
    end = np.array([1, 0, 0])
    params = [
        (start, end, 1.0,  np.radians(30), 1.),
        (start, end, 0.5,  np.radians(45), 1.),
        (start, end, 1.2,  np.radians(60), 1.),
        (start, end, 0.2,  np.radians(15), 1.)
    ]
    # tilt changes
    params = [
        (start, end, 0.5,  np.radians(0), 1.),
        (start, end, 0.5,  np.radians(30), 1.),
        (start, end, 0.5,  np.radians(45), 1.),
        (start, end, 0.5,  np.radians(60), 1.),
    ]
    
    # hegiht changes
    params = [
        (start, end, 0.2,  np.radians(0), 1.),
        (start, end, 0.5,  np.radians(0), 1.),
        (start, end, 0.9,  np.radians(0), 1.),
        (start, end, 1.5,  np.radians(0), 1.),
    ]
    
    # velocity changes
    params = [
        (start, end, 0.5,  np.radians(0), 1.),
        (start, end, 0.5,  np.radians(0), 2.),
        (start, end, 0.5,  np.radians(0), 0.5),
        (start, end, 0.5,  np.radians(0), 0.1),
    ]
    
    # start end changes
    params = [
        (start, end, 0.5,  np.radians(0), 1.),
        (start, np.array([1, 0.5, 0]), 0.5,  np.radians(0), 2.),
        (start, np.array([1, -0.2, 0]), 0.5,  np.radians(0), 0.5),
        (start, np.array([1, 1., 0]), 0.5,  np.radians(0), 0.1),
    ]
    
    trajectories_and_actions = []
    for start, end, height, tilt, velocity in params:
        trajectory = generate_bezier_trajectory(start, end, height, tilt, velocity, dt=0.1)
        actions = compute_actions_from_trajectory(trajectory)
        trajectories_and_actions.append((trajectory, actions))
        trajectories_and_actions.append((trajectory, actions))
        
    visualize_multiple_trajectories(trajectories_and_actions)
    
    
    


    print()