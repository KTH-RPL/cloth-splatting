import numpy as np
import pyflex

def infer_dt_from_displacement():
    """
    Infers the internal simulation time step (dt) by giving a particle an initial velocity,
    advancing the simulation one step, and measuring the displacement to calculate dt.
    
    Returns:
    float: The inferred simulation time step (dt) in seconds.
    """
    pyflex.init(0, 1, 480, 480, 0)

    # Define initial position and velocity
    initial_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    initial_velocity = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 1 unit per second along the x-axis
    particle_radius = 0.05
    particle_mass = 1.0  # Assuming a mass of 1 kg for simplicity

    # Add a single particle
    pyflex.add_sphere(particle_radius, initial_position, [0, 1, 0, 0])

    # Set initial velocity
    pyflex.set_velocities(np.array([initial_velocity]))

    # Step the simulation
    pyflex.step()

    # Get the new position
    new_position = pyflex.get_positions()

    # Calculate displacement
    displacement = np.linalg.norm(new_position - initial_position)

    # Calculate dt assuming displacement = velocity * dt
    dt = displacement / np.linalg.norm(initial_velocity)

    # Cleanup
    pyflex.clean()

    return dt


dt = infer_dt_from_displacement()
print(f"Inferred internal simulation time step (dt): {dt} seconds")