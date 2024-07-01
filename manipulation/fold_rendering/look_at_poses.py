import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import roma
import random
# import rerun as rr

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=0.05, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
        
########################
def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion.
    Args:
        axis: The axis [n, 3]
        angle: The angle [n]

    Returns: The quaternion [n, 4] (XYZW)
    """
    qxyz = axis * torch.sin(angle / 2).unsqueeze(1)
    qw = torch.cos(angle / 2).unsqueeze(1)
    return torch.cat([qxyz, qw], dim=1)


def vertice_rotation(normals_a: torch.Tensor, normals_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation between sets of normals (elemt-wise).
    Args:
        normals_a: Initial normals [n, 3]
        normals_b: Rotated normals [n, 3]

    Returns: The quaternion [n, 4] (XYZW)

    """
    # Compute cross product to find the rotation axis
    cross_prod = torch.cross(normals_a, normals_b, dim=1)
    # Compute the dot product to find the cosine of the angle
    dot_prod = torch.sum(normals_a * normals_b, dim=1)
    # Compute the angle between the vectors
    angles = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
    axes = cross_prod / torch.linalg.norm(cross_prod, dim=1, keepdim=True)
    return axis_angle_to_quat(axes, angles)


def get_poses_marcel(r = 0.5, n = 50):
    
    poses = []
    for cam_id in range(n):
        theta = 2 * np.pi * random.random()
        phi = np.pi * random.random() / 3
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        
        translation = np.array([x, y, z])
        position = torch.tensor(translation, dtype=torch.float32)
        center = torch.tensor([[0.0, 0.0, 0.0]])
        direction = (center - position)
        direction = direction / torch.linalg.norm(direction)
        orientation = vertice_rotation(torch.tensor([[0, 0, -1.0]]), direction)
        rotation = roma.unitquat_to_rotmat(orientation).detach().numpy()
        pose = np.eye(4)
        pose[:3, 3] = translation
        pose[:3, :3] = rotation
        
        poses.append(pose)
    return poses
############################

def generate_camera_matrices(n):
    def look_at(eye, center, up):
        f = np.array(center) - np.array(eye)
        f = f / np.linalg.norm(f)
        up = up / np.linalg.norm(up)
        s = np.cross(f, up)
        u = np.cross(s, f)
        
        m = np.eye(4)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        t = np.eye(4)
        t[:3, 3] = -np.array(eye)
        
        return m @ t

    def spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.array([x, y, z])
    
    radius = 5  # or any radius you prefer
    center = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    cameras = []
    for i in range(n):
        theta = 2 * np.pi * (i / n)
        phi = np.pi * (i / n)
        eye = spherical_to_cartesian(radius, theta, phi)
        camera_matrix = look_at(eye, center, up)
        cameras.append(camera_matrix.tolist())
    
    return cameras

def save_to_json(camera_matrices, filename="lookat_camera_matrices.json"):
    data = {
        "frames": [
            {
                "file_path": f"./train/r_{cam_id}",
                "rotation": 0.012566370614359171,
                "transform_matrix": matrix.tolist()
            }
            for cam_id, matrix in enumerate(camera_matrices)
        ]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_camera_poses(camera_matrices):
    xlim = (-2, 2)
    ylim = (-2, 2)
    zlim = (-2, 2)

    visualizer = CameraPoseVisualizer(xlim, ylim, zlim)
    
    for i, matrix in enumerate(camera_matrices):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.array(matrix)[:3, :3]  # Set rotation part
        extrinsic[:3, 3] = np.array(matrix)[:3, 3]  # Set translation part
        visualizer.extrinsic2pyramid(extrinsic, color=plt.cm.rainbow(i / len(camera_matrices)))

    visualizer.customize_legend([f'Camera {i+1}' for i in range(len(camera_matrices))])
    visualizer.show()


        
if __name__ == "__main__":
    n = 12  # number of cameras
    camera_matrices = generate_camera_matrices(n)
    camera_matrices = get_poses_marcel(r = 2.0, n = n)
    save_to_json(camera_matrices)

    for i, matrix in enumerate(camera_matrices):
        print(f"Camera {i+1} Transformation Matrix:")
        print(matrix)
        print()
        
    # Example camera_matrices input for testing
    # camera_matrices = [
    #     [[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, -0.5], [0, 0, 0, 1]],
    #     [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, -0.5], [0, 0, 0, 1]]
    # ]


    # Plot the camera poses
    plot_camera_poses(camera_matrices)