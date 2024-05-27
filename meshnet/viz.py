import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import imageio.v2 as imageio

def plot_pcd_list(pcd_list, center_plot=None, elev=None, azim=None):
    # plot point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pcd in pcd_list:
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2])

    # find the min, max, and range for each axis across all point clouds
    min_values = [min(pcd[:, i].min() for pcd in pcd_list) for i in range(3)]
    max_values = [max(pcd[:, i].max() for pcd in pcd_list) for i in range(3)]
    ranges = [max_val - min_val for min_val, max_val in zip(min_values, max_values)]

    # find the maximum range
    max_range = max(ranges)

    # calculate the means for each axis if not provided
    if center_plot is None:
        center_plot = [(min_val + max_val) / 2 for min_val, max_val in zip(min_values, max_values)]

    # set the range for each axis based on the desired mean and maximum range
    ax.set_xlim([center_plot[0] - max_range / 2, center_plot[0] + max_range / 2])
    ax.set_ylim([center_plot[1] - max_range / 2, center_plot[1] + max_range / 2])
    ax.set_zlim([center_plot[2] - max_range / 2, center_plot[2] + max_range / 2])

    # set the view angle
    if elev is not None and azim is not None:
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


def plot_mesh(points, edges, center_plot=None, white_bkg=False, save_fig=False, file_name='mesh.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=2)

    # Plot edges
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='b', linewidth=1)

    # find the min, max, and range for each axis across all point clouds
    min_values = [points[:, i].min() for i in range(3)]
    max_values = [points[:, i].max() for i in range(3)]
    ranges = [max_val - min_val for min_val, max_val in zip(min_values, max_values)]

    # find the maximum range
    max_range = max(ranges)

    # calculate the means for each axis if not provided
    if center_plot is None:
        center_plot = [(min_val + max_val) / 2 for min_val, max_val in zip(min_values, max_values)]

    # set the range for each axis based on the desired mean and maximum range
    ax.set_xlim([center_plot[0] - max_range / 2, center_plot[0] + max_range / 2])
    ax.set_ylim([center_plot[1] - max_range / 2, center_plot[1] + max_range / 2])
    ax.set_zlim([center_plot[2] - max_range / 2, center_plot[2] + max_range / 2])


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    plt.show()
    
    
def plot_mesh_predictions(gt_points, pred_points, edges, center_plot=None, white_bkg=False, save_fig=False, return_image=False, file_name='mesh.png', azim=30, elev=0): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth points
    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='r', marker='o', s=2, label='Ground Truth')

    # Plot predicted points
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='g', marker='x', s=2, label='Predicted')

    # Plot edges
    for edge in edges:
        p1, p2 = gt_points[edge[0]], gt_points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='r', linewidth=1)
        
    # Plot edges
    for edge in edges:
        p1, p2 = pred_points[edge[0]], pred_points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='g', linewidth=1)

    # find the min, max, and range for each axis across all point clouds
    min_values = [min(gt_points[:, i].min(), pred_points[:, i].min()) for i in range(3)]
    max_values = [max(gt_points[:, i].max(), pred_points[:, i].max()) for i in range(3)]
    ranges = [max_val - min_val for min_val, max_val in zip(min_values, max_values)]

    # find the maximum range
    max_range = max(ranges)

    # calculate the means for each axis if not provided
    if center_plot is None:
        center_plot = [(min_val + max_val) / 2 for min_val, max_val in zip(min_values, max_values)]

    # set the range for each axis based on the desired mean and maximum range
    ax.set_xlim([center_plot[0] - max_range / 2, center_plot[0] + max_range / 2])
    ax.set_ylim([center_plot[1] - max_range / 2, center_plot[1] + max_range / 2])
    ax.set_zlim([center_plot[2] - max_range / 2, center_plot[2] + max_range / 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    ax.view_init(elev=elev, azim=azim)
    
    plt.legend()

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    
    if return_image:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close()
        return img
    
    plt.show()
    
def plot_losses(losses, return_image=False):
    fig = plt.figure()
    # make the x axis to start from 1
    plt.plot(losses)
    # set range for y axis
    plt.ylim([0, 0.01])
    # set x an y titles
    plt.xlabel('Prediction step')
    plt.ylabel('Prediction Loss')
    
    if return_image:
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close()
        return img

def plot_mesh_and_points(mesh_points, edges, points, 
                         center_plot=None, white_bkg=False, 
                         elev=0, azim=30, 
                         save_fig=False, file_name='mesh.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], c='r', marker='o', s=2)

    # Plot edges
    for edge in edges:
        p1, p2 = mesh_points[edge[0]], mesh_points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='black', linewidth=1)

    # find the min, max, and range for each axis across all point clouds
    min_values = [mesh_points[:, i].min() for i in range(3)]
    max_values = [mesh_points[:, i].max() for i in range(3)]
    ranges = [max_val - min_val for min_val, max_val in zip(min_values, max_values)]

    # find the maximum range
    max_range = max(ranges)

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='g', marker='x', s=10)

    # set elev and azimuth
    ax.view_init(elev=elev, azim=azim)

    # calculate the means for each axis if not provided
    if center_plot is None:
        center_plot = [(min_val + max_val) / 2 for min_val, max_val in zip(min_values, max_values)]

    # set the range for each axis based on the desired mean and maximum range
    # ax.set_xlim([center_plot[0] - max_range / 2, center_plot[0] + max_range / 2])
    # ax.set_ylim([center_plot[1] - max_range / 2, center_plot[1] + max_range / 2])
    # ax.set_zlim([center_plot[2] - max_range / 2, center_plot[2] + max_range / 2])
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([-0.3, 0.3])


    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    if white_bkg:
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    if save_fig:
        plt.savefig(file_name)  # Save the figure to a file
        plt.close(fig)  # Close the figure to free memory
    plt.show()


def plot_mesh_ax(points, edges, ax, center_plot=None, white_bkg=False):
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=2)

    # Plot edges
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='b', linewidth=1)


    # fix axis
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if white_bkg:
        ax.set_facecolor('white')



def create_gif(image_paths, gif_path, fps=1):
    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))
    imageio.mimwrite(gif_path, images, 'GIF', fps=fps, loop=0)
    # imageio.mimsave(gif_path, images, fps=fps)  # fps controls the speed of the GIF
    
    
