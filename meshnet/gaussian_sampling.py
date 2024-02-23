from scipy.spatial import ConvexHull, Delaunay
import copy
import numpy as np
import matplotlib.pyplot as plt

def sample_gaussian_means(triangle_vertices, barycentric_coords, num_samples_per_triangle):
    # add one dimension to the baricenters and repeat for each coordinate of the vertices
    barycentric_coords = np.expand_dims(barycentric_coords, axis=3)
    barycentric_coords = np.repeat(barycentric_coords, 3, axis=3)

    # add one dimension to the vertices and repeat for each num_sample_per_triange
    triangle_vertices = np.expand_dims(triangle_vertices, axis=0)
    triangle_vertices = np.repeat(triangle_vertices, num_samples_per_triangle, axis=0)

    # get the sample points as multiplitcation of the barycentric coordinates and the vertices
    sampled_points = np.einsum('ijkm,ijkm->ijkm', barycentric_coords, triangle_vertices)
    sampled_points = sampled_points.sum(axis=-2)

    return sampled_points


def get_mesh(points, plot=False):
    points2d = points[:, :2]
    # Compute the Delaunay triangulation
    tri = Delaunay(points2d)

    if plot:
        # Visualize the triangulation
        plt.triplot(points2d[:, 0], points2d[:, 1], tri.simplices)
        plt.plot(points2d[:, 0], points2d[:, 1], 'o')
        plt.show()

    # Extract the edges from the triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = (min(simplex[i], simplex[(i + 1) % 3]), max(simplex[i], simplex[(i + 1) % 3]))
            edges.add(edge)

    # set to numpy array
    edges = list(edges)
    edges = np.array(edges)

    triangle_vertices = points[tri.simplices]
    # n_siomplex = tri.simplices.shape[0]

    return edges, triangle_vertices




