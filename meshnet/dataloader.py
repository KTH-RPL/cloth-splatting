from torch_geometric.data import Data
from meshnet.data_utils import load_traj, plot_pcd_list, process_traj
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data


class SamplesClothDataset(torch.utils.data.Dataset):
    """
    MESHNET dataset class for cloth data
    Code inspired by https://github.com/geoelements/gns/blob/main/meshnet/data_loader.py
    """

    def __init__(self, data_path, FLAGS,  input_length_sequence=1, dt=1., knn=3, delaunay=False, subsample=False, num_samples=300, transform=None, sim_data=True):
        super().__init__()
        self._dt = dt
        self.delaunay = delaunay
        self.k = knn
        self.subsample = subsample
        self.num_samples = num_samples
        self._input_length_sequence = input_length_sequence
        self._action_steps = FLAGS.action_steps
        self._future_sequence_length = FLAGS.future_sequence_length                
        self.sim_data = sim_data

        self._data = self.load_data(data_path)
        self.transform = transform

        # length of each trajectory in the dataset
        self._compute_cumulative_lengths()
        
    def _compute_cumulative_lengths(self):
        # length of each trajectory in the dataset
        self._data_lengths = [x["pos"].shape[0] - self._input_length_sequence - self._future_sequence_length + 1 for x in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def load_data(self, data_path):
        # TODO: extend to multiple trajectories and eventually transfor it into dictionary for different labels (e.g. position, velocity, label)
        data = []
        traj = load_traj(data_path)
        trajectory_data = process_traj(traj, self._dt, self.k, self.delaunay, subsample=self.subsample, num_samples=self.num_samples)
        data.append(trajectory_data)
        return data


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data. Assume `input_sequence_length`=1
        # Always use the first graph as input
        positions = self._data[trajectory_idx]["pos"][0]  # (nnode, dimension)
        # positions = np.transpose(positions)
        n_node_per_example = positions.shape[0]  # (nnode, )#
        node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)

        position_target = self._data[trajectory_idx]["pos"][time_idx]  # (nnode, dimension)
        time_idx_vector = np.full(positions.shape[0], time_idx - 1)  # (nnode, )
        time_vector = time_idx_vector * self._dt  # (nnodes, )
        time_vector = np.reshape(time_vector, (time_vector.size, 1))  # (nnodes, 1)

        edge_index = self._data[trajectory_idx]["edge_index"][time_idx - 1]
        edge_displacement = self._data[trajectory_idx]["edge_displacement"][time_idx - 1]
        edge_norm = self._data[trajectory_idx]["edge_norm"][time_idx - 1]

        # aggregate node features
        node_features = torch.hstack(
            (
            torch.tensor(node_type).contiguous(),
             torch.tensor(time_vector).to(torch.float32).contiguous())
        )

        edge_features = torch.hstack(
            (edge_displacement.clone().detach().to(torch.float32).contiguous(),
            edge_norm.clone().detach().to(torch.float32).contiguous(),)
        )

        # Create the graph for the first point cloud
        graph = Data(x=node_features,
                     edge_index=edge_index.clone().detach().contiguous(),
                     edge_attr=edge_features,
                     y=torch.tensor(position_target).to(torch.float32).contiguous(),
                     pos=torch.tensor(positions).to(torch.float32).contiguous())


        return graph


class TrajectoriesClothDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, FLAGS, dt=1., knn=3, delaunay=False, subsample=False, num_samples=300):
        super().__init__()
        # whose shapes are (600, 1876, 2), (600, 1876, 1), (600, 1876, 2), (600, 3518, 3), (600, 1876, 1)
        # convert to list of tuples
        self._dt = dt
        self.k = knn
        self.delaunay = delaunay
        self.subsample = subsample
        self.num_samples = num_samples
        self._input_length_sequence = FLAGS.input_sequence_length
        self._action_steps = FLAGS.action_steps
        self._data = self.load_data(data_path)

        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["pos"].shape[-1]
        self._length = len(self._data)

    def load_data(self, data_path):
        data = []
        traj = load_traj(data_path)
        trajectory_data = process_traj(traj, self._dt, self.k, self.delaunay, subsample=self.subsample, num_samples=self.num_samples)
        data.append(trajectory_data)
        return data


    def __len__(self):
        return self._length


    def __getitem__(self, idx):
        positions = self._data[idx]["pos"]  # (timesteps, nnode, dims)
        n_node_per_example = positions.shape[1]  # (nnode, )
        node_type = self._data[idx]["node_type"]  # (timesteps, nnode, dims)
        velocity_feature = self._data[idx]["velocity"]  # (timesteps, nnode, dims)
        edge_index = self._data[idx]["edge_index"]
        edge_displacement = self._data[idx]["edge_displacement"]
        edge_norm = self._data[idx]["edge_norm"]

        # TODO: build the time vector
        time_vectors = []
        for i in range(positions.shape[0]):
            time_idx = i + 1
            time_idx_vector = np.full(positions[i].shape[0], time_idx - 1)  # (nnode, )
            time_vector = time_idx_vector * self._dt  # (nnodes, )
            time_vector = np.reshape(time_vector, (time_vector.size, 1))  # (nnodes, 1)
            time_vectors.append(time_vector)

        time_vectors = torch.from_numpy(np.asarray(time_vectors))

        trajectory = (
            torch.tensor(positions).to(torch.float32).contiguous(),
            torch.tensor(node_type).contiguous(),
            time_vectors.clone().to(torch.float32).contiguous(),
            edge_index.clone().contiguous(),
            edge_displacement.clone().to(torch.float32).contiguous(),
            edge_norm.clone().to(torch.float32).contiguous(),
        )

        return trajectory


def get_data_loader_by_samples(path, input_length_sequence, dt, batch_size, knn=3, delaunay=False, subsample=False, num_samples=300, shuffle=True):
    dataset = SamplesClothDataset(path, input_length_sequence, dt, knn, delaunay=delaunay, subsample=subsample, num_samples=num_samples)
    return torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_data_loader_by_trajectories(path, knn=3, delaunay=False, subsample=False, num_samples=300):
    dataset = TrajectoriesClothDataset(path, knn=knn, delaunay=delaunay, subsample=subsample, num_samples=num_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)


if __name__=='__main__':
    data_path = './data/final_scenes/smaller_scene/final_scene_1_gt_eval.npz'
    # dataset = SamplesClothDataset(data_path)
    dataloader = get_data_loader_by_samples(data_path, input_length_sequence=1, dt=0.01, batch_size=8, shuffle=True)
    # get item 0 from dataloader
    graph = next(iter(dataloader))
    print("FATTOOO")