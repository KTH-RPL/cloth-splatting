import torch
import torch.nn as nn
import enum
from torch_geometric.data import Data

class NodeType(enum.IntEnum):
    CLOTH = 0
    # OBSTACLE = 1
    # AIRFOIL = 2
    # HANDLE = 3
    # INFLOW = 4
    # OUTFLOW = 5
    # WALL_BOUNDARY = 6
    # SIZE = 9


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10 ** 6, std_epsilon=1e-8, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate:
            # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data ** 2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count,
                                   torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count,
                                   torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):

        dict = {'_max_accumulations': self._max_accumulations,
                '_std_epsilon': self._std_epsilon,
                '_acc_count': self._acc_count,
                '_num_accumulations': self._num_accumulations,
                '_acc_sum': self._acc_sum,
                '_acc_sum_squared': self._acc_sum_squared,
                'name': self.name
                }

        return dict

def get_velocity_noise(graph, noise_std, device):
    velocity_sequence = graph.x[:, 1:4]
    type = graph.x[:, 0]
    noise = torch.normal(std=noise_std, mean=0.0, size=velocity_sequence.shape).to(device)
    mask = type != NodeType.CLOTH
    noise[mask] = 0
    return noise.to(device)

def datas_to_graph(training_example, dt, device):

    # features
    node_coords = training_example[0][0].to(device)  # (nnodes, dims)
    node_type = training_example[0][1].to(device)  # (nnodes, 1)
    velocity_feature = training_example[0][2].to(device)  # (nnodes, dims)
    time_vector = training_example[0][3] * dt  # (nnodes, )
    time_vector = time_vector.unsqueeze(1).to(device)
    # n_node_per_example = training_example[0][6]
    edge_index = training_example[0][4].to(device)
    edge_displacement = training_example[0][5].to(device)
    edge_norm = training_example[0][6].to(device)

    # aggregate node features
    node_features = torch.hstack((node_type, velocity_feature, time_vector)).to(device)

    # aggregate edge features
    edge_features = torch.hstack((edge_displacement, edge_norm)).to(device)

    # target velocity
    velocity_target = training_example[1].to(device)  # (nnodes, dims)

    # make graph
    graph = Data(x=node_features.to(torch.float32).contiguous(),
                 edge_index=edge_index,
                 edge_attr=edge_features,
                 y=velocity_target,
                 pos=node_coords)

    return graph



def datas_to_graph_pos(training_example, dt, device):

    # features
    node_coords = training_example[0][0].to(device)  # (nnodes, dims)
    node_type = training_example[0][1].to(device)  # (nnodes, 1)
    time_feature = training_example[0][2].to(device)  # (nnodes, dims)
    time_vector = training_example[0][3] * dt  # (nnodes, )
    time_vector = time_vector.unsqueeze(1).to(device)
    # n_node_per_example = training_example[0][6]
    edge_index = training_example[0][4].to(device)
    edge_displacement = training_example[0][5].to(device)
    edge_norm = training_example[0][6].to(device)

    # aggregate node features
    # node_features = torch.hstack((node_type, time_feature, time_vector)).to(device)
    node_features = torch.hstack((node_type, time_feature)).to(device)

    # aggregate edge features
    edge_features = torch.hstack((edge_displacement, edge_norm)).to(device)

    # target psoition
    pos_target = training_example[1].to(device)  # (nnodes, dims)

    # make graph
    graph = Data(x=node_features.to(torch.float32).contiguous(),
                 edge_index=edge_index,
                 edge_attr=edge_features,
                 y=pos_target,
                 pos=node_coords)

    return graph


def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)