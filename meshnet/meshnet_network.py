from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import glob
import re
import os

from meshnet.model_utils import Normalizer
from meshnet.graph_network import EncodeProcessDecode


class MeshSimulator(nn.Module):

    def __init__(
            self,
            simulation_dimensions: int,
            nnode_in: int,
            nedge_in: int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            nnode_types: int,
            node_type_embedding_size: int,
            device="cpu"):
        """Initializes the model.

        Args:
          simulation_dimensions: Dimensionality of the problem.
          nnode_in: Number of node inputs.
          nedge_in: Number of edge inputs.
          latent_dim: Size of latent dimension (128)
          nmessage_passing_steps: Number of message passing steps.
          nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
          mlp_hidden_dim: Dimension of hidden layers in the MLP 128.
          nnode_types: Number of different particle types.
          node_type_embedding_size: Embedding size for the particle type.
          device: Runtime device (cuda or cpu).

        """
        super(MeshSimulator, self).__init__()
        self._nnode_types = nnode_types
        self._node_type_embedding_size = node_type_embedding_size

        # Initialize the EncodeProcessDecode
        self._encode_process_decode = EncodeProcessDecode(
            nnode_in_features=nnode_in,  # 3 current velocities + 1 node_type + 1 TIME  (node_type dimensions corresponds to the number of node types, potentially include time)
            nnode_out_features=simulation_dimensions,  # 3
            nedge_in_features=nedge_in,  # 3 relative disp + 1 norm
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim).to(device)

        self._output_normalizer = Normalizer(
            size=simulation_dimensions, name='output_normalizer', device=device)
        self._node_normalizer = Normalizer(
            size=nnode_in, name='node_normalizer', device=device)
        self._device = device

    def forward(self):
        """Forward hook runs on class instantiation"""
        pass

    def _encoder_preprocessor(self,
                              init_position: torch.tensor,
                              time_vector: torch.tensor,
                              node_type: torch.tensor,
                              position_noise: torch.tensor = None):
        """
        Take `current_velocity` (nnodes, dims) and node type (nnodes, 1),
        impose `velocity_noise`, convert integer `node_type` to onehot embedding `node_type`,
        concatenate as `node_features` and normalize it.

        Args:
            current_velocities: current velocity at nodes (nnodes, dims)
            node_type: node_types (nnodes, )
            velocity_noise: velocity noise (nnodes, dims)

        Returns:
            processed_node_features (i.e., noised & normalized)
        """

        # node feature
        node_features = []

        # impose noise to velocity when training. Rollout does not impose noise to velocity.
        if position_noise is not None:  # for training
            noised_positions = init_position + position_noise
            node_features.append(noised_positions)
        if position_noise is None:  # for rollout
            node_features.append(init_position)
            pass

        # embed time
        if time_vector.dim() == 1:
            time_vector = time_vector[:, None]
        node_features.append(time_vector)

        # embed integer node_type to onehot vector
        node_type = torch.squeeze(node_type.long())
        node_type_onehot = torch.nn.functional.one_hot(node_type, self._node_type_embedding_size)
        node_features.append(node_type_onehot)

        node_features = torch.cat(node_features, dim=1)
        processed_node_features = self._node_normalizer(node_features, self.training)

        return processed_node_features

    def predict_dx(
            self,
            init_position,
            time_vector,
            node_type,
            edge_index,
            edge_features,
            target_positions=None,
            position_noise=None):
        """
        Predict acceleration using current features

        Args:
            current_velocities: current velocity at nodes (nnodes, dims)
            node_type: node_types (nnodes, )
            edge_index: index describing edge connectivity between nodes (2, nedges)
            edge_features: [relative_distance, norm] (nedges, 3)
            target_velocities: ground truth velocity at next timestep
            velocity_noise: velocity noise (nnodes, dims)

        Returns:
            predicted_normalized_accelerations, target_normalized_accelerations
        """

        # prepare node features, edge features, get connectivity
        processed_node_features = self._encoder_preprocessor(
            init_position,
            time_vector,
            node_type,
            position_noise)

        # predict acceleration
        predicted_normalized_displacements = self._encode_process_decode(
            processed_node_features.to(torch.float32), edge_index, edge_features)

        if target_positions is None:
            return predicted_normalized_displacements, None

        # target acceleration
        noised_positions = init_position + position_noise
        target_positions_displacements = target_positions - noised_positions
        target_normalized_positions_displacements = self._output_normalizer(target_positions_displacements, self.training)

        # print(self._output_normalizer._mean())
        # print(self._output_normalizer._std_with_epsilon())

        return predicted_normalized_displacements, target_normalized_positions_displacements

    def predict_position(self,
                         init_positions,
                         time_vector,
                         node_type,
                         edge_index,
                         edge_features):
        """
        Predict velocity using current features when rollout

        Args:
            current_velocities: current velocity at nodes (nnodes, dims)
            node_type: node_types (nnodes, )
            edge_index: index describing edge connectivity between nodes (2, nedges)
            edge_features: [relative_distance, norm] (nedges, 3)
        """

        # prepare node features, edge features, get connectivity
        processed_node_features = self._encoder_preprocessor(
            init_positions,
            time_vector,
            node_type,
            position_noise=None)

        # predict dynamics
        predicted_normalized_displacements = self._encode_process_decode(
            processed_node_features, edge_index, edge_features)

        # denormalize the predicted_normalized_accelerations for actual physical domain
        predicted_displacements = self._output_normalizer.inverse(predicted_normalized_displacements)
        predicted_positions = init_positions + predicted_displacements

        return predicted_positions

    def save(self, path=None):
        """

        Args:
            path: The model save path, default should be 'model-<STEP>.pt'

        Returns:

        """

        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()

        save_data = {'model': model,
                     '_output_normalizer': _output_normalizer,
                     '_node_normalizer': _node_normalizer}

        torch.save(save_data, path)

    def load(self, path: str, file='latest'):
        """
        Load model state from file

        Args:
          path: Model path
          file: The filename, if 'latest' will look for model-<STEP>.pt with the highest <STEP> number
        """

        # TODO Make intermediate training state loadable

        if file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(os.path.join(path, '*model*pt'))
            if len(fnames) == 0:
                raise ValueError(f"Did not find any pre-trained weights for the meshnet in: {path}")
            max_model_number = 0
            expr = re.compile('.*model-(\d+).pt')
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            model_file = os.path.join(path, f'model-{max_model_number}.pt')
        else:
            model_file = os.path.join(path, file)

        dicts = torch.load(model_file)
        self.load_state_dict(dicts["model"])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%model_file)


class SinusoidalEncoder(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_freqs: int,
                 min_freq_log2: int = 0,
                 max_freq_log2: Optional[int] = None,
                 scale: float = 1.0,
                 use_identity: bool = True,
                 device='cpu',
                 **kwargs):
        """ A vectorized sinusoidal encoding.

        Args:
          num_freqs: the number of frequency bands in the encoding.
          min_freq_log2: the log (base 2) of the lower frequency.
          max_freq_log2: the log (base 2) of the upper frequency.
          scale: a scaling factor for the positional encoding.
          use_identity: if True use the identity encoding as well.
        """
        super(SinusoidalEncoder, self).__init__()
        self.num_freqs = num_freqs
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2 if max_freq_log2 else min_freq_log2 + num_freqs - 1.0
        self.scale = scale
        self.use_identity = use_identity

        freq_bands = 2.0 ** torch.linspace(self.min_freq_log2,
                                            self.max_freq_log2,
                                            int(self.num_freqs), device=device)

        # (F, 1).
        self.register_buffer('freqs', torch.reshape(freq_bands, (self.num_freqs, 1)))

        self.input_dim = input_dim
        self.output_dim = input_dim * self.num_freqs * 2
        if self.use_identity:
            self.output_dim += self.input_dim

    def __call__(self, x, alpha: Optional[float] = None):
        """A vectorized sinusoidal encoding.

        Args:
          x: the input features to encode.
          alpha: a dummy argument for API compatibility.

        Returns:
          A tensor containing the encoded features.
        """
        if self.num_freqs == 0:
            return x

        x_expanded = torch.unsqueeze(x, dim=-2)  # (1, C).
        # Will be broadcasted to shape (F, C).
        angles = self.scale * x_expanded * self.freqs

        # The shape of the features is (F, 2, C) so that when we reshape it
        # it matches the ordering of the original NeRF code.
        # Vectorize the computation of the high-frequency (sin, cos) terms.
        # We use the trigonometric identity: cos(x) = sin(x + pi/2)
        features = torch.stack((angles, angles + torch.pi / 2), dim=-2)
        features = features.flatten(start_dim=-3, end_dim=-1)
        features = torch.sin(features)

        # Prepend the original signal for the identity.
        if self.use_identity:
            features = torch.cat([x, features], dim=-1)
        return features


class ResidualMeshSimulator(torch.nn.Module):

    def __init__(self,
                 mesh_predictions: torch.Tensor,
                 n_times: int = -1,
                 device='cuda'):

        super().__init__()
        self.mesh_predictions = mesh_predictions.to(device)
        # replace all predictions by the first mesh (used for ablation studies)
        # self.mesh_predictions = self.mesh_predictions[0].unsqueeze(0).repeat(self.mesh_predictions.shape[0], 1, 1)

        if n_times > 0:
            self.n_times = n_times
        else:
            self.n_times = self.mesh_predictions.shape[0]

        self.time_delta = 1.0 / (self.n_times - 1)

        n_nodes = self.mesh_predictions.shape[1]

        self.encoder = SinusoidalEncoder(input_dim=1, num_freqs=6, device=device)
        self.input = torch.nn.Linear(self.encoder.output_dim, 256, device=device)
        self.hidden = torch.nn.Linear(256, 256, device=device)
        self.output = torch.nn.Linear(256, n_nodes*3, device=device)
        nn.init.normal_(self.output.weight, 0.0, 0.00001)
        nn.init.normal_(self.output.bias, 0.0, 0.000001)

    def forward(self, time_vector):
        time = time_vector[0, :]
        h = self.encoder(time)
        h = self.input(h)
        h = torch.nn.functional.relu(h)
        h = self.hidden(h)
        h = torch.nn.functional.relu(h)
        residual_deform = self.output(h).reshape(-1, 3)

        time_id = torch.round(time / self.time_delta).to(dtype=torch.long)
        if time_id >= self.n_times:
            raise ValueError(f"Time {time} is out of bounds for the mesh simulator.")
        return self.mesh_predictions[time_id].squeeze() + residual_deform

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResidualMeshSimulatorEmbedding(torch.nn.Module):

    def __init__(self,
                 mesh_predictions: torch.Tensor,
                 device='cpu'):

        super().__init__()
        self.mesh_predictions = mesh_predictions.to(device)

        self.n_times = self.mesh_predictions.shape[0]
        self.time_delta = 1 / (self.n_times - 1)

        n_nodes = self.mesh_predictions.shape[1]

        self.embedding = torch.nn.Embedding(self.n_times, n_nodes*3)
        nn.init.normal_(self.embedding.weight, 0.0, 0.001)

    def forward(self, time_vector):
        time = time_vector[0, :]
        time_id = torch.round(time / self.time_delta).to(dtype=torch.long)

        residual_deform = self.embedding(time_id).reshape(-1, 3)

        return self.mesh_predictions[time_id].squeeze() + residual_deform

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

