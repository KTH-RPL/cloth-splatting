import torch
import torch.nn as nn
import numpy as np
import glob
import re
import os

from meshnet.model_utils import Normalizer, IdentityNormalizer
from meshnet.graph_network import EncodeProcessDecode



class ClothMeshSimulator(nn.Module):

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
            normalize: bool = False,
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
        super(ClothMeshSimulator, self).__init__()
        self._nnode_types = nnode_types
        self._node_type_embedding_size = node_type_embedding_size

        # Initialize the EncodeProcessDecode
        self._encode_process_decode = EncodeProcessDecode(
            nnode_in_features=nnode_in,  # 3 current velocities + 1 node_type + 3 actions  (node_type dimensions corresponds to the number of node types, potentially include time)
            nnode_out_features=simulation_dimensions,  # 3
            nedge_in_features=nedge_in,  # 3 relative disp + 1 norm
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim).to(device)

        # create identity normalizer to avoid normalization
        if normalize:
            self._output_normalizer = Normalizer(
                size=simulation_dimensions, name='output_normalizer', device=device)
            self._node_normalizer = Normalizer(
                size=nnode_in, name='node_normalizer', device=device)
        else:
            self._output_normalizer = IdentityNormalizer(size=simulation_dimensions, name='output_normalizer', device=device)
            self._node_normalizer = IdentityNormalizer(size=simulation_dimensions, name='output_normalizer', device=device)
        self._device = device

    def forward(self):
        """Forward hook runs on class instantiation"""
        pass

    def _encoder_preprocessor(self,
                              velocity: torch.tensor,      # was pos
                              node_type: torch.tensor,
                              velocity_noise: torch.tensor = None):
        """
        Take `current_velocity` (nnodes, dims), action (nnodes, dims) and node type (nnodes, 1),
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
        if velocity_noise is not None:  # for training
            noised_velocity = velocity + velocity_noise
            node_features.append(noised_velocity)
        if velocity_noise is None:  # for rollout
            node_features.append(velocity)
            pass


        # embed integer node_type to onehot vector
        node_type = torch.squeeze(node_type.long())
        node_type_onehot = torch.nn.functional.one_hot(node_type, self._node_type_embedding_size)
        node_features.append(node_type_onehot)

        node_features = torch.cat(node_features, dim=1)
        processed_node_features = self._node_normalizer(node_features, self.training)

        return processed_node_features

    def predict_acceleration(
            self,
            velocity,
            node_type,
            edge_index,
            edge_features,
            target_velocities=None,
            velocity_noise=None):
        """
        Predict acceleration using current features

        Args:
            current_velocities: current velocity at nodes (nnodes, dims)
            actions: action at nodes (nnodes, dims)
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
            velocity,
            node_type,
            velocity_noise)

        # predict acceleration
        predicted_normalized_accelerations = self._encode_process_decode(
            processed_node_features.to(torch.float32), edge_index, edge_features)

        if target_velocities is None:
            return predicted_normalized_accelerations, None


        # target acceleration
        noised_velocities = velocity 
        if velocity_noise is not None:
            noised_velocities += velocity_noise
            target_accelerations = target_velocities - noised_velocities
        else: 
            target_accelerations = target_velocities - velocity[:, -3:]
        target_normalized_accelerations = self._output_normalizer(target_accelerations, self.training)


        # print(self._output_normalizer._mean())
        # print(self._output_normalizer._std_with_epsilon())

        return predicted_normalized_accelerations, target_normalized_accelerations

    def predict_velocity(self,
                         velocities,
                         node_type,
                         edge_index,
                         edge_features):
        """
        Predict velocity using current features when rollout

        Args:
            current_velocities: current velocity at nodes (nnodes, dims)
            actions: action at nodes (nnodes, dims)
            node_type: node_types (nnodes, )
            edge_index: index describing edge connectivity between nodes (2, nedges)
            edge_features: [relative_distance, norm] (nedges, 3)
        """

        # prepare node features, edge features, get connectivity
        processed_node_features = self._encoder_preprocessor(
            velocities,
            node_type,
            velocity_noise=None)

        # predict dynamics
        predicted_normalized_accelerations = self._encode_process_decode(
            processed_node_features, edge_index, edge_features)

        # denormalize the predicted_normalized_accelerations for actual physical domain
        predicted_accelerations = self._output_normalizer.inverse(predicted_normalized_accelerations)
        predicted_velocity = velocities[:, -3:] + predicted_accelerations

        return predicted_velocity

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

