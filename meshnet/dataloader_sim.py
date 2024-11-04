from torch_geometric.data import Data
from meshnet.data_utils import *
import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from meshnet.dataloader import SamplesClothDataset, TrajectoriesClothDataset
import glob
import os
from torch_geometric.data import Batch

def get_goal_fold(init_particles, pick, place):
    """
    Fold the cloth in half based on the pick and place locations using PyTorch.
    
    Parameters:
    init_particles (torch.Tensor): Initial particle positions.
    pick (torch.Tensor): Pick location.
    place (torch.Tensor): Place location.
    
    Returns:
    torch.Tensor: Final particle positions after folding.
    """
    # Create a copy of the initial particles
    final_particles = init_particles.clone()
    
    # Define the fold axis (line between pick and place)
    fold_axis = place - pick
    
    # Normalize the fold axis to get the direction
    fold_axis = fold_axis / torch.norm(fold_axis)
    
    # Calculate the midpoint of the fold axis
    midpoint = (pick + place) / 2
    
    # Project the particles onto the fold axis to find their positions relative to the fold axis
    projections = torch.matmul(init_particles - midpoint, fold_axis.view(-1, 1)).squeeze()
    
    # Determine which particles are on the side to be folded
    particles_to_fold = projections < 0
    
    # Reflect the positions of the selected particles across the fold axis
    for i in range(len(final_particles)):
        if particles_to_fold[i]:
            distance_to_fold = torch.dot(init_particles[i] - midpoint, fold_axis)
            final_particles[i] = init_particles[i] - 2 * distance_to_fold * fold_axis

    return final_particles

class SamplesClothSimDataset(SamplesClothDataset):
    
    def __init__(self,
                data_path, 
                input_length_sequence, 
                FLAGS, 
                dt=1., 
                knn=3, 
                delaunay=False, 
                subsample=False, 
                num_samples=300, 
                transform=None,
                sim_data=True
                ): 
        super().__init__(            
                data_path, 
                FLAGS, 
                input_length_sequence=input_length_sequence, 
                dt=dt, 
                knn=knn, 
                delaunay=delaunay, 
                subsample=subsample, 
                num_samples=num_samples,
                sim_data=sim_data
        )

        self.sim_data = sim_data
        
        self.sim_data = sim_data
        
    def load_data(self, data_paths):
<<<<<<< HEAD
        # TODO: extend to multiple envs and eventually transfor it into dictionary for different labels (e.g. position, velocity, label)
        load_keys=['pos', 'vel', 'actions', 'trajectory_params', 'gripper_pos', 'pick', 'place', 'grasped_particle']
        load_keys=['pos', 'actions',  'gripper_pos', 'pick', 'place']
=======

>>>>>>> 9b63d7a (Commit minor changes)
        data = []
        
        # if data_paths is none then initialize it empty
        if data_paths is None:
            print("No data paths provided, empty initialization")
            return data
        
        env_all_trajs = get_env_trajs_path(data_paths)            
        self.num_clothes = len(env_all_trajs)
        
        for all_trajs in env_all_trajs:
            for data_path in all_trajs:
                params = (self._dt, self.k, self.delaunay, self.subsample, self.num_samples, self._input_length_sequence, self._action_steps)
<<<<<<< HEAD
                trajectory_data = get_data_traj(data_path, load_keys, params, sim_data=self.sim_data )                
=======
                trajectory_data = get_data_traj(data_path, self.load_keys, params, sim_data=self.sim_data )                
>>>>>>> 9b63d7a (Commit minor changes)
                data.append(trajectory_data)
        return data
    

    
    # def __getitem__old(self, idx):
    #     # Select the trajectory immediately before
    #     # the one that exceeds the idx
    #     # (i.e., the one in which idx resides).
    #     trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

    #     # Compute index of pick along time-dimension of trajectory.
    #     start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
    #     time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)
    #     time_past = time_idx - self._input_length_sequence

    #     # Prepare training data. Assume `input_sequence_length`=1
    #     # Always use the first graph as input
    #     positions = self._data[trajectory_idx]["pos"][time_idx - 1]  # (nnode, dimension)
        
    #     if self._input_length_sequence == 1:
    #         velocity = self._data[trajectory_idx]["velocity"][time_idx - 1]  # (nnode, dimension)
    #     else: 
    #         velocity = self._data[trajectory_idx]["velocity"][time_past:time_idx]  # (nnode, dimension)
    #         velocity = np.concatenate([v for v in velocity], 1)
            
    #     node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)     

    #     position_target = self._data[trajectory_idx]["pos"][time_idx]  # (nnode, dimension)        
    #     velocity_target = self._data[trajectory_idx]["velocity"][time_idx]  # (nnode, dimension)
        
    #     action = self._data[trajectory_idx]["actions"][time_idx - 1]
    #     grasped_particle = self._data[trajectory_idx]['grasped_particle']
        
    #     faces = self._data[trajectory_idx]["edge_faces"][time_idx - 1]

    #     graph = self._data_to_graph(action, grasped_particle, velocity, node_type, faces, velocity_target, position_target, positions)

    #     return graph
    
    
    
    def __getitem__(self, idx):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)
        time_past = time_idx - self._input_length_sequence
        time_future = time_idx + self._future_sequence_length # min(time_idx + self._future_sequence_length, len(self._data[trajectory_idx]["velocity"])-1)  

        # Prepare training data. Assume `input_sequence_length`=1
        # Always use the first graph as input
        positions = self._data[trajectory_idx]["pos"][time_idx - 1]  # (nnode, dimension)
        
        if self._input_length_sequence == 1:
            velocity = self._data[trajectory_idx]["velocity"][time_idx - 1]  # (nnode, dimension)
        else: 
            velocity = self._data[trajectory_idx]["velocity"][time_past:time_idx]  # (nnode, dimension)
            velocity = np.concatenate([v for v in velocity], 1)
            
        node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)     

        position_target = self._data[trajectory_idx]["pos"][time_idx:time_future]  # (nnode, dimension)        
        velocity_target = self._data[trajectory_idx]["velocity"][time_idx:time_future]  # (nnode, dimension)
        
        action = self._data[trajectory_idx]["actions"][time_idx - 1:time_future-1]
        grasped_particle = self._data[trajectory_idx]['grasped_particle']
        
        faces = self._data[trajectory_idx]["edge_faces"][time_idx - 1]

        graph = self._data_to_graph(action, grasped_particle, velocity, node_type, faces, velocity_target, position_target, positions)

        return graph
    
    
        
    def __get_val_item__(self, idx, future=1):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        if future == -1:
            time_idx = self._input_length_sequence 
            time_past = 0
            time_future =  len(self._data[trajectory_idx]["velocity"])  
        else:
            time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)
            time_past = time_idx - self._input_length_sequence
            time_future = min(time_idx + future, len(self._data[trajectory_idx]["velocity"])-1)  
            
        

        # Prepare training data
        positions = self._data[trajectory_idx]["pos"][time_idx - 1:time_future]  # (nnode, dimension)
        
        if self._input_length_sequence == 1:
            velocity = self._data[trajectory_idx]["velocity"][time_idx - 1:time_future]  # (nnode, dimension)
        else: 
            velocity = self._data[trajectory_idx]["velocity"][time_past:time_future]  # (nnode, dimension)
            # velocity = np.concatenate([v for v in velocity], 1)
            
        node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)     

        # position_target = self._data[trajectory_idx]["pos"][time_idx]  # (nnode, dimension) 
        # velocity_target = self._data[trajectory_idx]["velocity"][time_future]  # (nnode, dimension)
          
        # future_steps = time_future - time_idx
        actions = self._data[trajectory_idx]["actions"][time_idx - 1: time_future - 1]
        grasped_particle = self._data[trajectory_idx]['grasped_particle']
        
        time_idx_vector = np.full(positions.shape[0], time_idx - 1)  # (nnode, )
        time_vector = time_idx_vector * self._dt  # (nnodes, )
        time_vector = np.reshape(time_vector, (time_vector.size, 1))  # (nnodes, 1)

        edge_index = self._data[trajectory_idx]["edge_index"][time_idx - 1]
        # edge_displacement = self._data[trajectory_idx]["edge_displacement"][time_idx - 1:time_future]
        # edge_norm = self._data[trajectory_idx]["edge_norm"][time_idx - 1:time_future]
        
        faces = self._data[trajectory_idx]["edge_faces"][time_idx - 1]

        trajectory_dict = {'pos':torch.tensor(positions).to(torch.float32).contiguous(),
                           'vel':torch.tensor(velocity).to(torch.float32).contiguous(),
                            'actions':torch.tensor(actions).to(torch.float32).contiguous(),
                            'node_type':torch.tensor(node_type).to(torch.float32).contiguous(),
                            # 'time':time_vectors.clone().to(torch.float32).contiguous(),
                            'edge_index':edge_index.clone().contiguous(),
                            # 'edge_displacement':edge_displacement.clone().to(torch.float32).contiguous(),
                            # 'edge_norm':edge_norm.clone().to(torch.float32).contiguous(),
                            'faces':faces.clone().to(torch.float32).contiguous(),
                            'grasped_particle':grasped_particle
                            
        }

        return trajectory_dict
    
    def get_batch_with_candidate_actions(self, idx, candidate_actions):
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx-1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)
        time_past = time_idx - self._input_length_sequence
        time_future = time_idx + self._future_sequence_length

        # Prepare training data. Assume `input_length_sequence`=1
        # Always use the first graph as input
        positions = self._data[trajectory_idx]["pos"][time_idx - 1]  # (nnode, dimension)
        
        if self._input_length_sequence == 1:
            velocity = self._data[trajectory_idx]["velocity"][time_idx - 1]  # (nnode, dimension)
        else: 
            velocity = self._data[trajectory_idx]["velocity"][time_past:time_idx]  # (nnode, dimension)
            velocity = np.concatenate([v for v in velocity], 1)
            
        node_type = self._data[trajectory_idx]["node_type"][time_idx - 1]  # (nnode, 1)     

        position_target = self._data[trajectory_idx]["pos"][time_idx:time_future]  # (nnode, dimension)        
        velocity_target = self._data[trajectory_idx]["velocity"][time_idx:time_future]  # (nnode, dimension)
        
        grasped_particle = self._data[trajectory_idx]['grasped_particle']
        
        faces = self._data[trajectory_idx]["edge_faces"][time_idx - 1]

        # Generate a batch of graphs with candidate actions
        graphs = []
        for action in candidate_actions:
            graph = self._data_to_graph(action, grasped_particle, velocity, node_type, faces, velocity_target, position_target, positions)
            graphs.append(graph)
        
        # Create a batch of graphs
        batch = Batch.from_data_list(graphs)
    
        return batch
    
    def collect_observation(self, observation, first=False, modality='gt', rw_processing=False):
        # This function for now does not handle multiple trajectories in the same dataset, it is done for online planning
        
        # Process the new observation and add it to the dataset
        # You may need to transform the observation to match the expected input format
        params = (self._dt, self.k, self.delaunay, self.subsample, self.num_samples, self._input_length_sequence, self._action_steps)
        data_path = None
        sampled_point_indeces = None
        if not first:
            sampled_point_indeces = self.sampled_point_indeces
        trajectory_data = get_data_traj(data_path, self.load_keys, params, observations=observation, 
                                        sim_data=self.sim_data, sampled_points_indices=sampled_point_indeces,
                                        rw_processing=rw_processing) 
        
        
        
        # TODO: Integrate different positions based on the evaluation.
        # If we want to keep using ground_truth then do nothing, otherwise substitute pos with predicted pos from the model at the 
        # desired indeces. To be defined how the defined indeces are going to be used.
        
        # if this is the first time, then initialize the data, otherwise delete the last element and append the updated one  
        if first:
            self.sampled_point_indeces = trajectory_data['sampled_point_indeces']
        else:
            self._data.pop()
            
        init_particles = torch.from_numpy(trajectory_data['pos'][0]).to(torch.float32)
        pick = trajectory_data['pick']
        goal_place = trajectory_data['place']
        goal_particles = get_goal_fold(init_particles, torch.from_numpy(pick).to(torch.float32), torch.from_numpy(goal_place).to(torch.float32))
        
        # keep track of the ground truth position and velocity
        trajectory_data.update({'gt_pos':copy.deepcopy(trajectory_data['pos'])})
        trajectory_data.update({'gt_vel':copy.deepcopy(trajectory_data['velocity'])})
        
        # TODO: debug this part
        if modality == 'cloth_splatting':
            # update the position with the predicted one, the -1 is because the first elements
            # are repetitions of the first frame based on the input_length_sequence
            # but we want to keep the last. Then update velocity features.
            trajectory_data['pos'][(self._input_length_sequence - 1 ):] = observation['refined_pos']
            trajectory_data['velocity'][self._input_length_sequence:] = observation['refined_pos'][1:] - observation['refined_pos'][:-1]
            
        if modality == 'open_loop':
            # update the position with the predicted one, the -1 is because the first elements
            # are repetitions of the first frame based on the input_length_sequence
            # but we want to keep the last. Then update velocity features.
            trajectory_data['pos'][(self._input_length_sequence - 1 ):] = observation['predicted_pos']
            trajectory_data['velocity'][self._input_length_sequence:] = observation['predicted_pos'][1:] - observation['predicted_pos'][:-1]
        
                
        self._data.append(trajectory_data)
        
        
        
        # update cumulative lengths
        self._compute_cumulative_lengths()
        
        return goal_particles

    
    
    def _data_to_graph(self, action, grasped_particle, velocity, node_type, faces, velocity_target, position_target, positions):
        
        # old version to append the action
        # TODO: handle longer actions
        # This is useful to update the actions in a curriculum learning fashion
        particle_actions = torch.zeros(velocity_target.shape)
        particle_actions[:, grasped_particle] = torch.tensor(action) if  not isinstance(action, torch.Tensor) else action  # (nnode, 3)
        
        # move the position of the particle by that action and set already the correct velocity
        positions_actions = positions.clone() if isinstance(positions, torch.Tensor) else positions.copy()
        positions_actions[grasped_particle] += action[0]
        velocity_actions =  velocity.clone() if isinstance(velocity, torch.Tensor) else velocity.copy()
        # velocity_actions[grasped_particle, :-3] = velocity[grasped_particle][3:]
        velocity_actions[grasped_particle, -3:] = torch.tensor(velocity_target[0, grasped_particle]) if  not isinstance(velocity_target[0, grasped_particle], torch.Tensor) else velocity_target[0, grasped_particle] 
        
        if isinstance(velocity, torch.Tensor):
                    
            # aggregate node features
            node_features = torch.hstack(   
                                        (         
                # actions.to(torch.float32).contiguous(),
                velocity_actions.to(torch.float32).contiguous(),
                node_type.contiguous(),
                                        )
            )

            # Create the graph for the first point cloud
            graph = Data(x=node_features,
                        face=faces.to(torch.int64).contiguous(),
                        y=velocity_target.permute(1, 0, 2).to(torch.float32).contiguous(),
                        pos=positions_actions.to(torch.float32).contiguous(),     
                        # pos=positions.to(torch.float32).contiguous(),
                        pos_target=position_target.permute(1, 0, 2).to(torch.float32).contiguous(),
                        vel=velocity.to(torch.float32).contiguous(),
                        particle_actions=particle_actions.permute(1, 0, 2).to(torch.float32).contiguous(),
                        grasped_particle=grasped_particle
                        )
            
        else:
            # aggregate node features
            node_features = torch.hstack(   
                                        (         
                # actions.to(torch.float32).contiguous(),
                torch.tensor(velocity_actions).to(torch.float32).contiguous(),
                torch.tensor(node_type).contiguous(),
                                        )
            )

            # Create the graph for the first point cloud
            graph = Data(x=node_features,
                        face=faces.to(torch.int64).contiguous(),
                        y=torch.tensor(velocity_target).transpose(1,0).to(torch.float32).contiguous(),
                        pos=torch.tensor(positions_actions).to(torch.float32).contiguous(),   
                        # pos=torch.tensor(positions).to(torch.float32).contiguous(),
                        pos_target=torch.tensor(position_target).transpose(1,0).to(torch.float32).contiguous(),
                        vel=torch.tensor(velocity_actions).to(torch.float32).contiguous(),
                        particle_actions=particle_actions.transpose(1,0).to(torch.float32).contiguous(),
                        grasped_particle=grasped_particle
                        )
                
        return graph
    
    def _graph_to_data(self, graph, input_length_sequence=None):
        if input_length_sequence is None:
            input_length_sequence = self._input_length_sequence
        # action = graph.x[:, :3]
        velocity = graph.x[:, : 3*input_length_sequence]
        node_types = graph.x[:, -1].unsqueeze(1)
        edge_index = graph.edge_index
        edge_features = graph.edge_attr
        target_velocities = graph.y
        particle_actions = graph.particle_actions
        positions = graph.pos
        return velocity, node_types, edge_index, edge_features, target_velocities, particle_actions, positions
            
    
class TrajectoriesClothSimDataset(TrajectoriesClothDataset):

    def __init__(self, data_path, input_length_sequence, FLAGS, dt=1., knn=3, delaunay=False, subsample=False, num_samples=300, sim_data=True):
        super().__init__(data_path, FLAGS=FLAGS, dt=dt, knn=knn, delaunay=delaunay, subsample=subsample, num_samples=num_samples)
        # whose shapes are (600, 1876, 2), (600, 1876, 1), (600, 1876, 2), (600, 3518, 3), (600, 1876, 1)
        # convert to list of tuples
        self._dt = dt
        self.k = knn
        self.delaunay = delaunay
        self.subsample = subsample
        self.num_samples = num_samples
        self.sim_data = sim_data
        # self._data = self.load_data(data_path)

        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0]["pos"].shape[-1]
        self._length = len(self._data)

    def load_data(self, data_paths):
        load_keys=['pos', 'vel', 'actions', 'trajectory_params', 'gripper_pos', 'pick', 'place', 'grasped_particle']        
        load_keys=['pos', 'actions',  'gripper_pos', 'pick', 'place']
        data = []
        
        env_all_trajs = get_env_trajs_path(data_paths)        
            
        self.num_clothes = len(env_all_trajs)        
        
        for all_trajs in env_all_trajs:
            for data_path in all_trajs:
                params = (self._dt, self.k, self.delaunay, self.subsample, self.num_samples, self._input_length_sequence, self._action_steps)
                trajectory_data = get_data_traj(data_path, load_keys, params, sim_data=self.sim_data )                    
                
                data.append(trajectory_data)
                
        return data
    
    def _data_to_graph(self, action, grasped_particle, velocity, node_type, faces, velocity_target, position_target, positions,
                       ):
        # This is useful to update the actions in a curriculum learning fashion
        particle_actions = torch.zeros(velocity_target.shape)
        particle_actions[:, grasped_particle] = torch.tensor(action) if  not isinstance(action, torch.Tensor) else action  # (nnode, 3)
        
        # actions = torch.zeros(positions.shape).to(positions.device)
        # actions[grasped_particle] = action.to(positions.device)  # (nnode, 3)
        # move the position of the particle by that action and set already the correct velocity
        positions_actions = positions.clone()
        positions_actions[grasped_particle] += action[0]
        velocity_actions = velocity.clone()
        velocity_actions[grasped_particle, :-3] = velocity[grasped_particle][3:]
        velocity_actions[grasped_particle, -3:] = velocity_target[0, grasped_particle]
        
        
        # aggregate node features
        node_features = torch.hstack(   
                                    (         
            # actions.to(torch.float32).contiguous(),
            velocity_actions.to(torch.float32).contiguous(),
            node_type[0].contiguous(),
                                    )
        )

        # Create the graph for the first point cloud
        graph = Data(x=node_features,
                    face=faces[0].to(torch.int64).contiguous(),
                    y=velocity_target[0].to(torch.float32).contiguous(),
                    pos=positions_actions.to(torch.float32).contiguous(),
                    pos_target=position_target[0].to(torch.float32).contiguous(),
                    vel=velocity.to(torch.float32).contiguous(),
                    particle_actions=particle_actions.transpose(1,0).to(torch.float32).contiguous(),
                    )
            
        return graph
    
    def _graph_to_data(self, graph, input_length_sequence):
        if input_length_sequence is None:
            input_length_sequence = self._input_length_sequence
        # action = graph.x[:, :3]
        velocity = graph.x[:, :  3*input_length_sequence]
        node_types = graph.x[:, -1].unsqueeze(1)
        edge_index = graph.edge_index
        edge_features = graph.edge_attr
        target_velocities = graph.y
        particle_actions = graph.particle_actions
        positions = graph.pos
        return velocity, node_types, edge_index, edge_features, target_velocities, particle_actions, positions


    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        positions = self._data[idx]["pos"]  # (timesteps, nnode, dims)
        velocities = self._data[idx]["velocity"]  # (timesteps, nnode, dims)
        # actions = np.zeros_like(velocities)
        # actions[:, self._data[idx]['grasped_particle']] =  self._data[idx]["actions"] # (timesteps, nnode, dims)
        actions = self._data[idx]["actions"]  # (timesteps, nnode, dims)
        # n_node_per_example = positions.shape[1]  # (nnode, )
        node_type = self._data[idx]["node_type"]  # (timesteps, nnode, dims)
        # velocity_feature = self._data[idx]["velocity"]  # (timesteps, nnode, dims)
        edge_index = self._data[idx]["edge_index"]
        edge_displacement = self._data[idx]["edge_displacement"]
        edge_norm = self._data[idx]["edge_norm"]
        faces = self._data[idx]["edge_faces"]

        # TODO: build the time vector
        time_vectors = []
        for i in range(positions.shape[0]):
            time_idx = i + 1
            time_idx_vector = np.full(positions[i].shape[0], time_idx - 1)  # (nnode, )
            time_vector = time_idx_vector * self._dt  # (nnodes, )
            time_vector = np.reshape(time_vector, (time_vector.size, 1))  # (nnodes, 1)
            time_vectors.append(time_vector)

        time_vectors = torch.from_numpy(np.asarray(time_vectors))
        grasped_particle = self._data[idx]['grasped_particle']

        # trajectory = (
        #     torch.tensor(positions).to(torch.float32).contiguous(),
        #     torch.tensor(velocities).to(torch.float32).contiguous(),
        #     torch.tensor(actions).to(torch.float32).contiguous(),
        #     torch.tensor(node_type).to(torch.float32).contiguous(),
        #     time_vectors.clone().to(torch.float32).contiguous(),
        #     edge_index.clone().contiguous(),
        #     edge_displacement.clone().to(torch.float32).contiguous(),
        #     edge_norm.clone().to(torch.float32).contiguous(),
        #     grasped_particle
        # )
        
        trajectory_dict = {'pos':torch.tensor(positions).to(torch.float32).contiguous(),
                           'vel':torch.tensor(velocities).to(torch.float32).contiguous(),
                            'actions':torch.tensor(actions).to(torch.float32).contiguous(),
                            'node_type':torch.tensor(node_type).to(torch.float32).contiguous(),
                            'time':time_vectors.clone().to(torch.float32).contiguous(),
                            'edge_index':edge_index.clone().contiguous(),
                            'edge_displacement':edge_displacement.clone().to(torch.float32).contiguous(),
                            'edge_norm':edge_norm.clone().to(torch.float32).contiguous(),
                            'faces':faces.clone().to(torch.float32).contiguous(),
                            'grasped_particle':grasped_particle
                            
        }

        return trajectory_dict
    
    
def get_data_loader_by_samples(path, input_length_sequence, FLAGS, dt, batch_size, knn=3, delaunay=False, subsample=False, num_samples=300, shuffle=True, sim_data=True):
    dataset = SamplesClothSimDataset(path, input_length_sequence, FLAGS, dt, knn, delaunay=delaunay, subsample=subsample, num_samples=num_samples, sim_data=sim_data)
    return torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_data_loader_by_trajectories(path, input_length_sequence, FLAGS, knn=3, delaunay=False, subsample=False, num_samples=300, sim_data=True):
    dataset = TrajectoriesClothSimDataset(path, input_length_sequence, FLAGS, knn=knn, delaunay=delaunay, subsample=subsample, num_samples=num_samples, sim_data=sim_data)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)


if __name__=='__main__':
    data_path = '../sim_dataset/test_dataset_0414/TOWEL/00000'
    # dataset = SamplesClothDataset(data_path)
    dataloader = get_data_loader_by_samples(data_path, input_length_sequence=1, dt=0.01, batch_size=8, shuffle=True)
    # get item 0 from dataloader
    graph = next(iter(dataloader))
    print("FATTOOO")