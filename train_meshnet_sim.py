import sys
import os
import glob
import numpy as np
import torch
import torch_geometric.transforms as T
import re
import time
import pickle
from meshnet.cloth_network import ClothMeshSimulator
from meshnet.model_utils import optimizer_to, NodeType, datas_to_graph_pos, datas_to_graph
from meshnet.model_utils import get_velocity_noise
import meshnet.dataloader_sim as data_loader
from meshnet.data_utils import compute_edge_features
from meshnet.viz import plot_mesh, plot_mesh_and_points, plot_mesh_predictions, plot_losses, plot_pcd_list
from tqdm import tqdm
from absl import flags
from absl import app
import wandb
import random
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
import h5py
import torch.optim as optim
import torch_geometric
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
edge_transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])



def predict(simulator, device, FLAGS):

    # Load simulator
    if os.path.exists(FLAGS.model_path + FLAGS.model_file):
        simulator.load(FLAGS.model_path, FLAGS.model_file,)
    else:
        raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    # Load trajectory data.
    # TODO: integrate a proper train-test split
    # ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz", knn=FLAGS.knn)
    ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}",
                                                     FLAGS=FLAGS,
                                                     knn=FLAGS.knn,
                                                     delaunay=True*FLAGS.delaunay,
                                                     subsample=True*FLAGS.subsample,
                                                     num_samples=FLAGS.num_samples,
                                                     )
    input_sequence_length = FLAGS.input_sequence_length

    # Rollout
    with torch.no_grad():
        for i, features in enumerate(ds):
            nsteps = len(features['pos']) - input_sequence_length
            prediction_data = rollout(simulator, ds, features, nsteps, device, FLAGS.input_sequence_length, FLAGS.dt)
            print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']}")
            
            # make a plot to visualize the rollout
            plot_mesh(prediction_data['initial_pos'], prediction_data['edge_index'][0].T, )
            plot_mesh(prediction_data['initial_pos'] + prediction_data['predicted_rollout'][0], prediction_data['edge_index'][0].T, )
            for i in range(len(prediction_data['predicted_rollout'])):
                gt_points = prediction_data['node_coords'][i+input_sequence_length]
                # pred_points = prediction_data['initial_pos'] + prediction_data['ground_truth_rollout'][:i+1].sum(axis=0)
                pred_points = prediction_data['initial_pos'] + prediction_data['predicted_rollout'][:i+1].sum(axis=0)
                edges = prediction_data['edge_index'][i].T
                plot_mesh_predictions(gt_points, pred_points, edges, center_plot=None, white_bkg=False, 
                                      save_fig=False, file_name='mesh.png')
            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                filename = f'{FLAGS.rollout_filename}_{i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(prediction_data, f)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']}")

def rollout(simulator, ds, features, nsteps, device, input_sequence_length, dt, real_world=False):

    node_coords = features['pos'] # (timesteps, nnode, ndims)
    node_vels = features['vel']  # (timesteps, nnode, ndims)
    actions = features['actions']  # (timesteps, nnode, ndims)
    node_types = features['node_type']  # (timesteps, nnode, )
    # times = features['time']  # (timesteps, nnode, ndims)
    edge_index = features['edge_index']  # (2, nedges)
    # edge_features = features['edge_displacement']  # (nedges, 3)
    # edge_displacement = features['edge_norm']  # (nedges, 3)
    faces = features['faces']  # (nfaces, 3)
    grasped_particle = features['grasped_particle']  # (timesteps, nnode, )
    
    # plot_pcd_list([node_coords[0], np.asarray([node_coords[0][grasped_particle]])])
    # plot_mesh(node_coords[0], edge_index[0].T)

    initial_velocities = node_vels[:input_sequence_length]
    ground_truth_velocities = node_vels[input_sequence_length:]

    current_velocities = initial_velocities.to(device)

    # Compute the edge lengths for the original mesh
    edge_vectors = node_coords[0][edge_index.T[:, 1]] - node_coords[0][edge_index.T[:, 0]]
    edge_vectors = edge_vectors.to(device)
    original_edge_lengths = torch.norm(edge_vectors, dim=1).to(device)

    predictions = []
    mask = None
    current_node_coords = node_coords[0].to(device)
    # for step in tqdm(range(nsteps), total=nsteps):
    
    # get time in seconds

    time__now = time.time()
    for step in range(nsteps):
        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords_real = node_coords[step].to(device)
        current_node_type = node_types.to(device)
        current_action = actions[step:step+1].to(device)
        current_faces = faces.to(device)

        # current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        next_ground_truth_velocities = ground_truth_velocities[step:step+1].to(device)
        next_ground_truth_pos = node_coords[step + 1:step+2].to(device)
        ########################################################################
        # current_example = (
        #     (current_node_coords, current_node_type, current_velocities, current_action, current_time, 
        #      current_edge_index, current_edge_features, current_edge_displacement),
        #     next_ground_truth_velocities, next_ground_truth_pos)

        # # Make graph
        # graph = datas_to_graph(current_example, dt=dt, device=device)
        ########################################################################
        velocity = torch.cat([c for c in current_velocities], 1).to(device)
        graph = ds.dataset._data_to_graph(current_action, grasped_particle, 
                                          velocity, current_node_type, current_faces,
                                          next_ground_truth_velocities, next_ground_truth_pos, 
                                          current_node_coords)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        graph = transformer(graph)

        # Predict next velocity
        velocities, node_ty, edge_ind, edge_feat, target_vel, particle_actions, positions= ds.dataset._graph_to_data(graph, input_sequence_length)
        predicted_next_velocity = simulator.predict_velocity(
            velocities=velocities,
            node_type=node_ty,
            edge_index=edge_ind,
            edge_features=edge_feat)
        
        # todo: remember that you added this
        # predicted_next_velocity[features['grasped_particle']] = current_action

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the same for the later timesteps
            # mask = torch.logical_or(current_node_type == NodeType.NORMAL, current_node_type == NodeType.OUTFLOW)
            mask = current_node_type == NodeType.CLOTH
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1).to(device)
            
            
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        # predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
        predicted_next_velocity[grasped_particle] = current_action
        # predicted_next_velocity[grasped_particle] = 0
        
        ########################################################################################################
        # updated_node_coords = current_node_coords + predicted_next_velocity
        # updated_edge_vectors = updated_node_coords[edge_index.T[:, 1]] - updated_node_coords[edge_index.T[:, 0]]
        # updated_edge_lengths = torch.norm(updated_edge_vectors, dim=1)
        
        
        # # Compute the deviation from the original edge lengths
        # deviation_norm = original_edge_lengths - updated_edge_lengths

        # # Ensure the ratios are between 0.9 and 1.1 to avoid too much deviation
        # length_ratios = torch.clamp(length_ratios, 0.9, 1.1)
        
        # # Calculate the corrected edge vectors
        # corrected_edge_vectors = edge_vectors * length_ratios.unsqueeze(1)
        # # Calculate the displacement needed to adjust the velocities
        # displacement = corrected_edge_vectors - updated_edge_vectors
        
        
        # # Distribute the displacement equally among the connected nodes
        # displacement = displacement / 2
        # predicted_next_velocity[edge_index[0]] += displacement
        # predicted_next_velocity[edge_index[1]] -= displacement
        ################################
        # # Compute the scaling factors to ensure edge lengths do not deviate too much
        # scaling_factors = torch.min(original_edge_lengths / updated_edge_lengths, torch.ones_like(original_edge_lengths))
        # scaling_factors = torch.max(scaling_factors, torch.ones_like(original_edge_lengths))
        
        # Apply scaling factors to the predicted velocities
        # scaling_factors = scaling_factors.unsqueeze(1).repeat(1, 3)  # Repeat for x, y, z dimensions
        # corrected_velocity_displacements = (updated_edge_vectors * scaling_factors).reshape(-1, 3)
        # predicted_next_velocity[edge_index.T[:, 1]] = corrected_velocity_displacements
        # predicted_next_velocity[edge_index.T[:, 0]] -= corrected_velocity_displacements
        #############################################
        if real_world:

        
            # Detach and set requires_grad to True for optimization
            predicted_next_velocity_optim = predicted_next_velocity.detach().clone().requires_grad_(True)

            # Define the optimizer
            optimizer = optim.Adam([predicted_next_velocity_optim], lr=1e-3)

            # Optimization loop
            for _ in range(10):
                optimizer.zero_grad()  # Reset gradients

                # Compute updated node coordinates
                updated_node_coords = current_node_coords + predicted_next_velocity_optim

                # Compute updated edge vectors and lengths
                updated_edge_vectors = updated_node_coords[edge_index[0]] - updated_node_coords[edge_index[1]]
                updated_edge_lengths = torch.norm(updated_edge_vectors, dim=1)

                # Compute the regularization term based on the deviation from the original edge lengths
                length_deviation = updated_edge_lengths - original_edge_lengths
                length_deviation[grasped_particle] *= 0
                regularization_term = torch.sum(length_deviation ** 2)


                # Backpropagation
                regularization_term.backward(retain_graph=True)

                # Check gradients
                # print(f"Gradient norm: {predicted_next_velocity_optim.grad.norm()}")

                optimizer.step()
                
                # predicted_next_velocity_optim[grasped_particle] = current_action

            # Apply the optimized velocities
            predicted_next_velocity = predicted_next_velocity_optim.detach()
            predicted_next_velocity[grasped_particle] = current_action

        ########################################################################################################


        predictions.append(predicted_next_velocity)
        
        # here we want to add a refularization to the updated position not to get the original norms of the edges to deviate too much from the original ones
        #(add code here to do a soft clip the predicted velocities to the original norms of the edges)
        
        current_node_coords += predicted_next_velocity
        

        # Update current position for the next prediction
        current_velocities[:input_sequence_length-1] = current_velocities[1:]
        current_velocities[-1] = predicted_next_velocity.to(device)
        
    print(f"Time for rollout: {time.time() - time__now}")
<<<<<<< HEAD
    # Prediction with shape (time, nnodes, dim)
=======
    # Prediction with shape (time, nnodes, dim)    predictions = torch.stack(predictions)
>>>>>>> 9b63d7a (Commit minor changes)
    predictions = torch.stack(predictions)
    
    loss = (predictions- ground_truth_velocities.to(device)) ** 2
    # loss_dumb_prediction = ((velocities[:-input_sequence_length].clone().to(device) - ground_truth_positions.to(device)) ** 2).mean().cpu().numpy()
    # print(f'Loss: {loss.mean()}, dumb loss: {loss_dumb_prediction}')

    output_dict = {
        'initial_pos': node_coords[0].cpu().numpy(),
        'predicted_rollout': predictions.detach().cpu().numpy(),
        'ground_truth_rollout': ground_truth_velocities.cpu().numpy(),
        'node_coords': node_coords.cpu().numpy(),
        'node_types': node_types.cpu().numpy(),
        'edge_index': edge_index.cpu().numpy(),
        'faces': faces.cpu().numpy(),
        'dt': dt,
        'mean_loss': loss.mean().detach().cpu().numpy()
    }

    return output_dict

def validate(simulator, ds, device, FLAGS, use_wandb=False, future=10):
    nsteps = -1
    while nsteps < future:
        idx = random.randint(0, len(ds.dataset)-1)
        features = ds.dataset.__get_val_item__(idx=idx, future=future)
        nsteps = len(features['actions']) 
    with torch.no_grad():
        prediction_data = rollout(simulator, ds, features, nsteps, device, FLAGS.input_sequence_length, FLAGS.dt)
        images = []
        losses = [0]
        for i in range(len(prediction_data['predicted_rollout'])):
            gt_points = prediction_data['node_coords'][i+1]
            pred_points = prediction_data['initial_pos'] + prediction_data['predicted_rollout'][:i+1].sum(axis=0)
            edges = prediction_data['edge_index'].T
            loss =  (gt_points - pred_points) ** 2
            losses.append(loss.mean().item())
            # get figure and load it in wandb
            image = plot_mesh_predictions(gt_points, pred_points, edges, center_plot=None, white_bkg=False, 
                                        save_fig=False, return_image=use_wandb, file_name='mesh.png', azim=-30, elev=20)
            images.append(image)

        # plt.imshow(image)
        # plt.show()
        
        losses_im = plot_losses(losses, return_image=True)
        
        if use_wandb:
            return images, losses_im
            
def update_prediction(velocity_noise, velocity, pred_acc, init_position, edge_index, old_particle_actions, particle_actions, device, input_sequence_length):
    if velocity_noise is not None:
        velocity = velocity + velocity_noise
    new_vel = velocity[:, -3:] + pred_acc
    
    # reset the known vel to the original instead of the predicted
    new_vel[old_particle_actions!= 0] = old_particle_actions[old_particle_actions!= 0].to(torch.float64)
    
  
    # update all the particles except the grasped particles (already there) by adding predicted velocity
    new_pos = init_position.clone()  
    new_pos[particle_actions == 0] += new_vel[particle_actions == 0]
    
    # add the action to the grasped particles
    new_pos += particle_actions    
    
    # compute new edge features
    # displacement, norm = compute_edge_features(new_pos, edge_index)
    temp_graph  =  Data(
            edge_index=edge_index,
            pos=new_pos,
            )
    temp_graph = edge_transformer(temp_graph)
    edge_features = temp_graph.edge_attr
    
    # add action to the conditioning
    # OLD CODE, probably not working
    # new_action_vel = particle_actions + particle_actions  # to handle multiple steps[i]                
    # velocity[:, :-3] = velocity[:, 3:]
    # velocity[:, -3:] = new_action_vel.to(device)
 
    # sift the velocity and add action to the conditioning  
    new_action_vel = copy.deepcopy(velocity[:, -3:]).to(torch.float64)     #+ particle_actions   
    new_action_vel[particle_actions!= 0] = particle_actions[particle_actions!= 0].to(torch.float64) 
    velocity[:, :-3] = velocity[:, 3:]
    velocity[:, -3:] = new_action_vel.to(device)
    
    return velocity, edge_features, new_pos

def train(simulator, device, FLAGS):

    print(f"device = {device}")

    input_sequence_length = FLAGS.input_sequence_length
    noise_std = FLAGS.noise_std
    node_type_embedding_size = FLAGS.node_type_embedding_size
    dt = FLAGS.dt
    knn = FLAGS.knn
    delaunay = True*FLAGS.delaunay
    subsample = True*FLAGS.subsample
    num_samples = FLAGS.num_samples

    lr_init = FLAGS.lr_init
    lr_decay_rate = FLAGS.lr_decay_rate
    lr_decay_steps = FLAGS.lr_decay_steps
    loss_report_step = FLAGS.loss_report_step

    # Initiate training.
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_init)
    step = 0

    # initialize wandb
    # set logging variable as off
    wandb_mode = 'dryrun' if FLAGS.wandb == 0 else 'run'
    os.environ['WANDB_MODE'] = wandb_mode
    # set experiment name
    # exp_name = f"Fig_test_val_berzelius{FLAGS.berzelius}"

    # exp_name = f"cloth-splatting-SIM-curr{FLAGS.curriculum}-astep{FLAGS.action_steps}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-history{FLAGS.input_sequence_length}-batch{FLAGS.batch_size}"
    prefix = 'REB_'
    exp_name = f"{prefix}cloth-splatting-SIM-curr{FLAGS.curriculum}-astep{FLAGS.action_steps}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-nodes{FLAGS.num_samples}"
    wandb.init(project="cloth-splatting", config=FLAGS, name=exp_name)
    
    print(f"Experimen name: {exp_name}")
    flag_values = FLAGS.flag_values_dict()
    for param_name in ['curriculum', 'action_steps', 'message_passing', 'noise_std', 'input_sequence_length', 'batch_size']:
        print(f"{param_name}: {flag_values[param_name]}")

    if FLAGS.berzelius:
        FLAGS.data_path = FLAGS.data_path.replace('.', '/proj/berzelius-2023-364/data/cloth_splatting', 1)
        FLAGS.data_val_path = FLAGS.data_val_path.replace('.', '/proj/berzelius-2023-364/data/cloth_splatting', 1)

    # Set model and its path to save, and load model.
    # If model_path does not exist create new directory and begin training.
    model_path = os.path.join(FLAGS.model_path, exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # # If model_path does exist and model_file and train_state_file exist continue training.
    # if FLAGS.model_file is not None:

    #     if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
    #         # find the latest model, assumes model and train_state files are in step.
    #         fnames = glob.glob(f"{model_path}*model*pt")
    #         max_model_number = 0
    #         expr = re.compile(".*model-(\d+).pt")
    #         for fname in fnames:
    #             model_num = int(expr.search(fname).groups()[0])
    #             if model_num > max_model_number:
    #                 max_model_number = model_num
    #         # reset names to point to the latest.
    #         FLAGS.model_file = f"model-{max_model_number}.pt"
    #         FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

    #     if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
    #         # load model
    #         simulator.load(model_path + FLAGS.model_file)

    #         # load train state
    #         train_state = torch.load(model_path + FLAGS.train_state_file)
    #         # set optimizer state
    #         optimizer = torch.optim.Adam(simulator.parameters())
    #         optimizer.load_state_dict(train_state["optimizer_state"])
    #         optimizer_to(optimizer, device)
    #         # set global train state
    #         step = train_state["global_train_state"].pop("step")
    #     else:
    #         raise FileNotFoundError(
    #             f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found.")

    simulator.train()
    simulator.to(device)

    wandb.watch(simulator, log="gradients", log_freq=10)

    # Load data
    ds = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}',
                                                FLAGS=FLAGS,
                                                # path=f'{FLAGS.data_path}/{FLAGS.mode}.npz',
                                                input_length_sequence=input_sequence_length,
                                                dt=dt,
                                                knn=knn,
                                                delaunay=delaunay,
                                                subsample=subsample,
                                                num_samples=num_samples,
                                                batch_size=FLAGS.batch_size)
    
        # Load data
    ds_val = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_val_path}',
                                                FLAGS=FLAGS,
                                                # path=f'{FLAGS.data_path}/{FLAGS.mode}.npz',
                                                input_length_sequence=input_sequence_length,
                                                dt=dt,
                                                knn=knn,
                                                delaunay=delaunay,
                                                subsample=subsample,
                                                num_samples=num_samples,
                                                batch_size=FLAGS.batch_size)

    not_reached_nsteps = True
    # plot_mesh(ds.dataset._data[0]['pos'][0], ds.dataset._data[0]['edge_index'][0].T)
    try:
        # Initialize the progress bar
        pbar = tqdm(total=FLAGS.ntraining_steps, desc="Loss: N/A")
        
        # Increase predictions steps for curriculum learning
        for step in range(FLAGS.ntraining_steps):
            if FLAGS.curriculum:
                if 0.33 < step/FLAGS.ntraining_steps < 0.66  and  FLAGS.future_sequence_length != 2:
                    FLAGS.future_sequence_length = 2
                    for data_set in [ds, ds_val]:
                        data_set.dataset._future_sequence_length = 2
                        data_set.dataset._compute_cumulative_lengths()
                if 0.66 < step/FLAGS.ntraining_steps and  FLAGS.future_sequence_length != 3:
                    FLAGS.future_sequence_length = 3
                    for data_set in [ds, ds_val]:
                        data_set.dataset._future_sequence_length = 3
                        data_set.dataset._compute_cumulative_lengths()
                    
            for i, graph in enumerate(ds):
                # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                # graph = transformer(graph.to(device))
                graph = graph.to(device)
                graph = transformer(graph)

                # Get inputs
                velocity, node_types, edge_index, edge_features, target_velocities, particle_actions, positions = ds.dataset._graph_to_data(graph)
                init_position = positions
                loss = 0
                for f in range(FLAGS.future_sequence_length):
                    # Get velocity noise, add it only to the first iteration otherwise it accumulates 
                    velocity_noise = None
                    if f == 0:
                        velocity_noise = get_velocity_noise(graph, noise_std=noise_std, input_sequence_length=input_sequence_length, device=device)
                    
                    # Predict dynamics
                    pred_acc, target_acc = simulator.predict_acceleration(
                        velocity,
                        node_types,
                        edge_index,
                        edge_features,
                        target_velocities[:, f],      # traget_velocities[i]
                        velocity_noise=velocity_noise)

                    # Compute loss
                    # mask = torch.logical_or(node_types == NodeType.CLOTH, node_types == NodeType.OUTFLOW)
                    # mask = node_types == NodeType.CLOTH
                    errors = ((pred_acc - target_acc)**2)#[mask]  # only compute errors if node_types is NORMAL or OUTFLOW
                    loss += torch.mean(errors)
                    
                    
                    # Update states with predictions for the next step
                    if FLAGS.future_sequence_length > 1 and f < FLAGS.future_sequence_length - 1:
                        unnormalized_predicted_accelerations = simulator._output_normalizer.inverse(pred_acc)
                        velocity, edge_features, init_position = update_prediction(velocity_noise, velocity, unnormalized_predicted_accelerations, init_position, edge_index, particle_actions[:, f], particle_actions[:, f+1], device, input_sequence_length)
                
                
                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   
                
                wandb_dict = {"Train/loss": loss, "Train/opt_step": (i+1)*(step+1),}

                if i == len(ds)-1:
                    wandb_dict.update({"Train/epoch_loss": loss, "Train/epoch": step, })
                    val_images, plot_losses = validate(simulator, ds_val, device, FLAGS, use_wandb=True, future=np.asarray(ds.dataset._data_lengths).min()-1)
                    # Validation
                    image_losses = wandb.Image(plot_losses, caption=f"Losses at setp {step}")
                    wandb_dict.update({f"Im/Loss/loss_plot": image_losses})
                    for s in range(1, len(val_images)) :
                        val_image= val_images[s-1]
                        image = wandb.Image(val_image, caption=f"Validation at setp {s}")
                        wandb_dict.update({f"Im/Mesh/val_{s}": image})                   


                # Log metrics to wandb
                wandb.log(wandb_dict)

                # Update learning rate
                lr_new = lr_init * (lr_decay_rate ** (step / lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new


                # Save model state
                if step % FLAGS.nsave_steps == 0:
                    simulator.save(model_path + '/model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                    torch.save(train_state, f"{model_path}train_state-{step}.pt")


            pbar.set_description(f"Loss: {loss.item():.7f}")
            pbar.update()

    except KeyboardInterrupt:
        pass

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load simulator
    simulator = ClothMeshSimulator(
        simulation_dimensions=3,
        nnode_in=(2 + 3*FLAGS.input_sequence_length),                  # node (2) type, vel (3) and action (3)
        nedge_in=4,                  # relative positions of node i,j (3) edge norm (1)
        latent_dim=128,
        nmessage_passing_steps=FLAGS.message_passing,      # number of message passing steps, start low, default 15
        nmlp_layers=2,
        mlp_hidden_dim=128,
        nnode_types=2,              # number of different particle types
        node_type_embedding_size=2,     # this is one hot encoding for the type, so it is 1 as far as we have 1 type
        normalize=FLAGS.normalize,
        device=device)


    if FLAGS.mode == 'train':
        train(simulator, device, FLAGS)
    elif FLAGS.mode in ['valid', 'rollout']:
        predict(simulator, device, FLAGS)

    print()


if __name__=='__main__':
    # TRAIN FLAGS - uncomment to train the network
    flags.DEFINE_enum(
        'mode', 'train', ['train', 'valid', 'rollout'],
        help='Train model, validation or rollout evaluation.')
    flags.DEFINE_string('model_file', None,
                        help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    # flags.DEFINE_string('data_path', './sim_datasets/train_dataset_0415/TOWEL', help='The dataset directory.')

    # ROLLOUT FLAGS  - Uncomment if testing
    # flags.DEFINE_enum(
    #     'mode', 'rollout', ['train', 'valid', 'rollout'],
    #     help='Train model, validation or rollout evaluation.')
    # flags.DEFINE_string('model_file', 'model-1950.pt', #'model-2000.pt'
    #                     help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    # flags.DEFINE_string('data_path', './sim_datasets/test_dataset/TOWEL', help='The dataset directory.')
    # flags.DEFINE_string('data_val_path', './sim_datasets/test_dataset/TOWEL', help='The dataset directory.')
    
    
    # flags.DEFINE_string('data_path', './sim_datasets/train_dataset_0415/TOWEL', help='The dataset directory.')
    # flags.DEFINE_string('data_val_path', './sim_datasets/test_dataset_0415/TOWEL', help='The dataset directory.')
    
    flags.DEFINE_string('data_path', './sim_datasets/train_dataset_0702/TOWEL', help='The dataset directory.')
    flags.DEFINE_string('data_val_path', './sim_datasets/test_dataset_0702/TOWEL', help='The dataset directory.')
    
    flags.DEFINE_integer('berzelius', 0, help='Whether it is running on berzelius or not.')
    flags.DEFINE_integer('wandb', 1, help='Whether it is using wandb to log or not.')


    flags.DEFINE_integer('batch_size', 32, help='The batch size.')
    flags.DEFINE_string('data_name', 'final_scene_1_gt_eval', help='Name of the dataset file.')
    flags.DEFINE_string('model_path', "data/model_checkpoint_sim/", help=('The path for saving checkpoints of the model.'))
    flags.DEFINE_string('output_path', "data/rollouts_pos_sim/", help='The path for saving outputs (e.g. rollouts).')

    flags.DEFINE_string('train_state_file', None, help=(
        'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    flags.DEFINE_integer("cuda_device_number", None,
                         help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
    flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
    flags.DEFINE_integer('ntraining_steps', int(3E2), help='Number of training steps.')
    flags.DEFINE_integer('nsave_steps', int(10), help='Number of steps at which to save the model.')

    # Model parameters and training details
    flags.DEFINE_integer('input_sequence_length', int(2), help='Lenght of the sequence in input, default 1.')
<<<<<<< HEAD
    flags.DEFINE_integer('future_sequence_length', int(1), help='Lenght of the sequence in input, default 1.')
=======
    flags.DEFINE_integer('future_sequence_length', int(3), help='Lenght of the sequence in input, default 1.')
>>>>>>> 9b63d7a (Commit minor changes)
    flags.DEFINE_integer('curriculum', int(0), help='Whether to use curriculum learning or not, wehre curriculum is the # of future steps to predict.')
   
    flags.DEFINE_integer('action_steps', int(1), help='Number of actions to predict. Default 1.')
    flags.DEFINE_integer('message_passing', int(15), help='Number of message passing steps. Default 15')
    flags.DEFINE_float('noise_std', float(0), help='Noise standard deviation.')
    flags.DEFINE_integer('node_type_embedding_size', int(1), help='Number of different types of nodes. So far only 1.')
    flags.DEFINE_float('dt', float(1.), help='Simulator delta time.')
    flags.DEFINE_float('lr_init', float(3e-4), help='Initial learning rate.')
    flags.DEFINE_float('lr_decay_rate', float(0.1), help='Decay of the learning rate.')
    flags.DEFINE_integer('lr_decay_steps', int(3E2), help='Steps decay.')
    flags.DEFINE_integer('loss_report_step', int(1), help='Number of steps at which to report the loss.')
    flags.DEFINE_integer('normalize', int(1), help='Whether to use data normalization or not.')

    # Data Processing
    flags.DEFINE_integer('knn', int(10), help='Number of neighbor to construct the graph.')
    flags.DEFINE_integer('delaunay', int(1), help='Whether to use delaunay to traingulation or not.')
    flags.DEFINE_integer('subsample', int(1), help='Whether to subsample or not the initial set of points.')
    flags.DEFINE_integer('num_samples', int(200), help='Number of points to subsample. Default 300')

    FLAGS = flags.FLAGS

    app.run(main)


    print()