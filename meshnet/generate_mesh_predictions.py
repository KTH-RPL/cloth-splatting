import sys
import os
import glob
import numpy as np
import torch
import torch_geometric.transforms as T
import re
import pickle
from meshnet.cloth_network import ClothMeshSimulator
from meshnet.model_utils import optimizer_to, NodeType, datas_to_graph_pos, datas_to_graph
from meshnet.model_utils import get_velocity_noise
import meshnet.dataloader_sim as data_loader
from meshnet.data_utils import compute_edge_features
from meshnet.viz import plot_mesh, plot_mesh_and_points, plot_mesh_predictions, plot_losses
from tqdm import tqdm
from absl import flags
from absl import app
import wandb
import random
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from train_meshnet_sim import update_prediction, rollout
from moviepy.editor import ImageSequenceClip
import h5py
import torch_geometric

#change matplotlib backend
plt.switch_backend('agg')

# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
edge_transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])

def save_mesh(mesh, path, name="mesh.hdf5"):
    mesh = mesh.to_dict()
    with h5py.File(os.path.join(path, name), "w") as f:
        for key, value in mesh.items():
            f.create_dataset(key, data=value.detach().cpu().numpy())
            
def get_mesh_data(points, faces):
    mesh = torch_geometric.data.Data(pos=points, face=faces)
    mesh = torch_geometric.transforms.FaceToEdge(remove_faces=False)(mesh)
    mesh = torch_geometric.transforms.GenerateMeshNormals()(mesh)

    return mesh


def predict(simulator, device, FLAGS, exp_name, save_gif=False):

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
        
    gif_path = os.path.join(FLAGS.output_path, 'gifs', exp_name, FLAGS.object)
    os.makedirs(gif_path, exist_ok=True)

    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'


    # load trajectories for one specific cloth ID
    ds = data_loader.get_data_loader_by_samples(path=f"{FLAGS.data_path}/{FLAGS.mesh_idx:05}/",
                                                     input_length_sequence=FLAGS.input_sequence_length,
                                                     FLAGS=FLAGS,
                                                     dt=FLAGS.dt,
                                                    batch_size=FLAGS.batch_size,
                                                     knn=FLAGS.knn,
                                                     delaunay=True*FLAGS.delaunay,
                                                     subsample=True*FLAGS.subsample,
                                                     num_samples=FLAGS.num_samples,
                                                     )
    print("Loaded dataset")
    input_sequence_length = FLAGS.input_sequence_length

    # Rollout
    with torch.no_grad():
        t = FLAGS.traj_idx
        features = ds.dataset.__get_val_item__(ds.dataset._precompute_cumlengths[t]-1, future=-1)
        nsteps = len(features['actions'])
        prediction_data = rollout(simulator, ds, features, nsteps, device, FLAGS.input_sequence_length, FLAGS.dt)
        print(f"Rollout for example{t}: loss = {prediction_data['mean_loss']}")
        
        graph_paths_before =f"{FLAGS.data_path}/{FLAGS.mesh_idx:05}/{t:05}/splits"
        graph_paths = f"{FLAGS.data_path}/{FLAGS.mesh_idx:05}/{t:05}/splits/mesh_predictions/"
        os.makedirs(graph_paths, exist_ok=True)
        
        # save starting point
        mesh = get_mesh_data(torch.from_numpy(prediction_data['initial_pos']), torch.from_numpy(prediction_data['faces']).long())
        save_mesh(mesh, graph_paths_before, name="init_mesh.hdf5")
        save_mesh(mesh, graph_paths, name=f"mesh_{0:03}.hdf5")
        images = []

        losses = [0]
        for i in range(len(prediction_data['predicted_rollout'])):
            gt_points = prediction_data['node_coords'][i+1]
            pred_points = prediction_data['initial_pos'] + prediction_data['predicted_rollout'][:i+1].sum(axis=0)
            edges = prediction_data['edge_index'].T
            faces = prediction_data['faces']
            loss =  (gt_points - pred_points) ** 2
            losses.append(loss.mean().item())
            # get figure and load it in wandb
            if save_gif:
                image = plot_mesh_predictions(gt_points, pred_points, edges, center_plot=None, white_bkg=False, 
                                            save_fig=False, return_image=1, file_name='mesh.png', azim=-30, elev=20)
                images.append(image)
            
            mesh = get_mesh_data(torch.from_numpy(pred_points), torch.from_numpy(faces).long())
            save_mesh(mesh, graph_paths, name=f"mesh_{i+1:03}.hdf5")
            
                   
            
        if save_gif:
            gif_file = os.path.join(gif_path, f'AAAAAAAAAAArollout_{t}.gif')
            clip = ImageSequenceClip(images, fps=1)
            clip.write_gif(gif_file, fps=1)
            
            print(f"Saved gif to {gif_file}")
        

        
        # save also images
        # for i, image in enumerate(images):
        #     plt.imsave(os.path.join(gif_path, f'rollout_{t}_{i}.png'), image)
            

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']}")



def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # params = {'curriculum': [0, 1, 0, 0],
    #           'action_steps': [1, 1, 1, 3],
    #           'noise_std': [0.0, 0.0, 0.01, 0.0],}
    # params = {'curriculum': [0, 1, 0, 1, 0, 1],
    #           'action_steps': [1, 1, 1, 1, 1, 1],
    #           'noise_std': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    #           'num_samples': [100, 100, 200, 200, 300, 300]}
    
    params = {'curriculum': [0],
              'action_steps': [1],
              'noise_std': [0.01],
              'num_samples': [200]}
    
    i = 0

    FLAGS.curriculum = params['curriculum'][i]
    FLAGS.action_steps = params['action_steps'][i]
    FLAGS.noise_std = params['noise_std'][i]      
    FLAGS.num_samples = params['num_samples'][i]   
    
        
    model_path_original = FLAGS.model_path
    original_data_path = FLAGS.data_path 
    
    FLAGS.data_path = os.path.join(original_data_path, FLAGS.object)
        

    # exp_name = f"cloth-splatting-SIM-curr{FLAGS.curriculum}-astep{FLAGS.action_steps}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-history{FLAGS.input_sequence_length}-batch{FLAGS.batch_size}"
    exp_name = f"cloth-splatting-SIM-curr{FLAGS.curriculum}-astep{FLAGS.action_steps}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-nodes{FLAGS.num_samples}"

    model_path = os.path.join(model_path_original, exp_name)
    # TODO: change this
    # model_file = f'{exp_name}model-190.pt'
    model_file = f'model-290.pt'
    if FLAGS.action_steps == 3:
        model_file =  f'{exp_name}model-120.pt'
    FLAGS.model_path = model_path + '/'
    FLAGS.model_file = model_file

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


    predict(simulator, device, FLAGS, exp_name)

    print()
            

if __name__=='__main__':  
    flags.DEFINE_enum(
        'mode', 'rollout', ['train', 'valid', 'rollout'],
        help='Train model, validation or rollout evaluation.')
    
    flags.DEFINE_string('model_file', 'cloth-splatting-SIM-curr0-astep3-propagation15-noise0.0-history2-batch32/model-190.pt', #'model-2000.pt'
                        help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

    flags.DEFINE_string('data_path', './sim_datasets/test_dataset_0415/', help='The dataset directory.')
    # flags.DEFINE_string('data_val_path', './sim_datasets/test_dataset_0415/', help='The dataset directory.')
    flags.DEFINE_string('object', 'TOWEL', help='The dataset directory.')
    flags.DEFINE_integer('mesh_idx', 0, help='Which mesh to load for prediction.')
    flags.DEFINE_integer('traj_idx', 0, help='Which trajectory to unroll for predictions.')

    
    flags.DEFINE_integer('berzelius', 0, help='Whether it is running on berzelius or not.')
    flags.DEFINE_integer('wandb', 0, help='Whether it is using wandb to log or not.')



    flags.DEFINE_integer('batch_size', 32, help='The batch size.')
    flags.DEFINE_string('data_name', 'final_scene_1_gt_eval', help='Name of the dataset file.')
    flags.DEFINE_string('model_path', "./data/berzelius/model_checkpoint_sim/", help=('The path for saving checkpoints of the model.'))
    flags.DEFINE_string('output_path', "./data/berzelius/rollouts_pos_sim/", help='The path for saving outputs (e.g. rollouts).')

    flags.DEFINE_string('train_state_file', None, help=(
        'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    flags.DEFINE_integer("cuda_device_number", None,
                         help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
    flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
    flags.DEFINE_integer('ntraining_steps', int(2E2), help='Number of training steps.')
    flags.DEFINE_integer('nsave_steps', int(10), help='Number of steps at which to save the model.')

    # Model parameters and training details
    flags.DEFINE_integer('input_sequence_length', int(2), help='Lenght of the sequence in input, default 1.')
    flags.DEFINE_integer('future_sequence_length', int(1), help='Lenght of the sequence in input, default 1.')
    flags.DEFINE_integer('curriculum', int(0), help='Whether to use curriculum learning or not, wehre curriculum is the # of future steps to predict.')
   
    flags.DEFINE_integer('action_steps', int(3), help='Number of actions to predict. Default 1.')
    flags.DEFINE_integer('message_passing', int(15), help='Number of message passing steps. Default 15')
    flags.DEFINE_float('noise_std', float(0), help='Noise standard deviation.')
    flags.DEFINE_integer('node_type_embedding_size', int(1), help='Number of different types of nodes. So far only 1.')
    flags.DEFINE_float('dt', float(1.), help='Simulator delta time.')
    flags.DEFINE_float('lr_init', float(3e-4), help='Initial learning rate.')
    flags.DEFINE_float('lr_decay_rate', float(0.1), help='Decay of the learning rate.')
    flags.DEFINE_integer('lr_decay_steps', int(2E2), help='Steps decay.')
    flags.DEFINE_integer('loss_report_step', int(1), help='Number of steps at which to report the loss.')
    flags.DEFINE_integer('normalize', int(1), help='Whether to use data normalization or not.')

    # Data Processing
    flags.DEFINE_integer('knn', int(10), help='Number of neighbor to construct the graph.')
    flags.DEFINE_integer('delaunay', int(1), help='Whether to use delaunay to traingulation or not.')
    flags.DEFINE_integer('subsample', int(1), help='Whether to subsample or not the initial set of points.')
    flags.DEFINE_integer('num_samples', int(100), help='Number of points to subsample. Default 300')

    FLAGS = flags.FLAGS

    app.run(main)

