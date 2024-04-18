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
from meshnet.viz import plot_mesh, plot_mesh_and_points, plot_mesh_predictions
from tqdm import tqdm
from absl import flags
from absl import app
import wandb
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# an instance that transforms face-based graph to edge-based graph. Edge features are auto-computed using "Cartesian" and "Distance"
transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


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

def rollout(simulator, ds, features, nsteps, device, input_sequence_length, dt):

    # TODO: adjust this
    node_coords = features['pos'] # (timesteps, nnode, ndims)
    node_vels = features['vel']  # (timesteps, nnode, ndims)
    actions = features['actions']  # (timesteps, nnode, ndims)
    node_types = features['node_type']  # (timesteps, nnode, )
    times = features['time']  # (timesteps, nnode, ndims)
    edge_index = features['edge_index']  # (2, nedges)
    edge_features = features['edge_displacement']  # (nedges, 3)
    edge_displacement = features['edge_norm']  # (nedges, 3)
    faces = features['faces']  # (nfaces, 3)
    grasped_particle = features['grasped_particle']  # (timesteps, nnode, )

    initial_velocities = node_vels[:input_sequence_length]
    ground_truth_velocities = node_vels[input_sequence_length:]

    current_velocities = initial_velocities.to(device)

    predictions = []
    mask = None

    for step in tqdm(range(nsteps), total=nsteps):

        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords = node_coords[step].to(device)
        current_node_type = node_types[step].to(device)
        current_action = actions[step].to(device)
        current_faces = faces[step].to(device)
        # current_time = times[step].to(device)
        current_edge_index = edge_index[step].to(device)
        current_edge_features = edge_features[step].to(device)
        current_edge_displacement = edge_displacement[step].to(device)


        # current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        next_ground_truth_velocities = ground_truth_velocities[step].to(device)
        next_ground_truth_pos = node_coords[step + 1].to(device)
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
        velocities, node_ty, edge_ind, edge_feat, target_vel= ds.dataset._graph_to_data(graph, input_sequence_length)
        predicted_next_velocity = simulator.predict_velocity(
            velocities=velocities,
            node_type=node_ty,
            edge_index=edge_ind,
            edge_features=edge_feat)

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the same for the later timesteps
            # mask = torch.logical_or(current_node_type == NodeType.NORMAL, current_node_type == NodeType.OUTFLOW)
            mask = current_node_type == NodeType.CLOTH
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1).to(device)
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        # predicted_next_velocity[mask] = next_ground_truth_velocities[mask]
        predictions.append(predicted_next_velocity)

        # Update current position for the next prediction
        current_velocities[:input_sequence_length-1] = current_velocities[1:]
        current_velocities[-1] = predicted_next_velocity.to(device)
        

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    loss = (predictions - ground_truth_velocities.to(device)) ** 2
    # loss_dumb_prediction = ((velocities[:-input_sequence_length].clone().to(device) - ground_truth_positions.to(device)) ** 2).mean().cpu().numpy()
    # print(f'Loss: {loss.mean()}, dumb loss: {loss_dumb_prediction}')

    output_dict = {
        'initial_pos': node_coords[0].cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_velocities.cpu().numpy(),
        'node_coords': node_coords.cpu().numpy(),
        'node_types': node_types.cpu().numpy(),
        'edge_index': edge_index.cpu().numpy(),
        'dt': dt,
        'mean_loss': loss.mean().cpu().numpy()
    }

    return output_dict




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
    os.environ['WANDB_MODE'] = 'run' #'dryrun'
    # set experiment name
    exp_name = f"cloth-splatting-ACC-SIM-knn{FLAGS.knn}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-lr{FLAGS.lr_init}-batch{FLAGS.batch_size}"
    wandb.init(project="cloth-splatting", config=FLAGS, name=exp_name)


    # Set model and its path to save, and load model.
    # If model_path does not exist create new directory and begin training.
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # If model_path does exist and model_file and train_state_file exist continue training.
    if FLAGS.model_file is not None:

        if FLAGS.model_file == "latest" and FLAGS.train_state_file == "latest":
            # find the latest model, assumes model and train_state files are in step.
            fnames = glob.glob(f"{model_path}*model*pt")
            max_model_number = 0
            expr = re.compile(".*model-(\d+).pt")
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            # reset names to point to the latest.
            FLAGS.model_file = f"model-{max_model_number}.pt"
            FLAGS.train_state_file = f"train_state-{max_model_number}.pt"

        if os.path.exists(model_path + FLAGS.model_file) and os.path.exists(model_path + FLAGS.train_state_file):
            # load model
            simulator.load(model_path + FLAGS.model_file)

            # load train state
            train_state = torch.load(model_path + FLAGS.train_state_file)
            # set optimizer state
            optimizer = torch.optim.Adam(simulator.parameters())
            optimizer.load_state_dict(train_state["optimizer_state"])
            optimizer_to(optimizer, device)
            # set global train state
            step = train_state["global_train_state"].pop("step")
        else:
            raise FileNotFoundError(
                f"Specified model_file {model_path + FLAGS.model_file} and train_state_file {model_path + FLAGS.train_state_file} not found.")

    simulator.train()
    simulator.to(device)

    wandb.watch(simulator, log="gradients", log_freq=10)

    # Load data
    ds = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}',
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
        for step in range(FLAGS.ntraining_steps):
            for i, graph in enumerate(ds):
                # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                # graph = transformer(graph.to(device))
                graph = graph.to(device)
                graph = transformer(graph)

                # Get inputs
                velocity, node_types, edge_index, edge_features, target_velocities = ds.dataset._graph_to_data(graph)
                # action = graph.x[:, :3]
                # velocity = graph.x[:, 3:6]
                # node_types = graph.x[:, 6].unsqueeze(1)
                # edge_index = graph.edge_index
                # edge_features = graph.edge_attr
                # target_velocities = graph.y

                # TODO: integrate noise
                # position_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
                # plot_mesh_and_points(graph.pos[graph.batch==0].cpu(), edge_index[:].cpu().T, points=graph.pos[graph.batch==0][node_types[graph.batch==0].squeeze() == 1].cpu())

                # Predict dynamics
                pred_acc, target_acc = simulator.predict_acceleration(
                    velocity,
                    node_types,
                    edge_index,
                    edge_features,
                    target_velocities,
                    velocity_noise=None)

                # Compute loss
                # mask = torch.logical_or(node_types == NodeType.CLOTH, node_types == NodeType.OUTFLOW)
                mask = node_types == NodeType.CLOTH
                errors = ((pred_acc - target_acc)**2)#[mask]  # only compute errors if node_types is NORMAL or OUTFLOW
                loss = torch.mean(errors)

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log metrics to wandb
                wandb.log({"loss": loss, "step": step})

                # Update learning rate
                # TODO: Integrate learning rate decay
                lr_new = lr_init * (lr_decay_rate ** (step / lr_decay_steps)) + 1e-6
                for param in optimizer.param_groups:
                    param['lr'] = lr_new

                # if step % loss_report_step == 0:
                #     print(f"Training step: {step}/{FLAGS.ntraining_steps}. Loss: {loss}.")

                # Save model state
                if step % FLAGS.nsave_steps == 0:
                    simulator.save(model_path + 'model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                    torch.save(train_state, f"{model_path}train_state-{step}.pt")

                # Complete training
                # if (step >= FLAGS.ntraining_steps):
                #     not_reached_nsteps = False
                #     break

                # step += 1
                # Update the progress bar
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
    flags.DEFINE_string('data_path', '../sim_datasets/train_dataset_0415/TOWEL', help='The dataset directory.')

    # ROLLOUT FLAGS  - Uncomment if testing
    # flags.DEFINE_enum(
    #     'mode', 'rollout', ['train', 'valid', 'rollout'],
    #     help='Train model, validation or rollout evaluation.')
    # flags.DEFINE_string('model_file', 'model-1950.pt', #'model-2000.pt'
    #                     help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    # flags.DEFINE_string('data_path', '../sim_datasets/test_dataset/TOWEL', help='The dataset directory.')


    flags.DEFINE_integer('batch_size', 16, help='The batch size.')
    flags.DEFINE_string('data_name', 'final_scene_1_gt_eval', help='Name of the dataset file.')
    flags.DEFINE_string('model_path', "data/model_pos_checkpoint_sim_0417/", help=('The path for saving checkpoints of the model.'))
    flags.DEFINE_string('output_path', "data/rollouts_pos_sim/", help='The path for saving outputs (e.g. rollouts).')

    flags.DEFINE_string('train_state_file', None, help=(
        'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    flags.DEFINE_integer("cuda_device_number", None,
                         help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
    flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
    flags.DEFINE_integer('ntraining_steps', int(2E3), help='Number of training steps.')
    flags.DEFINE_integer('nsave_steps', int(50), help='Number of steps at which to save the model.')

    # Model parameters and training details
    flags.DEFINE_integer('input_sequence_length', int(2), help='Lenght of the sequence in input, default 1.')
    flags.DEFINE_integer('message_passing', int(15), help='Number of message passing steps. Default 15')
    flags.DEFINE_float('noise_std', float(0), help='Noise standard deviation.')
    flags.DEFINE_integer('node_type_embedding_size', int(1), help='Number of different types of nodes. So far only 1.')
    flags.DEFINE_float('dt', float(1.), help='Simulator delta time.')
    flags.DEFINE_float('lr_init', float(3e-4), help='Initial learning rate.')
    flags.DEFINE_float('lr_decay_rate', float(0.1), help='Decay of the learning rate.')
    flags.DEFINE_integer('lr_decay_steps', int(2E3), help='Steps decay.')
    flags.DEFINE_integer('loss_report_step', int(1), help='Number of steps at which to report the loss.')
    flags.DEFINE_integer('normalize', int(1), help='Whether to use data normalization or not.')

    # Data Processing
    flags.DEFINE_integer('knn', int(10), help='Number of neighbor to construct the graph.')
    flags.DEFINE_integer('delaunay', int(1), help='Whether to use delaunay to traingulation or not.')
    flags.DEFINE_integer('subsample', int(1), help='Whether to subsample or not the initial set of points.')
    flags.DEFINE_integer('num_samples', int(100), help='Number of points to subsample. Default 300')

    FLAGS = flags.FLAGS

    app.run(main)


    print()