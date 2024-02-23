import sys
import os
import glob
import numpy as np
import torch
import torch_geometric.transforms as T
import re
import pickle
from meshnet.meshnet import MeshSimulator
from meshnet.model_utils import optimizer_to, NodeType, datas_to_graph_pos
from meshnet.model_utils import get_velocity_noise
import meshnet.dataloader as data_loader
from tqdm import tqdm
from absl import flags
from absl import app
import wandb

def predict(simulator, device, FLAGS):

    # Load simulator
    if os.path.exists(FLAGS.model_path + FLAGS.model_file):
        simulator.load(FLAGS.model_path + FLAGS.model_file)
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
    ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}/{FLAGS.data_name}.npz",
                                                     knn=FLAGS.knn,
                                                     delaunay=True*FLAGS.delaunay,
                                                     subsample=True*FLAGS.subsample,
                                                     num_samples=FLAGS.num_samples,
                                                     )
    input_sequence_length = FLAGS.input_sequence_length

    # Rollout
    with torch.no_grad():
        for i, features in enumerate(ds):
            nsteps = len(features[0]) - input_sequence_length
            prediction_data = rollout(simulator, features, nsteps, device, FLAGS)
            print(f"Rollout for example{i}: loss = {prediction_data['mean_loss']}")

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                filename = f'{FLAGS.rollout_filename}_{i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(prediction_data, f)

    print(f"Mean loss on rollout prediction: {prediction_data['mean_loss']}")

def rollout(simulator,  features, nsteps, device, FLAGS):
    input_sequence_length = FLAGS.input_sequence_length
    dt = FLAGS.dt

    # TODO: adjust this
    node_coords = features[0]  # (timesteps, nnode, ndims)
    node_types = features[1]  # (timesteps, nnode, )
    times = features[2]  # (timesteps, nnode, ndims)
    edge_index = features[3]  # (2, nedges)
    edge_features = features[4]  # (nedges, 3)
    edge_displacement = features[5]  # (nedges, 3)

    initial_positions = node_coords[0:1]
    ground_truth_positions = node_coords[input_sequence_length:]

    initial_positions = initial_positions.squeeze().to(device)
    predictions = []
    mask = None

    for step in tqdm(range(nsteps), total=nsteps):

        # Predict next velocity
        # First, obtain data to form a graph
        current_node_coords = initial_positions
        current_node_type = node_types[step]
        current_time = times[step]
        current_edge_index = edge_index[step]
        current_edge_features = edge_features[step]
        current_edge_displacement = edge_displacement[step]

        current_time_idx_vector = torch.tensor(np.full(current_node_coords.shape[0], step)).to(torch.float32).contiguous()
        next_ground_truth_pos = ground_truth_positions[step].to(device)
        current_example = (
            (current_node_coords, current_node_type, current_time, current_time_idx_vector,
             current_edge_index, current_edge_features, current_edge_displacement),
            next_ground_truth_pos)

        # Make graph
        graph = datas_to_graph_pos(current_example, dt=dt, device=device)
        # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
        # graph = transformer(graph)

        # Predict next velocity
        predicted_next_position = simulator.predict_position(
            init_positions=graph.pos,
            time_vector=graph.x[:, 1],
            node_type=graph.x[:, 0],
            edge_index=graph.edge_index,
            edge_features=graph.edge_attr)

        # Apply mask.
        if mask is None:  # only compute mask for the first timestep, since it will be the same for the later timesteps
            # mask = torch.logical_or(current_node_type == NodeType.NORMAL, current_node_type == NodeType.OUTFLOW)
            mask = current_node_type == NodeType.CLOTH
            mask = torch.logical_not(mask)
            mask = mask.squeeze(1).to(device)
        # Maintain previous velocity if node_type is not (Normal or Outflow).
        # i.e., only update normal or outflow nodes.
        predicted_next_position[mask] = next_ground_truth_pos[mask]
        predictions.append(predicted_next_position)

        # Update current position for the next prediction
        # current_velocities = predicted_next_velocity.to(device)

    # Prediction with shape (time, nnodes, dim)
    predictions = torch.stack(predictions)

    loss = (predictions - ground_truth_positions.to(device)) ** 2
    # loss_dumb_prediction = ((velocities[:-input_sequence_length].clone().to(device) - ground_truth_positions.to(device)) ** 2).mean().cpu().numpy()
    # print(f'Loss: {loss.mean()}, dumb loss: {loss_dumb_prediction}')

    output_dict = {
        'initial_pos': initial_positions.cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
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
    os.environ['WANDB_MODE'] = 'dryrun'
    # set experiment name
    exp_name = f"cloth-splatting-POS-knn{FLAGS.knn}-propagation{FLAGS.message_passing}-noise{FLAGS.noise_std}-lr{FLAGS.lr_init}-batch{FLAGS.batch_size}"
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
    ds = data_loader.get_data_loader_by_samples(path=f'{FLAGS.data_path}/{FLAGS.data_name}.npz',
                                                # path=f'{FLAGS.data_path}/{FLAGS.mode}.npz',
                                                input_length_sequence=input_sequence_length,
                                                dt=dt,
                                                knn=knn,
                                                delaunay=delaunay,
                                                subsample=subsample,
                                                num_samples=num_samples,
                                                batch_size=FLAGS.batch_size)

    not_reached_nsteps = True
    try:
        while not_reached_nsteps:
            for i, graph in enumerate(ds):
                # Represent graph using edge_index and make edge_feature to be using [relative_distance, norm]
                # graph = transformer(graph.to(device))
                graph = graph.to(device)

                # Get inputs
                node_types = graph.x[:, 0]
                time_vector = graph.x[:, 1]
                init_position = graph.pos
                edge_index = graph.edge_index
                edge_features = graph.edge_attr
                target_positions = graph.y

                # TODO: integrate noise
                position_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)

                # Predict dynamics
                pred_pos, target_pos = simulator.predict_dx(
                    init_position,
                    time_vector,
                    node_types,
                    edge_index,
                    edge_features,
                    target_positions,
                    position_noise)

                # Compute loss
                # mask = torch.logical_or(node_types == NodeType.CLOTH, node_types == NodeType.OUTFLOW)
                mask = node_types == NodeType.CLOTH
                errors = ((pred_pos - target_pos)**2)[mask]  # only compute errors if node_types is NORMAL or OUTFLOW
                loss = torch.mean(errors)

                # Computes the gradient of loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log metrics to wandb
                wandb.log({"loss": loss, "step": step})

                # Update learning rate
                # TODO: Integrate learning rate decay
                # lr_new = lr_init * (lr_decay_rate ** (step / lr_decay_steps)) + 1e-6
                # for param in optimizer.param_groups:
                #     param['lr'] = lr_new

                if step % loss_report_step == 0:
                    print(f"Training step: {step}/{FLAGS.ntraining_steps}. Loss: {loss}.")

                # Save model state
                if step % FLAGS.nsave_steps == 0:
                    simulator.save(model_path + 'model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(), global_train_state={"step": step})
                    torch.save(train_state, f"{model_path}train_state-{step}.pt")

                # Complete training
                if (step >= FLAGS.ntraining_steps):
                    not_reached_nsteps = False
                    break

                step += 1

    except KeyboardInterrupt:
        pass

def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load simulator
    simulator = MeshSimulator(
        simulation_dimensions=3,
        nnode_in=5,                  # node (1) type, position (3) and time (1)
        nedge_in=4,                  # relative positions of node i,j (3) edge norm (1)
        latent_dim=128,
        nmessage_passing_steps=FLAGS.message_passing,      # number of message passing steps, start low, default 15
        nmlp_layers=2,
        mlp_hidden_dim=128,
        nnode_types=1,              # number of different particle types
        node_type_embedding_size=1,     # this is one hot encoding for the type, so it is 1 as far as we have 1 type
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

    # ROLLOUT FLAGS  - Uncomment if testing
    # flags.DEFINE_enum(
    #     'mode', 'rollout', ['train', 'valid', 'rollout'],
    #     help='Train model, validation or rollout evaluation.')
    # flags.DEFINE_string('model_file', 'model-2000.pt',
    #                     help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))


    flags.DEFINE_integer('batch_size', 16, help='The batch size.')
    flags.DEFINE_string('data_path', 'data/final_scenes/scene_1', help='The dataset directory.')
    flags.DEFINE_string('data_name', 'final_scene_1_gt_eval', help='Name of the dataset file.')
    flags.DEFINE_string('model_path', "data/model_pos_checkpoint/", help=('The path for saving checkpoints of the model.'))
    flags.DEFINE_string('output_path', "data/rollouts_pos/", help='The path for saving outputs (e.g. rollouts).')

    flags.DEFINE_string('train_state_file', None, help=(
        'Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
    flags.DEFINE_integer("cuda_device_number", None,
                         help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
    flags.DEFINE_string('rollout_filename', "rollout", help='Name saving the rollout')
    flags.DEFINE_integer('ntraining_steps', int(2E3), help='Number of training steps.')
    flags.DEFINE_integer('nsave_steps', int(50), help='Number of steps at which to save the model.')

    # Model parameters and training details
    flags.DEFINE_integer('input_sequence_length', int(1), help='Lenght of the sequence in input, default 1.')
    flags.DEFINE_integer('message_passing', int(15), help='Number of message passing steps.')
    flags.DEFINE_float('noise_std', float(0), help='Noise standard deviation.')
    flags.DEFINE_integer('node_type_embedding_size', int(1), help='Number of different types of nodes. So far only 1.')
    flags.DEFINE_float('dt', float(1.), help='Simulator delta time.')
    flags.DEFINE_float('lr_init', float(3e-4), help='Initial learning rate.')
    flags.DEFINE_float('lr_decay_rate', float(0.1), help='Decay of the learning rate.')
    flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Steps decay.')
    flags.DEFINE_integer('loss_report_step', int(1), help='Number of steps at which to report the loss.')

    # Data Processing
    flags.DEFINE_integer('knn', int(10), help='Number of neighbor to construct the graph.')
    flags.DEFINE_integer('delaunay', int(1), help='Whether to use delaunay to traingulation or not.')
    flags.DEFINE_integer('subsample', int(1), help='Whether to subsample or not the initial set of points.')
    flags.DEFINE_integer('num_samples', int(300), help='Number of points to subsample.')

    FLAGS = flags.FLAGS

    app.run(main)


    print()