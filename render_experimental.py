#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene_reconstruction.scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams, MeshnetParams
from scene_reconstruction.gaussian_mesh import GaussianMesh, MultiGaussianMesh
from meshnet.meshnet_network import ResidualMeshSimulator
from time import time
import glob 
import matplotlib.pyplot as plt
import seaborn as sns

tonumpy = lambda x : x.cpu().numpy()
to8 = lambda x : np.uint8(np.clip(x,0,1)*255)

def merge_deform_logs(folder, track_vertices=False):
    npz_files = glob.glob(os.path.join(folder,'log_deform_*.npz'),recursive=True)
    # sort based on the float number in the file name
    npz_files.sort(key=lambda f: float(f.split('/')[-1].replace('log_deform_','').replace('.npz','')))
    times = [float(''.join(filter(str.isdigit, os.path.basename(f)) )) for f in npz_files]
    trajs = []
    rotations = []
    for npz_file in npz_files:
        deforms_data = np.load(npz_file)
        if track_vertices:
            xyzs_deformed = deforms_data['vertice_deform']
            rotations.append(deforms_data['vertice_rotations'])
        else:
            xyzs_deformed = deforms_data['means3D_deform']
            rotations.append(deforms_data['rotations'])
        trajs.append(xyzs_deformed)


    trajs = np.stack(trajs)
    rotations = np.stack(rotations)
    
    np.savez(os.path.join(folder,'all_trajs.npz'),traj=trajs,rotations=rotations)
    print("saved all trajs to {}".format(os.path.join(folder,'all_trajs.npz')))
    print("shape of all trajs: {}".format(trajs.shape))
    


def visualize(depth):
    # subfig 
    ax = plt.subplot(1,2,1)
    ax.imshow(depth[0])
    # ax.scatter(projections[:,0],projections[:,1],s=1,c='r')
    # plot the points that made the cutoff
    # ax.scatter(visible_projections[depth_mask_visible,0],visible_projections[depth_mask_visible,1],s=5,c='b')
    # add cbar to ax
    cbar = plt.colorbar(ax.images[0],ax=ax)
    # depth_map_gaussians = np.zeros_like(depth[0])
    # depth_map_gaussians[visible_projections[:,1].astype(np.int),visible_projections[:,0].astype(np.int)] = gaussian_dists
    # ax2 = plt.subplot(1,2,2)
    # ax2.imshow(depth_map_gaussians)
    # cbar = plt.colorbar(ax2.images[0],ax=ax2)
    plt.show()

def project(means3D_deform,viewpoint_camera):
     # projecting to cam frame for later use in optic flow
    means3D_deform = torch.tensor(means3D_deform,device='cuda',dtype=torch.float32)
    means_deform_h = torch.cat([means3D_deform,torch.ones_like(means3D_deform[:,0:1])],dim=1).T 
    cam_transform = viewpoint_camera.full_proj_transform.to(means_deform_h.device).T

    projections = cam_transform.matmul(means_deform_h)
    projections = projections/projections[3,:]

    projections = projections[:2].T
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    projections_cam = torch.zeros_like(projections).to(projections.device)
    projections_cam[:,0] = ((projections[:,0] + 1.0) * W - 1.0) * 0.5
    projections_cam[:,1] = ((projections[:,1] + 1.0) * H - 1.0) * 0.5
    return projections_cam


def get_mask(projections=None,gaussian_positions=None,depth=None,cam_center=None,height=800,width=800,depth_threshold=0.2):
    if depth.ndim == 3:
        depth = depth[0]


    # assert none 
    assert projections is not None
    assert gaussian_positions is not None
    assert depth is not None
    assert cam_center is not None
    
    # get the visible projections
    mask_in_image = (projections[:,0] >= 0) & (projections[:,0] < height) & (projections[:,1] >= 0) & \
            (projections[:,1] < width)
    
    depth_mask = np.ones_like(mask_in_image, dtype=bool)
    
    visible_projections = projections[mask_in_image]
    visible_gaussian_positions = gaussian_positions[mask_in_image]

    # get the occlosion mask
    visible_depth = depth[visible_projections[:,1].astype(int), visible_projections[:,0].astype(int)]
    gaussian_dists = np.linalg.norm(visible_gaussian_positions - cam_center,axis=-1)

    depth_mask[mask_in_image] = (gaussian_dists - depth_threshold) <= visible_depth

    return depth_mask & mask_in_image , mask_in_image

def find_closest_gauss(gt,gauss):
    # gt : N x 3 : numpy array
    # gauss : M x 3 : numpy array
    # return : N x 1
    # for each gt point, find the closest gauss point
    # return shape N x 1 
    gt = torch.tensor(gt,device='cuda',dtype=torch.float32)
    gauss = torch.tensor(gauss,device='cuda',dtype=torch.float32)
    gt = gt.unsqueeze(0).repeat(gauss.shape[0],1,1)
    gauss = gauss.unsqueeze(1).repeat(1,gt.shape[1],1)
    dists = torch.norm(gt-gauss,dim=-1)
    return torch.argmin(dists,dim=0).cpu().numpy()

def render_set(model_path, name, iteration, views, gaussians: GaussianMesh, simulator: ResidualMeshSimulator,
               pipeline, background,log_deform=False, track_vertices=False, args=None, gt=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    video_imgs = []
    save_imgs = []
    gt_list = []
    render_list = []
    
    all_times = [view.time for view in views]
    n_points = gaussians.mesh.pos.shape[0] if track_vertices else gaussians._xyz.shape[0]
    todo_times = np.unique(all_times)
    n_times = len(todo_times)
    # colors = colormap[np.arange(n_gaussians) % len(colormap)]
    colors = sns.color_palette(n_colors=n_points)
    prev_projections = None
    current_projections = None 
    prev_visible = None
    
    all_trajs = None
    all_times = None

    prev_mask = None
    prev_time = 0.0

    view_id = views[0].view_id
    time_id = views[0].time_id

    arrow_color = (0,255,0)
    arrow_tickness = 1
    raddii_threshold = 0
    opacity_threshold = -10e10 # disabling this effectively
    depth_dist_threshold = 1.0
    
    opacities = None
    opacity_mask = None 
    gt_idxs = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        log_deform_path = None

        view_time = view.time
                
        if prev_projections is None:
            traj_img = np.zeros((view.image_height,view.image_width,3))

        if log_deform and view_time in todo_times:
            log_deform_path = os.path.join(model_path, name, "ours_{}".format(iteration), "log_deform_{}".format(view.time))

            # remove time from todo_times
            todo_times = todo_times[todo_times != view_time]
        
        view.image_height = int(view.image_height * args.scale)
        view.image_width = int(view.image_width * args.scale)

        render_pkg = render(view, gaussians, simulator,
                            pipeline, background, log_deform_path=log_deform_path, no_shadow=args.no_shadow,
                            project_vertices=track_vertices)
        rendering = tonumpy(render_pkg["render"]).transpose(1, 2, 0)

        if opacities is None:
            opacities = render_pkg["opacities"].to("cpu").numpy()
            opacity_mask = opacities > opacity_threshold
        
            
        
        depth = render_pkg["depth"].to("cpu").numpy()
            
        depth[depth < depth_dist_threshold] = 10e3  # set small depth to a large value for visualization purposes

        dict_key_deform = 'vertice_deform' if track_vertices else 'means3D_deform'
        dict_key_projections = 'vertice_projections' if track_vertices else 'projections'
        if gt_idxs is None:
            if gt is not None:
                gt_t0 = gt[0]
                gt_idxs = find_closest_gauss(gt_t0,render_pkg[dict_key_deform].cpu().numpy())
            else:
                gt_idxs = np.arange(n_points)
        
        if all_trajs is None:
            all_times = np.array([view_time])
            all_trajs = render_pkg[dict_key_deform][gt_idxs].unsqueeze(0).cpu().numpy()
        else:
            all_times = np.concatenate((all_times,np.array([view_time])),axis=0)
            all_trajs = np.concatenate((all_trajs,render_pkg[dict_key_deform][gt_idxs].unsqueeze(0).cpu().numpy()),axis=0)
        
        
                
        if args.show_flow:
            traj_img = np.zeros((view.image_height,view.image_width,3))
            current_projections = render_pkg[dict_key_projections].to("cpu").numpy()[gt_idxs]
            gaussian_positions = render_pkg[dict_key_deform].cpu().numpy()[gt_idxs]
            cam_center = view.camera_center.cpu().numpy()
            current_mask, image_mask = get_mask(projections=current_projections,gaussian_positions=gaussian_positions,depth=depth,cam_center=cam_center,
            height=view.image_height,width=view.image_width)

            rendering =  np.ascontiguousarray(rendering)   
            # show scatter on the currently visible gaussians
            for i in range(n_points)[::args.flow_skip]:
                if current_mask[i] and opacity_mask[i]:
                    color_idx = (i//args.flow_skip) % len(colors)
                    cv2.circle(rendering,(int(current_projections[i,0]),int(current_projections[i,1])),2,colors[color_idx],-1)
                    # rendering[int(current_projections[i,0]),int(current_projections[i,1]),:] = colors[color_idx]

            if view_id != view.view_id:
                prev_projections = None
                all_trajs = None
                traj_img = np.zeros((view.image_height,view.image_width,3))
            else:
                if all_trajs.shape[0] > 1:
                    # draw flow at previous frame
                    traj_img = np.ascontiguousarray(np.zeros((view.image_height,view.image_width,3)))
                    
                    if args.tracking_window > 0:
                        if args.tracking_window < all_trajs.shape[0]:
                            all_trajs = all_trajs[-args.tracking_window:]
                            all_times = all_times[-args.tracking_window:]

                    for j in range(all_trajs.shape[0]-1):

                        prev_gaussians = all_trajs[j]
                        prev_projections = project(all_trajs[j],view).cpu().numpy()
                        prev_time = all_times[j]
                        
                        current_gaussians = all_trajs[j+1]
                        current_projections = project(all_trajs[j+1],view).cpu().numpy()
                        current_time = all_times[j+1]
                        
                        prev_mask, _ = get_mask(projections=prev_projections,gaussian_positions=prev_gaussians,depth=depth,cam_center=cam_center,
                        height=view.image_height,width=view.image_width)
                        current_mask, _ = get_mask(projections=current_projections,gaussian_positions=current_gaussians,depth=depth,cam_center=cam_center,
                        height=view.image_height,width=view.image_width)

                        if current_time <= view_time and prev_time <= view_time:
                            for i in range(current_projections.shape[0])[::args.flow_skip]:
                                # draw arrow from prev_projections to current_projections
                                color_idx = (i//args.flow_skip) % len(colors)
                                if prev_mask[i] and opacity_mask[i]:
                                    #traj_img = cv2.arrowedLine(traj_img,(int(prev_projections[i,0]),int(prev_projections[i,1])),(int(current_projections[i,0]),int(current_projections[i,1])),colors[color_idx],arrow_tickness)
                                    # draw teh same but a line
                                    traj_img = cv2.arrowedLine(traj_img,(int(prev_projections[i,0]),int(prev_projections[i,1])),(int(current_projections[i,0]),int(current_projections[i,1])),colors[color_idx],arrow_tickness)
                rendering[traj_img > 0] = traj_img[traj_img > 0]
                prev_projections = current_projections
                prev_mask = current_mask
                prev_time = view_time
            view_id = view.view_id
            
        
        render_list.append(rendering)

        if name in ["train", "test"]:
            gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            gt_list.append(gt)

    video_imgs = [to8(img) for img in render_list]
    video_gt_imgs = [to8(img.detach().cpu().numpy().transpose(1, 2, 0)) for img in gt_list]
    save_imgs = [torch.tensor((img.transpose(2,0,1)),device="cpu") for img in render_list ]

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
    count = 0
    print("writing rendering images.")
    if len(save_imgs) != 0:
        for image in tqdm(save_imgs):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), video_imgs, fps=30, quality=8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb_gt.mp4'), video_gt_imgs, fps=30, quality=8)


def render_sets(dataset: ModelParams, hyperparam, iteration: int, pipeline: PipelineParams, meshnet_params: MeshnetParams,
                skip_train: bool, skip_test: bool, skip_video: bool, log_deform=False, track_vertices=False,
                user_args=None):
    gt_path = os.path.join(dataset.source_path, "gt.npz")
    gt = None
    if os.path.exists(gt_path):
        gt = np.load(gt_path)['traj']
        print("loaded gt from {}".format(gt_path)) 
    with torch.no_grad():
        scene = Scene(dataset, load_iteration=iteration, shuffle=False, user_args=user_args)

        # load simulator
        mesh_pos = torch.concat([mesh.pos.unsqueeze(0) for mesh in scene.mesh_predictions], dim=0)
        simulator = ResidualMeshSimulator(
            mesh_pos, device='cuda')

        gaussians = MultiGaussianMesh(dataset.sh_degree)
        gaussians.load_ply(os.path.join(scene.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter)))

        dataset.model_path = args.model_path

        meshnet_path = meshnet_params.meshnet_path if meshnet_params.meshnet_path != "" else os.path.join(args.model_path, "meshnet")
        if iteration == -1:
            simulator.load(meshnet_path)
        else:
            simulator.load(os.path.join(meshnet_path, f"model-{iteration}.pt"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.train_cameras,
                       gaussians, simulator, pipeline, background, log_deform=log_deform,
                       track_vertices=track_vertices, args=user_args)
        if not skip_test:
            log_folder = os.path.join(args.model_path, "test", "ours_{}".format(scene.loaded_iter))
            delete_previous_deform_logs(log_folder)
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.test_cameras,
                       gaussians, simulator, pipeline, background, log_deform=log_deform,
                       track_vertices=track_vertices, args=user_args)
            if user_args.log_deform:
                merge_deform_logs(log_folder, track_vertices=track_vertices)
        if not skip_video:
            render_set(dataset.model_path, "video", scene.loaded_iter, scene.video_cameras,
                       gaussians, simulator, pipeline, background, log_deform=log_deform,
                       track_vertices=track_vertices, args=user_args)
 
def delete_previous_deform_logs(folder):
    npz_files = glob.glob(os.path.join(folder,'log_deform_*.npz'),recursive=True)
    for npz_file in npz_files:
        os.remove(npz_file)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    meshnet_param = MeshnetParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--time_skip",type=int,default=None)
    parser.add_argument("--view_skip",default=None,type=int)
    parser.add_argument("--log_deform", action="store_true")
    parser.add_argument("--three_steps_batch",type=bool,default=False)
    parser.add_argument("--show_flow",action="store_true")
    parser.add_argument("--flow_skip",type=int,default=1)
    parser.add_argument("--no_shadow",action="store_true")
    parser.add_argument("--scale",type=float,default=1.0)
    parser.add_argument("--single_cam_video",action="store_true",
                        help="Only render from the first camera for the video viz")
    parser.add_argument("--tracking_window",type=int,default=-1)
    parser.add_argument("--track_vertices", action="store_true",
                        help="Track the vertices in the scene, otherwise the gaussians are tracked.")

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args),
                meshnet_param.extract(args), args.skip_train, args.skip_test, args.skip_video,
                log_deform=args.log_deform, track_vertices=args.track_vertices, user_args=args)
