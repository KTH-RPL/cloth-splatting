import sys, os
import json
import bpy
import numpy as np
import argparse
import matplotlib.pyplot as plt 
import shutil
import mathutils
from mathutils import Matrix, Vector
import glob 
import re
import pyroexr 
import copy
import dataclasses
from manipulation.materials.cloth_material import modify_bsdf_to_cloth, hsv_to_rgb, sample_hsv_color, get_image_material
from manipulation.materials.towels import create_gridded_dish_towel_material
from manipulation.materials.common import modify_bsdf_to_cloth

# default vals
DEBUG = False
RESOLUTION = 800
DEPTH_SCALE = 0.2
COLOR_DEPTH = 8
FORMAT = 'png'
extension = '.png'

@dataclasses.dataclass
class ClothMeshConfig:
    mesh_path: str

    solidify: bool = True
    subdivide: bool = True
    xy_randomization_range: float = 0.1
    # mesh_dir: List[str] = dataclasses.field(init=False)

    # def __post_init__(self):
        # TODO: imporve the mesh path
        # mesh_path = "." / pathlib.Path(self.mesh_path)
        # cloth_meshes = os.listdir(mesh_path)
        # cloth_meshes = [mesh_path / mesh for mesh in cloth_meshes]
        # cloth_meshes = [mesh for mesh in cloth_meshes if mesh.suffix == ".obj"]
        # self.mesh_dir = cloth_meshes
        
def load_cloth_mesh(mesh_path, solidify=True, subdivide=True):
    # bpy.ops.import_scene.obj(filepath=mesh_path, split_mode="OFF")  # keep vertex order with split_mode="OFF"
    bpy.ops.wm.obj_import(filepath=mesh_path)
    cloth_object = bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = cloth_object
    cloth_object.select_set(True)
    
        # first solidify, then subdivide. The modifier will then also smooth the edges of the solidified mesh.

    if solidify:
        thickness = np.random.uniform(0.001, 0.003)  # 1-3 mm cloth thickness.
        # note that this has impacts on the visibility of the keypoints
        # as these are now inside the mesh. Need to either test for 1-ring neighbours or make sure that the auxiliary cubes around a vertex
        # in the visibility check are larger than the solidify modifier thickness. The latter is what we do by default, since the rest distance of the cloth meshes
        # is assumed to be > 1cm.
        cloth_object.location[2] += thickness / 2  # offset for solidify modifier
        # solidify the mesh to give the cloth some thickness.
        bpy.ops.object.modifier_add(type="SOLIDIFY")
        # 2 mm, make sure the particle radius of the cloth simulator is larger than this!
        bpy.context.object.modifiers["Solidify"].thickness = thickness
        bpy.context.object.modifiers["Solidify"].offset = 0.0  # center the thickness around the original mesh

        # disable auto-smooth to enable gpu-accelerated subsurface division modifier

    bpy.ops.object.shade_flat()
    bpy.context.object.data.use_auto_smooth = False
    if subdivide:
        #  modifier is more powerful than operator
        # but it is also rather expensive. Make sure it is done on GPU!
        # higher subdivision -> more expensive rendering, so have to find lowest amount that is still good enough
        # also influenced by rendering resolution ofc.

        # bpy.ops.object.modifier_add(type="SUBSURF")
        # bpy.context.object.modifiers["Subdivision"].render_levels = 2
        # bpy.context.object.modifiers["Subdivision"].use_limit_surface = False

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(smoothness=1, number_cuts=1)
        bpy.ops.object.mode_set(mode="OBJECT")
        
    return cloth_object




def load_exr(path):
    exr = pyroexr.load(path)
    shape = exr.channels()['B'].shape
    img = np.zeros((shape[0],shape[1],1))
    img[:,:,0] = exr.channels()['R'] 
    img = img[np.newaxis,...]
    return img

def set_scene_params(scene, args, json_data):
    print('settings params')
    if args.frame_start != -1:
        scene.frame_start = args.frame_start

    if args.frame_end != -1:
        scene.frame_end = args.frame_end

    # Data to store in JSON file
    scene.render.resolution_x = args.res_x
    scene.render.resolution_y = args.res_y
    scene.render.resolution_x = args.res_x
    scene.render.resolution_y = args.res_y


    # set fps 
    scene.render.resolution_percentage = 100
    scene.render.resolution_percentage = 100

    cam = scene.objects['Camera']
    cam.data.angle_x = json_data['camera_angle_x']
    if args.fps is not None:
        bpy.context.scene.render.fps = args.fps
    
    render = scene.render
    if FORMAT == 'exr': 
        render.image_settings.file_format = 'OPEN_EXR'
    elif FORMAT == 'png':
        render.image_settings.file_format = 'PNG'

    render.image_settings.color_depth = str(COLOR_DEPTH)

    # render.engine = 'CYCLES'
    # render.engine = 'BLENDER_EEVEE'
    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = str(COLOR_DEPTH) # ('8', '16')
    render.resolution_percentage = 100
    render.film_transparent = True
    scene.view_layers[0].name = "View Layer"
    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_object_index = True

    bpy.data.scenes[0].render.engine = "CYCLES"
    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"
    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

def setup_depth(nodes,links,render_layers):
    # Create depth output nodes
    bpy.context.scene.view_layers["View Layer"].use_pass_z = True
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.color_depth = str(COLOR_DEPTH)
    depth_file_output.base_path = ''
    
    print('setting up depth')

    if FORMAT == 'exr':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        print('exr format')
    else:
        depth_file_output.format.color_mode = "BW"

        # Remap as other types can not represent the full range of depth.
        map = nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        # map.offset = [-0.7]
        map.size = [DEPTH_SCALE]
        map.use_min = True
        map.min = [0]
        map.use_max = True
        map.max = [255]

        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
    
    return depth_file_output

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def generate_towel_material():
    # background_color = [255, 255, 255]#hsv_to_rgb(sample_hsv_color())
    # vertical_color = [255, 0, 0]#hsv_to_rgb(sample_hsv_color())
    # horizontal_color = [255, 0, 0]#sv_to_rgb(sample_hsv_color())
    # intersection_color = [255, 0, 0]#hsv_to_rgb(sample_hsv_color())
    background_color = [0, 70/255, 162/255]#hsv_to_rgb(sample_hsv_color())
    vertical_color = [255/255, 4/255, 13/255]#hsv_to_rgb(sample_hsv_color())
    horizontal_color = [255/255, 46/255, 1/255]#sv_to_rgb(sample_hsv_color())
    intersection_color = [255/255, 4/255, 1/255]#hsv_to_rgb(sample_hsv_color())

    # rgb to rgba
    background_color = np.array([*background_color, 1])
    vertical_color = np.array([*vertical_color, 1])
    horizontal_color = np.array([*horizontal_color, 1])
    intersection_color = np.array([*intersection_color, 1])

    n_vertical_stripes = 10# np.random.randint(2, 20)
    n_horizontal_stripes = 10 #np.random.randint(2, 20)
    vertical_stripe_relative_width = 0.3 # np.random.uniform(0.05, 0.5)
    horizontal_stripe_relative_width = 0.3# np.random.uniform(0.05, 0.5)

    material = create_gridded_dish_towel_material(
        n_vertical_stripes,
        n_horizontal_stripes,
        vertical_stripe_relative_width,
        horizontal_stripe_relative_width,
        vertical_color,
        horizontal_color,
        intersection_color,
        background_color
    )

    material = modify_bsdf_to_cloth(material)
    return material
 #   if config.add_procedural_fabric_texture:
 #       material = _add_procedural_fabric_texture_to_bsdf(material)
 #   cloth_object.data.materials[0] = material


def render_poses_frames(args, obj_paths):
    # set format to args format
    global FORMAT
    FORMAT = args.format

    global extension
    global COLOR_DEPTH
    if FORMAT == 'exr':
        extension = '.exr'
        COLOR_DEPTH = 32
    else:
        extension = '.png'
        COLOR_DEPTH = 8

    fp = args.results
    json_data = json.load(open(args.poses))

    # load res scene
    # bpy.ops.wm.open_mainfile(filepath=args.res_scene)
    # remove the cube
    bpy.ops.object.delete()
    # remove the default light
    # bpy.ops.object.select_by_type(type="LIGHT")
    # bpy.ops.object.delete()


    # bpy.data.objects['Plane'].data.materials.append(material)

    # Set up rendering
    res_scene = bpy.context.scene
    
    set_scene_params(res_scene,args,json_data)

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')


    if args.depth:
        depth_file_output = setup_depth(nodes,links,render_layers)
            
    # frame-based structure
    frame_folders = []

    # make one frame-based folder
    frames_folder = os.path.join(fp,'frames')
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    else:   
        shutil.rmtree(frames_folder)
        os.makedirs(frames_folder)

    # make one view-based folder    
    views_folder = os.path.join(fp,'views')
    if not os.path.exists(views_folder):
        os.makedirs(views_folder)
    else:
        shutil.rmtree(views_folder)
        os.makedirs(views_folder)

    # TODO: make it dependent on the number of meshes
    num_frames = len(obj_paths)
    for j in range(num_frames):
        
        frame_dir = os.path.join(frames_folder,'frame_'+str(j))
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        else:
            shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)
        
        frame_folders.append(frame_dir)

        # add poses to each frame folder
        shutil.copy(args.poses,os.path.join(frame_dir,'poses.json'))

    # make folder for pac-nerf 
    pacnerf_dir = os.path.join(fp,'pacnerf')
    if not os.path.exists(pacnerf_dir):
        os.makedirs(pacnerf_dir)
    else:
        shutil.rmtree(pacnerf_dir)
        os.makedirs(pacnerf_dir)

    os.makedirs(os.path.join(pacnerf_dir,'data'))
    # copy pac-nerf config file
    shutil.copy(fp + '/' + 'pacnerf.json',os.path.join(pacnerf_dir,'all_data.json'))

    cam = bpy.data.objects['Camera']
    scene = res_scene
    depth_files_split = []
    
    material = generate_towel_material()
    image_path = args.material_path
    material = get_image_material(image_path)
    # mesh = bpy.data.objects["_sequence"]

    #############################

    # breakpoint()
    for j in range(num_frames):
        
        # breakpoint()
        # if args.stop_motion:
        #     mesh_data = bpy.data.meshes[f'{j:05}']
        #     bpy.data.meshes[f'{j:05}'].materials[0] = material 
            
        #     mesh.data = mesh_data
        #     mesh.active_material_index = len(mesh.data.materials) - 1 
            
        # Hard code this as we want one frame per time as we need to update the mesh
        # scene.frame_start = j
        # scene.frame_end = j
        

        # delete previous cloth objects
        bpy.ops.object.delete()
        
        # load mesh and append material to the mesh
        cloth_object = load_cloth_mesh(obj_paths[j], solidify=True, subdivide=True)
        # cloth_object.data.materials[0] = material     # this was the 3.4 version
        cloth_object.data.materials.append(material)
        
        # apply a rotation to the mesh
        cloth_object.rotation_euler[0] = np.pi
        cloth_object.rotation_euler[1] = np.pi
        cloth_object.rotation_euler[2] = np.pi
            
        # Hard code this as we want one frame per time as we need to update the mesh
        scene.frame_start = j
        scene.frame_end = j
    
        for i,frame in enumerate(json_data['frames']):
            
            transform_matrix = Matrix(frame['transform_matrix'])            

            cam.matrix_world = transform_matrix
            cam.keyframe_insert(data_path='location', frame=j)
            cam.keyframe_insert(data_path='rotation_euler', frame=j)
            
            view_dir = os.path.join(views_folder,'view_'+str(i))
            if not os.path.exists(view_dir):
                os.makedirs(view_dir)
            else:
                # shutil.rmtree(view_dir)
                os.makedirs(view_dir, exist_ok=True)
            
            scene.render.filepath = os.path.join(view_dir, 'r_' + str(i))
            
            if args.depth:
                depth_file_output.file_slots[0].path = scene.render.filepath + "_depth"
                print(scene.render.filepath + "_depth")
            
            
            if DEBUG:
                break
            else:
                bpy.ops.render.render(write_still=True,animation=True)  # render still
            
            written_files = glob.glob(scene.render.filepath+"*"+extension)
            
            # breakpoint()
            # remove strings with 'depth' in them
            png_files = [x for x in written_files if 'depth' not in x]
            png_files.sort()
            png_file = png_files[-1]
            # writing the rendered imgs to frame-based folder
            shutil.copy(png_file,os.path.join(frame_folders[j],'r_'+str(i)+extension))

            # writing the rendered imgs to pac-nerf folder
            shutil.copy(png_file,os.path.join(pacnerf_dir,'data','r_'+str(i)+'_'+str(j)+extension))

            if args.depth:
                depth_files = [x for x in written_files if 'depth' in x]
                depth_files.sort()
                depth_file = depth_files[-1]
                depth_files_split.append(depth_files)
            
                # writing the rendered depth.png to frame-based folder
                shutil.copy(depth_file, os.path.join(frame_folders[j], 'r_' + str(i) + '.depth'+extension))

                # writing the rendered depth.png to frame-based folder
                shutil.copy(depth_file, os.path.join(pacnerf_dir, 'data', 'r_' + str(i) + '_' + str(j) + '.depth'+extension))
        
        depth_files_loaded = []
        split_depth = None
        # breakpoint()
        if args.depth:
            for i in range(len(depth_files_split)):
                depth_files = depth_files_split[i]
                for j, depth_file in enumerate(depth_files):
                    depth_files_loaded.append(depth_file)
                    if split_depth is None:
                        split_depth = load_exr(depth_file)
                    else:
                        split_depth = np.concatenate((split_depth,load_exr(depth_file)),axis=0)


        #     # save depth_files_loaded 
        #     np.savez(os.path.join(pacnerf_dir,'data','depth.npz'),depth=split_depth,filenames=np.array(depth_files_loaded))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', nargs='?', default="cloth/scenes/cloth_ball.blend", help="Path to the scene.blend file")
    parser.add_argument('--frame_start',type=int,default=-1,required=False)
    parser.add_argument('--frame_end',type=int,default=-1,required=False)
    parser.add_argument('-p','--poses',type=str,default='cloth/poses/lego/train.json')
    args = parser.parse_args()

    
    obj_paths = [ glob.glob('/home/omniverse/workspace/cloth-splatting/sim_datasets/test_dataset_0415/TOWEL/00000/00000/obj/*.obj')]
    obj_paths.sort()
    
    config = ClothMeshConfig(mesh_path=obj_paths[0])
    
    render_poses_frames(args, obj_paths, config)
