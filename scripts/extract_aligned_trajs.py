import glob
import os.path
import shutil
import argparse

parser = argparse.ArgumentParser("Extract aligned trajectories from a directory of ours_* directories.")
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument("--target_name", type=str, default=None)
parser.add_argument("--src_dir", type=str, required=True)
parser.add_argument("--iteration", type=int, default=None)
parser.add_argument("--take_all", action='store_true')
args = parser.parse_args()

dirs = sorted(glob.glob(os.path.join(args.src_dir, 'test/ours_*')))
if not args.take_all:
    if args.iteration is not None:
        dirs = [os.path.join(args.src_dir, f'test/ours_{args.iteration}')]
    else:
        dirs = [dirs[-1]]

os.makedirs(args.target_dir, exist_ok=True)
for d in dirs:
    iter = d.split('_')[-1]
    trajs = os.path.join(d, 'all_trajs_aligned.npz')
    target_name = args.target_name if args.target_name is not None else f'{iter}.npz'
    dst_file = os.path.join(args.target_dir, target_name)
    shutil.copyfile(trajs, dst_file)