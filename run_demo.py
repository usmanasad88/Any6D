
import os
import trimesh
import yaml
import numpy as np
import cv2
import torch

from PIL import Image
from estimater import Any6D

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()

if __name__=='__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="/home/mani/Repos/Any6D", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    args = parser.parse_args()


    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d

    results = []
    demo_path = 'demo_data'
    mesh_path = os.path.join(demo_path, f'mustard.obj')

    obj = 'demo_mustard'
    save_path = f'results/{obj}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_scale = 1000.0
    color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    Image.fromarray(color).save(os.path.join(save_path, 'color.png'))

    label = np.load(os.path.join(demo_path, 'labels.npz'))
    obj_num = 5
    mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)

    if img_to_3d:
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)

        input_image = preprocess_image(color, mask_refine, save_path, obj)
        images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
        instant_mesh_process(images, save_path, obj)

        mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

        mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
    else:
        mesh = trimesh.load(mesh_path)


    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)

    # camera info
    intrinsic_path = f"{demo_path}/836212060125_640x480.yml"
    with open(intrinsic_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    intrinsic = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]], [0.0, data["depth"]["fy"], data["depth"]["ppy"]], [0.0, 0.0, 1.0], ], )
    np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)

    pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')

    pose_list = label['pose_y']
    index_list = np.unique(label['seg'])
    index = (np.where(index_list == obj_num)[0] - 1).tolist()[0]
    tmp = pose_list[index]
    gt_pose = np.eye(4)
    gt_pose[:3, :] = tmp

    gt_mesh = trimesh.load(f'{ycb_model_path}/models/006_mustard_bottle/textured_simple.obj')

    chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh)
    print(chamfer_dis)

    np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
    np.savetxt(os.path.join(save_path, f'{obj}_gt_pose.txt'), gt_pose)
    est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))

    np.savetxt(os.path.join(save_path, f'{obj}_cd.txt'), [chamfer_dis])

    results.append({
        'Object': obj,
        'Object_Number': obj_num,
        'Chamfer_Distance': float(chamfer_dis)
        })



