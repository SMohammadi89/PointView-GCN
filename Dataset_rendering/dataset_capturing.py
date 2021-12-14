#This is the code for single-view pcd generation. We generate 20 single-view pcds
#taken one scane using generated camera posision (args.camera_position)

import open3d as o3d
from numpy.linalg import inv
import scipy.io as sio
import numpy as np
import os
import re
import argparse

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def poi(dir_point):
    dataList = list()
    for (dir, _, files) in os.walk(dir_point):
        for f in sorted(files):
            path = os.path.join(dir, f)
            if os.path.exists(path):
                dataList.append(path)

    po = list()
    prog = re.compile('.ply$')
    for d in range(len(dataList)):
        binMatch = prog.search((dataList[d]))
        if binMatch:
            po.append(binMatch.string)

    return po


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=1, help="Number of iteration")
    parser.add_argument('--camera-pose', type=str, default="../Dataset_rendering/camera_position.mat", help="path to the camera pose")
    parser.add_argument('--output-dir', type=str, default="../data/single_view_modelnet/", help="generated_single_view_pcd")
    parser.add_argument('--input-dir', type=str, default="../data/ModelNet40_Normalize_1/", help="generated_single_view_pcd")
    #path of the input split train/test, here you just have to run two times the code and chenge the "train" to test
    parser.add_argument('--out-split-dir', type=str, default="/train/", help="the split of the input/output data")
    args = parser.parse_args()

    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase','wardrobe','xbox']


    o_dir = args.output_dir
    i_dir = args.input_dir
    mood = args.out_split_dir

    for Class in classes:
        print("Processing {} ...".format(Class))
        out_scene_point = o_dir + 'scene_3dpoint' + '/' + Class + mood
        out_depth_file = o_dir + 'depth' + '/' + Class + mood
        dir_point = "{}{}{}".format(i_dir, Class, mood)
        PO = poi(dir_point)

        list_scene_name = list()
        for data in range(len(PO)):
            scene = os.path.split(PO[data])[-1][:-4]
            list_scene_name.append(scene)

        for scene_name in list_scene_name:
            point = dir_point + scene_name + '.ply'  # Directory of related scene
            print(point)

            pcd = o3d.io.read_triangle_mesh(point)
            posdata = sio.loadmat(args.camera_pose)
            poset = np.array(posdata['transformCell'])
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

            for i in range(poset.shape[0]):
                number1 = poset[i][0]
                number = inv(number1)
                cam.extrinsic = number
                vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                vis.poll_events()
                vis.update_renderer()

                depth = vis.capture_depth_float_buffer(False)
                image = vis.capture_screen_float_buffer(False)
                file_name = os.path.split(point)[-1][:-4]
                file_name1 = os.path.split(point)[-1][:-9]

                depth_directory = (out_depth_file + file_name1)
                if not os.path.exists(out_depth_file):
                    os.makedirs(out_depth_file)
                #capture and save the depth image
                depth_save = vis.capture_depth_image(os.path.join(out_depth_file, '{}_{:03d}_depth.png'.format(scene_name,i+1)))
                depth_raw = o3d.io.read_image(os.path.join(out_depth_file, '{}_{:03d}_depth.png'.format(scene_name,i+1)))

                out_scene = (out_scene_point + file_name1  )
                if not os.path.exists(out_scene_point):
                    os.makedirs(out_scene_point)
                #generate pcd from depth image
                pc = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, cam.intrinsic, cam.extrinsic)
                out_mesh = o3d.io.write_point_cloud(os.path.join(out_scene_point, '{}_{:03d}.xyz'.format(scene_name, i + 1)),pc)
                #subsample pcd to 1024 points
                read_mesh = o3d.io.read_point_cloud(os.path.join(out_scene_point, '{}_{:03d}.xyz'.format(scene_name, i + 1)))
                pcd_points = np.asarray(read_mesh.points)
                point_set = farthest_point_sample(pcd_points, 1024)
                read_mesh.points = o3d.utility.Vector3dVector(point_set)
                sample_cloud = o3d.io.write_point_cloud(os.path.join(out_scene_point, '{}_{:03d}.xyz'.format(scene_name, i + 1)), read_mesh)



