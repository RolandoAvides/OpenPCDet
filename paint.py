import torch
import PIL
import os
import copy
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import io
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import json, random
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import time
import sys
from tqdm import trange
from numpy import load
from numpy import save
from numpy import asarray
import pickle
from skimage import io
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.utils.calibration_kitti import get_calib_from_file

def get_calib(calib_file, idx):
    return calibration_kitti.Calibration(calib_file)

def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def do_something():
    time.sleep(1)


def cam_to_lidar(pointcloud, projection_mats):
    """
    Takes in lidar in velo coords, returns lidar points in camera coords
    :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
    :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
    """

    lidar_velo_coords = copy.deepcopy(pointcloud)
    reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
    lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
    lidar_cam_coords = projection_mats['Tr_velo_to_cam'].dot(lidar_velo_coords.transpose())
    lidar_cam_coords = lidar_cam_coords.transpose()
    lidar_cam_coords[:, -1] = reflectances

    return lidar_cam_coords

def create_class_scores_mask(self, img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tensor_img = transform(img)
    tensor_img = tensor_img.unsqueeze(0).to(self.device)
    mask = self.deeplab101(tensor_img)
    mask = mask['out'] #ignore auxillary output
    _, preds = torch.max(mask, 1)
    class_scores = torch.where(preds==3, torch.ones(preds.shape).to(self.device), torch.zeros(preds.shape).to(self.device)) #convert preds to binary map (1 = car, else 0)
    class_scores = class_scores.squeeze()
    return class_scores

def augment_lidar_class_scores( class_scores, lidar_cam_coords, projection_mats):
    """
    Projects lidar points onto segmentation map, appends class score each point projects onto.
    """
    reflectances = copy.deepcopy(lidar_cam_coords[:, -1])
    lidar_cam_coords[:, -1] = 1 #homogenous coords for projection

    points_projected_on_mask = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
    points_projected_on_mask = points_projected_on_mask.transpose()
    points_projected_on_mask = points_projected_on_mask/(points_projected_on_mask[:,2].reshape(-1,1))

    true_where_x_on_img = (0 < points_projected_on_mask[:, 0]) & (points_projected_on_mask[:, 0] < class_scores.shape[1]) #x in img coords is cols of img
    true_where_y_on_img = (0 < points_projected_on_mask[:, 1]) & (points_projected_on_mask[:, 1] < class_scores.shape[0])
    true_where_z_on_img = (0 < points_projected_on_mask[:, 2])
    true_where_point_on_img = true_where_x_on_img & true_where_y_on_img
    true_where_point_on_img = true_where_point_on_img & true_where_z_on_img

    points_projected_on_mask = points_projected_on_mask[true_where_point_on_img] # filter out points that don't project to image
    # print(points_projected_on_mask.shape)
    lidar_cam_coords = torch.from_numpy(lidar_cam_coords[true_where_point_on_img])
    reflectances = reflectances[true_where_point_on_img]
    reflectances = torch.from_numpy(reflectances.reshape(-1, 1))
    points_projected_on_mask = np.floor(points_projected_on_mask).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
    points_projected_on_mask = torch.from_numpy(points_projected_on_mask[:, :2]) #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

    #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
    point_scores = class_scores[points_projected_on_mask[:, 1], points_projected_on_mask[:, 0]].reshape(-1, 1).double()
    # augmented_lidar_cam_coords = torch.cat((lidar_cam_coords[:, :-1].to(self.device), reflectances.to(self.device), point_scores.to(self.device)), 1)
    augmented_lidar_cam_coords = torch.cat((lidar_cam_coords[:, :-1].to("cuda:0"), reflectances.to("cuda:0"), point_scores.to("cuda:0")), 1)
    return augmented_lidar_cam_coords, true_where_point_on_img

def semantic_augmentation(pointcloud, calibration_matrix, im, predictor):
    outputs = predictor(im)

    class_scores = torch.zeros(im.shape[0], im.shape[1]).to("cuda:0")
    nr_instances = list(outputs["instances"].pred_classes.shape)[0]
    aux = torch.zeros(1,im.shape[0], im.shape[1]).to("cuda:0")
    for i in range(0,nr_instances):
        if outputs["instances"].pred_classes[i]==0:
                class_scores = torch.stack([outputs["instances"].pred_masks[i], class_scores], dim=0)
                class_scores = torch.amax(class_scores, dim=0)

    with calibration_matrix as f:
            lines = f.readlines()
            for l in lines:
                l = l.split(':')[-1]

            R0_rect = np.eye(4)
            Tr_velo_to_cam = np.eye(4)

            P2 = np.array(lines[2].split(":")[-1].split(), dtype=np.float32).reshape((3,4))
            R0_rect[:3, :3] = np.array(lines[4].split(":")[-1].split(), dtype=np.float32).reshape((3,3)) # makes 4x4 matrix
            Tr_velo_to_cam[:3, :4] = np.array(lines[5].split(":")[-1].split(), dtype=np.float32).reshape((3,4)) # makes 4x4 matrix
            projection_mats = {'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam':Tr_velo_to_cam}

    lidar_cam_coords = cam_to_lidar(pointcloud, projection_mats)
    augmented_lidar_cam_coords, mask_aux = augment_lidar_class_scores(class_scores, lidar_cam_coords, projection_mats)
    reduced_pointcloud = torch.tensor(pointcloud[mask_aux])
    augmented_lidar_coords = np.c_[reduced_pointcloud, augmented_lidar_cam_coords.cpu().numpy()[:,4]] 
    augmented_lidar_coords_tensor = torch.tensor(augmented_lidar_coords).to('cuda:0')

    return augmented_lidar_coords

def main():
        
        kitti_path = "/ctm-hdd-pool01/rmoreira/kitti/final/training/"
        kitti_save_path = "/ctm-hdd-pool01/rmoreira/kitti/final/training/semantics/"


        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
        predictor = DefaultPredictor(cfg)

        for idx in trange(0,1): #7481 
            img_name = 'image_2/'+str(idx).rjust(6,'0')+'.png'
            im = cv2.imread(kitti_path+img_name)
            outputs = predictor(im)
            save_path = kitti_save_path+str(idx).rjust(6,'0')+'.png'
            cv2.imwrite(save_path,out.get_image()[:, :, ::-1])
            """
            path_pointcloud = kitti_path+'velodyne/'+str(idx).rjust(6,'0')+'.bin'
            path_calib = kitti_path+'calib/'+str(idx).rjust(6,'0')+'.txt'

            path_pointcloud = kitti_path+'velodyne/'+str(idx).rjust(6,'0')+'.bin'
            path_calib = kitti_path+'calib/'+str(idx).rjust(6,'0')+'.txt'

            img_name = 'image_2/'+str(idx).rjust(6,'0')+'.png'
            im = cv2.imread(kitti_path+img_name)

            calib = get_calib(path_calib,idx)

            pointcloud = np.fromfile(path_pointcloud, dtype=np.float32).reshape(-1,4)
            pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, im.shape, calib)
            pointcloud = pointcloud[fov_flag]

            pcd = semantic_augmentation(pointcloud, open(path_calib), im, predictor)

            data = asarray(pcd)
            data = np.float32(data)    
            
            save(kitti_save_path+str(idx).rjust(6,'0')+'.npy', data)
            """


if __name__ == '__main__':
    main()
 
