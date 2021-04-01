import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch
import cv2
import copy

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)
    
     # begin - as minhas funçoes
    def get_mask(self, idx):
        mask_file = '/home/rmoreira/kitti/pcdet/training/masks/'+str(idx).rjust(6,'0')+'.pt'
        # assert mask_file.exists()
        return torch.load(mask_file)
    def cam_to_lidar(self, pointcloud, projection_mats):
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

    def augment_lidar_class_scores(self, class_scores, lidar_cam_coords, projection_mats):
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
        true_where_point_on_img = true_where_x_on_img & true_where_y_on_img
        
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

    def semantic_augmentation(self, pointcloud, calibration_matrix, im, idx):
        # outputs = self.predictor(im)
        class_scores = self.get_mask(idx)
        """
        class_scores = torch.zeros(im.shape[0], im.shape[1]).to("cuda:0")
        nr_instances = list(outputs["instances"].pred_classes.shape)[0]
        aux = torch.zeros(1,im.shape[0], im.shape[1]).to("cuda:0")
        for i in range(0,nr_instances):
            if outputs["instances"].pred_classes[i]==0:
                class_scores = torch.stack([outputs["instances"].pred_masks[i], class_scores], dim=0)
                class_scores = torch.amax(class_scores, dim=0)
       
        class_scores = torch.zeros(im.shape[0], im.shape[1]).to("cuda:0")
        nr_instances = list(outputs["instances"].pred_classes.shape)[0]
        aux = torch.zeros(1,im.shape[0], im.shape[1]).to("cuda:0")
        for i in range(0,nr_instances):
            mask = outputs["instances"].pred_classes[i] 
            if mask in [0,1,2]:
                    class_scores = torch.stack([outputs["instances"].pred_masks[i]*(mask+1), class_scores], dim=0)
                    class_scores = torch.amax(class_scores, dim=0)
        """

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

        lidar_cam_coords = self.cam_to_lidar(pointcloud, projection_mats)
        augmented_lidar_cam_coords, mask_aux = self.augment_lidar_class_scores(class_scores, lidar_cam_coords, projection_mats)
        reduced_pointcloud = torch.tensor(pointcloud[mask_aux])
        augmented_lidar_coords = np.c_[reduced_pointcloud, augmented_lidar_cam_coords.cpu().numpy()[:,4]] 
        augmented_lidar_coords_tensor = torch.tensor(augmented_lidar_coords).to('cuda:0')

        return augmented_lidar_coords

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        
        # begin - pointpainting
        points = np.fromfile("/home/rmoreira/OpenPCDet/data/kitti/training/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
        img_path = "/home/rmoreira/OpenPCDet/data/kitti/training/image_2/000000.png"
        img = cv2.imread(img_path)
        calib_path = "/home/rmoreira/OpenPCDet/data/kitti/training/calib/000000.txt"
        points_aug =  self.semantic_augmentation(points, open(calib_path), img, 0)
        
        input_dict = {
            'points': points_aug,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # print(pred_dicts)
            """
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)
            """

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
