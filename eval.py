from utils import *
from dataset import KittiDataset
import torch
import numpy as np
import os
import pickle

from model import SSD

def evaluate(test_loader, train_dataset, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()

    with torch.no_grad():
        for i, augmented_lidar_cam_coords in enumerate(test_loader):
            augmented_lidar_cam_coords = augmented_lidar_cam_coords[0]
            predicted_locs, predicted_scores, _ = model(augmented_lidar_cam_coords)
            # Don't understand what's going on than:
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.2, max_overlap=0.45,
                                                                                       top_k=10)
            det_boxes_batch = det_boxes_batch[0][0]
            det_labels_batch = det_labels_batch[0]
            det_scores_batch = det_scores_batch[0][0]
            boxes = det_boxes_batch * bev_len * pillar_resolution
            score = det_scores_batch
            # label = [l.cpu().numpy() for l in det_labels_batch]
            # score = [s.cpu().numpy() for s in det_scores_batch]
            # boxes = [b.cpu().numpy()*bev_len for b in det_boxes_batch]
            # boxes = np.concatenate(boxes)
            xmin, ymin, xmax, ymax = boxes[0], boxes[1], boxes[2], boxes[3]
            ymin = z_range[1] - ymin
            ymax = z_range[1] - ymax
            xmin = xmin + x_range[0]
            xmax = xmax + x_range[0]
            x = xmin + (xmax - xmin) / 2
            z = ymin + (ymax - ymin) / 2
            w, l = xmax - xmin, ymax - ymin
            if (w > l).all(): w, l, theta = l, w, 0
            train_dataset[i]
            cam_points = np.array([[xmin, ymin_cam, ymin, 1], [xmax, ymax_cam, ymax, 1]]).T
            pm = train_dataset.projection_mats;
            img_points = pm['P2'].dot(pm['R0_rect'].dot(cam_points)).T
            # img_points = cam_to_img.dot(cam_points).T
            f_name = file_nums_with_cars[i] + '.txt'
            with open(f_name,"w+") as f:
                f.write(f"Car 0 0 0 {img_points[0, 0]} {img_points[0, 2]} {img_points[1, 0]} {img_points[1, 2]} -1 {w} {l} {x} {y} {z} {theta} {score}")



device = torch.device("cpu")
device = 'cpu'
checkpoint = './pointpillars.pth'
checkpoint = torch.load(checkpoint, map_location='cpu')
model = SSD(resnet_type=34, n_classes=2)
model.load_state_dict(checkpoint)
model = model.to(device)
root = '/home/aulitin/workspace/kitti_data/'
results = './results'
if not os.path.exists(results):
    os.makedirs(results)
batch_size = 1
workers = 0
bev_len = 500
pillar_resolution=0.16
x_range=(-40, 40)
z_range=(0, 80)
ymax_cam, ymin_cam = 1.21, 2.91
h = ymin_cam - ymax_cam
y = (ymax_cam - ymin_cam) / 2
theta = np.pi/2
cam_to_img = np.array([[ 7.25995079e+02,  9.75088160e+00,  6.04164953e+02, 4.48572807e+01],
       [-5.84187313e+00,  7.22248154e+02,  1.69760557e+02, 2.16379106e-01],
       [ 7.40252715e-03,  4.35161404e-03,  9.99963105e-01, 2.74588400e-03],
       [0, 0, 0, 1]])

with open('file_nums.pkl', 'rb') as f:
    file_nums_with_cars = pickle.load(f)

model.eval()

# test_dataset = KittiDataset(root=root, mode="testing", valid=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                        #   collate_fn=test_dataset.collate_fn_eval, num_workers=workers)

train_dataset = KittiDataset(root=root, mode="training", valid=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=train_dataset.collate_fn, num_workers=workers)



# evaluate(test_loader, model)
evaluate(train_loader, train_dataset,  model)
