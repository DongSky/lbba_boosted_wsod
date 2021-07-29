# ---------------------------------------------------------
# OICR
# Written by Jaedong Hwang
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------

"""The layer used during training to get proposal labels for classifier refinement.

OICRLayer implements a Caffe Python layer.
"""
import numpy as np
import pdb
import torch
from utils.bbox import bbox_overlaps


def MISTLayer(boxes, cls_prob, im_labels, cfg_TRAIN_FG_THRESH=0.5):
    boxes = boxes[..., 1:]
    # print(cls_prob.shape)
    # exit()
    proposals = _get_p_percent_score_proposals(boxes, cls_prob, im_labels)

    labels, rois, cls_loss_weights = _sample_rois(boxes, proposals, 21)
    return labels, cls_loss_weights


def _get_p_percent_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with percent score."""
    num_rois = boxes.shape[0]
    k = int(num_rois * 0.15)
    
    # num_images, num_classes = im_labels.shape

    im_labels_tmp = im_labels[0, :]
   
    num_gt_cls = im_labels_tmp.sum()
    # gt_boxes = np.zeros((0, 4), dtype=np.float32)
    # gt_classes = np.zeros((0, 1), dtype=np.int32)
    # gt_scores = np.zeros((0, 1), dtype=np.float32)

    if 21 == cls_prob.shape[1]:  # added 1016
        cls_prob = cls_prob[:, 1:]
    
    gt_cls_inds = im_labels_tmp.nonzero(as_tuple=False)[:,0]
    sorted_scores, max_inds = cls_prob[:, gt_cls_inds].sort(dim=0, descending=True)
    sorted_scores = sorted_scores[:k]
   
    max_inds = max_inds[:k]

    _boxes = boxes[max_inds.t().contiguous().view(-1)].view(num_gt_cls.int(), -1, 4)
    ious = torch.zeros((_boxes.shape[0],_boxes.shape[1],_boxes.shape[1]))
    for i in range(_boxes.shape[0]):
        iou_c = bbox_overlaps(_boxes[i],_boxes[i])
        ious[i] = iou_c
    k_ind = torch.zeros(num_gt_cls.int(), k, dtype=torch.bool)
    k_ind[:, 0] = 1
    iou_th = 0.2
    for ii in range(1, k):
        max_iou, _ = torch.max(ious[:,ii:ii+1, :ii], dim=2)
              
        k_ind[:, ii] = (max_iou < iou_th).byte().squeeze(-1)
    
    gt_boxes = _boxes[k_ind]
    gt_cls_id = gt_cls_inds + 1
    temp_cls = torch.ones((_boxes.shape[:2])) * gt_cls_id.view(-1, 1).float()
    gt_classes = temp_cls[k_ind].view(-1, 1).long()
    gt_scores = sorted_scores.t().contiguous()[k_ind].view(-1, 1)
    
    # for i in range(num_classes):
    #     if im_labels_tmp[i] == 1:
    #         cls_prob_tmp = cls_prob[:, i].data
    #         max_index = np.argmax(cls_prob_tmp)
    #         gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1).cpu()))
    #         gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))  # for pushing ground
    #         gt_scores = np.vstack((gt_scores,
    #                                cls_prob_tmp[max_index]))  # * np.ones((1, 1), dtype=np.float32)))
    #         # cls_prob[:, max_index, :] = 0  # in-place operation <- OICR code but I do not agree

    proposals = {'gt_boxes': gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals




def _sample_rois(all_rois, proposals, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    try:
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
    except:
        pdb.set_trace()

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < 0.5)[0]

    labels[bg_inds] = 0
    real_labels = np.zeros((labels.shape[0], 21))
    for i in range(labels.shape[0]):
        real_labels[i, labels[i]] = 1
    rois = all_rois
    return real_labels, rois, cls_loss_weights
