"""
author: Timothy C. Arlen
date: 28 Feb 2018
revised by: Sadjad Asghari Esfeden
date: 10 Sep 2018
url: https://gist.github.com/sadjadasghari/dc066e3fb2e70162fbb838d4acb93ffc
Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. 
"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from traffic_analysis.d00_utils.bbox_helpers import bbox_intersection_over_union

sns.set_style("white")
sns.set_context("poster")

COLORS = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]


def get_single_frame_results(gt_bboxes: list, 
                             pred_bboxes: dict, 
                             iou_thr: float) -> dict:
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_bboxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_bboxes (list of list of floats): list of list of bboxes (formatted like `gt_bboxes`)
        iou_thr: value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_bboxes))
    all_gt_indices = range(len(gt_bboxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_bboxes)
        return {"true_pos": tp, "false_pos": fp, "false_neg": fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_bboxes)
        fn = 0
        return {"true_pos": tp, "false_pos": fp, "false_neg": fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_bbox in enumerate(pred_bboxes):
        for igb, gt_bbox in enumerate(gt_bboxes):
            iou = bbox_intersection_over_union(pred_bbox, gt_bbox)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_bboxes)
        fn = len(gt_bboxes)
    else:
        # print("matches found")
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_bboxes) - len(pred_match_idx)
        fn = len(gt_bboxes) - len(gt_match_idx)
    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}


def calc_precision_recall(frame_results):
    """Calculates precision and recall from the set of frames by summing the true positives, 
    false positives, and false negatives for each frame. 
    Args:
        frame_results (dict): dictionary formatted like:
            {
                'frame1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'frame2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for _, res in frame_results.items():
        true_pos += res["true_pos"]
        false_pos += res["false_pos"]
        false_neg += res["false_neg"]

    try:
        precision = true_pos / (true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos / (true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0
    return precision, recall


def get_model_scores_dict(pred_bboxes: dict) -> dict:
    """Creates a dictionary mapping model_scores to image ids.
    Args:
        pred_bboxes: dict of dicts of 'bboxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_scores_dict = {}
    for frame_id, frame_dict in pred_bboxes.items():
        if len(frame_dict["bboxes"]) > 0:
            for score in frame_dict["scores"]:
                if score not in model_scores_dict.keys():
                    model_scores_dict[score] = [frame_id]
                else:
                    model_scores_dict[score].append(frame_id)
    return model_scores_dict


def get_avg_precision_at_iou(gt_bboxes: dict, 
                             pred_bboxes: dict, 
                             iou_thr: float = 0.5):
    """Calculates average precision at given IoU threshold.
    Args:

        gt_bboxes: dictionary with frame index as keys, and a list of 
                    bboxes of ground truth objects as values. Bboxes should be in 
                    format [xmin, ymin, xmin+width, ymin+height].
        pred_bboxes: dictionary with frame index as keys, and subdictinaries for values. 
                     Each subdictionary has keys "bboxes", "scores". The key "bboxes" 
                     corresponds to a list of bboxes of predicted objects (in same format 
                     as specified above). The key "scores" corresponds to a list of  model 
                     confidences for each bbox predicted. 
        iou_thr: value of IoU to consider as threshold for a true prediction.

    Returns:
        dict: avg precision as well as summary info about the PR curve
            keys:
                'avg_prec' (float): average precision for this IoU threshold
                'precisions' (list of floats): precision value for the given
                    model_threshold
                'recall' (list of floats): recall value for given
                    model_threshold
                'models_thrs' (list of floats): model threshold value that
                    precision and recall were computed for.
    """
    model_scores_dict = get_model_scores_dict(pred_bboxes)
    sorted_model_scores = sorted(model_scores_dict.keys())

    # Sort the predicted boxes in ascending order (lowest scoring boxes first):
    for frame_id, frame_dict in pred_bboxes.items():
        if len(frame_dict["bboxes"]) > 0:
            arg_sort = np.argsort(frame_dict["scores"])
            # reorder frame_dict
            frame_dict["scores"] = np.array(frame_dict["scores"])[arg_sort].tolist()
            frame_dict["bboxes"] = np.array(frame_dict["bboxes"])[arg_sort].tolist()

    pred_bboxes_pruned = deepcopy(pred_bboxes)

    precisions = []
    recalls = []
    model_thrs = []
    frame_results = {}
    # Loop over model score thresholds and calculate precision, recall
    for thres_idx, model_score_thres in enumerate(sorted_model_scores[:-1]):
        # for each model_score_thres, prune pred_bboxes dictionary so that it only contains
        # bboxes corresponding to scores above the model_score_threshold
        # On first iteration, define frame_results for the first time:
        frame_ids = (
            gt_bboxes.keys() if thres_idx == 0 else model_scores_dict[model_score_thres]
        )

        for frame_id in frame_ids:
            gt_bboxes_frame = gt_bboxes[frame_id]
            if len(pred_bboxes_pruned[frame_id]["bboxes"]) > 0:
                bbox_scores = pred_bboxes_pruned[frame_id]["scores"]
                start_idx = 0
                for score in bbox_scores:
                    if score < model_score_thres:
                        start_idx += 1
                    else:
                        break

                # Remove boxes, scores of lower than threshold scores:
                pred_bboxes_pruned[frame_id]["scores"] = pred_bboxes_pruned[frame_id]["scores"][start_idx:]
                pred_bboxes_pruned[frame_id]["bboxes"] = pred_bboxes_pruned[frame_id]["bboxes"][start_idx:]

                # Recalculate frame results (i.e. tps, fps, fns) model threshold and iou_threshold)
                frame_results[frame_id] = get_single_frame_results(
                    gt_bboxes_frame, pred_bboxes_pruned[frame_id]["bboxes"], iou_thr
                )
            else:
                # no bboxes detected
                frame_results[frame_id] = {
                    "true_pos": 0,
                    "false_pos": 0,
                    "false_neg": len(gt_bboxes_frame),
                }
        prec, rec = calc_precision_recall(frame_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thres)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    return {
        "avg_prec": avg_prec,
        "precisions": precisions,
        "recalls": recalls,
        "model_thrs": model_thrs,
    }


def plot_pr_curve(precisions, 
                  recalls, 
                  category="Person", 
                  label=None, 
                  color=None, 
                  ax=None):
    """Simple plotting helper function"""
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve for {}".format(category))
    ax.set_xlim([0.0, 1.3])
    ax.set_ylim([0.0, 1.2])
    return ax
