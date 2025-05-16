# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# ActivityNet
# Copyright (c) 2015 ActivityNet
# Licensed under The MIT License
# [see https://github.com/activitynet/ActivityNet/blob/master/LICENSE for details]
# --------------------------------------------------------

"""Helper functions for AVA evaluation."""

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import logging
import pprint
import time
from collections import defaultdict
import os

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score   # ← 新增
import slowfast.utils.distributed as du
from slowfast.utils.env import pathmgr
from ava_evaluation import (
    object_detection_evaluation,
    standard_fields,
)

logger = logging.getLogger(__name__)


def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None, load_score=False):
    """Loads boxes and class labels from a CSV file in the AVA format.
    CSV file format described at https://research.google.com/ava/download.html.
    Args:
      csv_file: A file object.
      class_whitelist: If provided, boxes corresponding to (integer) class labels
        not in this set are skipped.
    Returns:
      boxes: A dictionary mapping each unique image key (string) to a list of
        boxes, given as coordinates [y1, x1, y2, x2].
      labels: A dictionary mapping each unique image key (string) to a list of
        integer class lables, matching the corresponding box in `boxes`.
      scores: A dictionary mapping each unique image key (string) to a list of
        score values lables, matching the corresponding label in `labels`. If
        scores are not provided in the csv, then they will default to 1.0.
    """
    boxes = defaultdict(list)
    labels = defaultdict(list)
    scores = defaultdict(list)
    with pathmgr.open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            assert len(row) in [7, 8], "Wrong number of columns: " + row
            image_key = make_image_key(row[0], row[1])
            x1, y1, x2, y2 = [float(n) for n in row[2:6]]
            action_id = int(row[6])
            if class_whitelist and action_id not in class_whitelist:
                continue
            score = 1.0
            if load_score:
                score = float(row[7])
            boxes[image_key].append([y1, x1, y2, x2])
            labels[image_key].append(action_id)
            scores[image_key].append(score)
    return boxes, labels, scores


def read_exclusions(exclusions_file):
    """Reads a CSV file of excluded timestamps.
    Args:
      exclusions_file: A file object containing a csv of video-id,timestamp.
    Returns:
      A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
      or an empty set if exclusions file is None.
    """
    excluded = set()
    if not exclusions_file or not os.path.isfile(exclusions_file):
        return excluded
    
    if exclusions_file:
        with pathmgr.open(exclusions_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 2, "Expected only 2 columns, got: " + row
                excluded.add(make_image_key(row[0], row[1]))
    return excluded


def read_labelmap(labelmap_file):
    """Read label map and class ids."""

    labelmap = []
    class_ids = set()
    name = ""
    class_id = ""
    with pathmgr.open(labelmap_file, "r") as f:
        for line in f:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
    return labelmap, class_ids


def evaluate_ava_from_files(labelmap, groundtruth, detections, exclusions):
    """Run AVA evaluation given annotation/prediction files."""

    categories, class_whitelist = read_labelmap(labelmap)
    excluded_keys = read_exclusions(exclusions)
    groundtruth = read_csv(groundtruth, class_whitelist, load_score=False)
    detections = read_csv(detections, class_whitelist, load_score=True)
    run_evaluation(categories, groundtruth, detections, excluded_keys)


# def evaluate_ava(
#     preds,
#     original_boxes,
#     metadata,
#     excluded_keys,
#     class_whitelist,
#     categories,
#     groundtruth=None,
#     video_idx_to_name=None,
#     name="latest",
# ):
#     """Run AVA evaluation given numpy arrays."""

#     eval_start = time.time()

#     detections = get_ava_eval_data(
#         preds,
#         original_boxes,
#         metadata,
#         class_whitelist,
#         video_idx_to_name=video_idx_to_name,
#     )

#     logger.info("Evaluating with %d unique GT frames." % len(groundtruth[0]))
#     logger.info("Evaluating with %d unique detection frames" % len(detections[0]))

#     write_results(detections, "detections_%s.csv" % name)
#     write_results(groundtruth, "groundtruth_%s.csv" % name)

#     results = run_evaluation(categories, groundtruth, detections, excluded_keys)

#     logger.info("AVA eval done in %f seconds." % (time.time() - eval_start))
#     return results["PascalBoxes_Precision/mAP@0.5IOU"]

def evaluate_ava(
    preds,
    original_boxes,
    metadata,
    excluded_keys,
    class_whitelist,
    categories,
    groundtruth=None,
    video_idx_to_name=None,
    name="latest",
):
    """
    Run AVA evaluation with predictions expressed为 0-49 连续索引，
    ground-truth 已经是原始 AVA id。这里只把预测映射回原 id。
    """
    eval_start = time.time()

    # 1. 构造连续索引 → 原始 AVA id 的映射
    sorted_orig_ids = sorted(class_whitelist)              # e.g. [1,2,4,…,80]
    index2orig = {idx: orig_id for idx, orig_id in enumerate(sorted_orig_ids)}

    # 2. 将预测标签映射回原始 AVA id
    detections = get_ava_eval_data(
        preds,
        original_boxes,
        metadata,
        class_whitelist,
        index2orig=index2orig,          # 只映射预测
        video_idx_to_name=video_idx_to_name,
    )

    # ------- 日志 -------
    logger.info(
        "Evaluating with %d unique GT frames.",
        len(groundtruth[0]),            # groundtruth[0] 是 GT 的 boxes dict
    )
    logger.info(
        "Evaluating with %d unique detection frames",
        len(detections[0]),
    )

    # ------- 写结果 CSV（可选）-------
    # write_results(detections, f"detections_{name}.csv")
    # write_results(groundtruth,  f"groundtruth_{name}.csv")

    # ------- 计算官方 mAP -------
    results = run_evaluation(categories, groundtruth, detections, excluded_keys)
    map50 = results["PascalBoxes_Precision/mAP@0.5IOU"]
    
    # 更稳健地提取每个类别的 AP
    ap_per_class = {}
    for cat in categories:
        cid = cat.get("id")
        if cid is None:
            continue
        cname = cat.get("name", f"class_{cid}")
        # 注意这里的 key 要跟 metrics 里的实际命名保持一致
        key = f"PascalBoxes_PerformanceByCategory/AP@0.5IOU/{cname}"
        ap = results.get(key)
        if ap is None:
            # 如果还拿不到，你就打印一下 key 看看哪一步不对
            logger.warning(f"No AP metric for key: {key}")
            continue
        ap_per_class[cid] = {
            "name": cname,
            "ap50": ap,
    }


    # ==========================================================
    # 附加计算：macro-averaged F1 与 AUROC
    #   - 逐个 box 把 50-维概率取出来作 y_scores
    #   - 把 GT 做 one-hot             → y_true
    #   - y_pred = (y_scores ≥ 0.5)
    # ==========================================================
    y_scores, y_true = [], []
    num_cls = preds.shape[1]

    # ❶ 先做一个 GT 查询表，便于 O(1) 判断 “某 box 是否命中某类”
    gt_lookup = set()
    gt_boxes, gt_labels, _ = groundtruth
    for k, labs in gt_labels.items():
        for l in labs:
            gt_lookup.add((k, l))         # (video,frame) 的字符串键 + 原始 id

    # ❷ 遍历每个预测 box，拼接 y_scores / y_true
    for i in range(preds.shape[0]):
        video_idx = int(np.round(metadata[i][0]))
        sec       = int(np.round(metadata[i][1]))
        video     = video_idx_to_name[video_idx]
        key       = f"{video},{sec:04d}"

        # predictions 已经是 0-49；转回原 id → 与 GT 空间一致
        prob_vec = preds[i].tolist()
        y_scores.extend(prob_vec)

        # 构造同样顺序的 0/1 真值
        for cls_idx in range(num_cls):
            orig_id = index2orig[cls_idx]
            y_true.append(1 if (key, orig_id) in gt_lookup else 0)
            
    # ---------- reshape 为 (N_box, num_cls) ----------
    y_true   = np.array(y_true).reshape(-1, num_cls)
    y_scores = np.array(y_scores).reshape(-1, num_cls)
    y_pred   = (y_scores >= 0.5).astype(int)
    
    # ---------- macro 计算 ----------
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ---- 过滤掉真值里全 0 或全 1 的动作列 ----
    keep = []
    for c in range(num_cls):
        s = y_true[:, c].sum()
        if 0 < s < y_true.shape[0]:   # 既有 0 也有 1
            keep.append(c)
    if keep:
        auroc_macro = roc_auc_score(
            y_true[:, keep], y_scores[:, keep], average="macro"
        )
    else:
        auroc_macro = 0.0   # 极端情况下所有列都单一
    
    #-------recall macro--------
    
    recall_macro = recall_score(y_true,y_pred,average="macro",zero_division=0)

    logger.info(
        "mAP@0.5 = %.5f | macro-F1 = %.5f | macro-AUROC = %.5f | macro-recall = %.5f",
        map50, f1_macro, auroc_macro, recall_macro
    )

    return map50, f1_macro, auroc_macro, recall_macro, ap_per_class, detections




def run_evaluation(categories, groundtruth, detections, excluded_keys, verbose=True):
    """AVA evaluation main logic."""

    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    boxes, labels, _ = groundtruth

    gt_keys = []
    pred_keys = []

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                (
                    "Found excluded timestamp in ground truth: %s. "
                    "It will be ignored."
                ),
                image_key,
            )
            continue
        pascal_evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.InputDataFields.groundtruth_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.InputDataFields.groundtruth_difficult: np.zeros(
                    len(boxes[image_key]), dtype=bool
                ),
            },
        )

        gt_keys.append(image_key)

    boxes, labels, scores = detections

    for image_key in boxes:
        if image_key in excluded_keys:
            logging.info(
                ("Found excluded timestamp in detections: %s. " "It will be ignored."),
                image_key,
            )
            continue
        pascal_evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes: np.array(
                    boxes[image_key], dtype=float
                ),
                standard_fields.DetectionResultFields.detection_classes: np.array(
                    labels[image_key], dtype=int
                ),
                standard_fields.DetectionResultFields.detection_scores: np.array(
                    scores[image_key], dtype=float
                ),
            },
        )

        pred_keys.append(image_key)

    metrics = pascal_evaluator.evaluate()

    if du.is_master_proc():
        pprint.pprint(metrics, indent=2)
    return metrics


# def get_ava_eval_data(
#     scores,
#     boxes,
#     metadata,
#     class_whitelist,
#     verbose=False,
#     video_idx_to_name=None,
# ):
#     """
#     Convert our data format into the data format used in official AVA
#     evaluation.
#     """

#     out_scores = defaultdict(list)
#     out_labels = defaultdict(list)
#     out_boxes = defaultdict(list)
#     count = 0
#     for i in range(scores.shape[0]):
#         video_idx = int(np.round(metadata[i][0]))
#         sec = int(np.round(metadata[i][1]))

#         video = video_idx_to_name[video_idx]

#         key = video + "," + "%04d" % (sec)
#         batch_box = boxes[i].tolist()
#         # The first is batch idx.
#         batch_box = [batch_box[j] for j in [0, 2, 1, 4, 3]]

#         one_scores = scores[i].tolist()
#         for cls_idx, score in enumerate(one_scores):
#             if cls_idx + 1 in class_whitelist:
#                 out_scores[key].append(score)
#                 out_labels[key].append(cls_idx + 1)
#                 out_boxes[key].append(batch_box[1:])
#                 count += 1

#     return out_boxes, out_labels, out_scores

def get_ava_eval_data(
    scores,
    boxes,
    metadata,
    class_whitelist,
    index2orig=None,           # <── 新增
    verbose=False,
    video_idx_to_name=None,
):
    """
    Convert model outputs into the format used in official AVA evaluation.
    `scores` shape: [N_box, num_classes] where class index is 0 … 49.
    If `index2orig` is provided, we map it back to the original AVA id.
    """

    out_scores = defaultdict(list)
    out_labels = defaultdict(list)
    out_boxes  = defaultdict(list)

    for i in range(scores.shape[0]):
        video_idx = int(np.round(metadata[i][0]))
        sec       = int(np.round(metadata[i][1]))
        video     = video_idx_to_name[video_idx]
        key       = f"{video},{sec:04d}"

        # xyxy to x1,y1,x2,y2  (keep原实现)
        batch_box = boxes[i].tolist()
        batch_box = [batch_box[j] for j in [0, 2, 1, 4, 3]]

        one_scores = scores[i].tolist()
        for cls_idx, score in enumerate(one_scores):
            orig_id = index2orig[cls_idx] if index2orig is not None else cls_idx + 1
            if orig_id in class_whitelist:
                out_scores[key].append(score)
                out_labels[key].append(orig_id)   # ← 现在写入原始 ID
                out_boxes[key].append(batch_box[1:])

    return out_boxes, out_labels, out_scores


def write_results(detections, filename):
    """Write prediction results into official formats."""
    start = time.time()

    boxes, labels, scores = detections
    with pathmgr.open(filename, "w") as f:
        for key in boxes.keys():
            for box, label, score in zip(boxes[key], labels[key], scores[key]):
                f.write(
                    "%s,%.03f,%.03f,%.03f,%.03f,%d,%.04f\n"
                    % (key, box[1], box[0], box[3], box[2], label, score)
                )

    logger.info("AVA results wrote to %s" % filename)
    logger.info("\ttook %d seconds." % (time.time() - start))
