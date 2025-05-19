#!/usr/bin/env python3
"""
SSD Evaluation script: computes mean Average Precision (mAP) for a single model
and writes out a JSON file keyed by epoch number.
"""

import argparse
import json
import logging
import re
import sys
import pathlib

import numpy as np
import torch

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import (
    create_mobilenetv1_ssd_lite,
    create_mobilenetv1_ssd_lite_predictor,
)
from vision.ssd.squeezenet_ssd_lite import (
    create_squeezenet_ssd_lite,
    create_squeezenet_ssd_lite_predictor,
)
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool


class MeanAPEvaluator:
    """
    Mean Average Precision (mAP) evaluator.
    """

    def __init__(
        self,
        dataset,
        net,
        arch="mb1-ssd",
        eval_dir="models/eval_results",
        nms_method="hard",
        iou_threshold=0.5,
        use_2007_metric=True,
        device="cuda:0",
    ):
        """
        Initializes evaluator.

        :param dataset: a Dataset object with .get_image, .get_annotation, .ids, .class_names
        :param net: a loaded SSD network
        :param arch: network architecture identifier
        :param eval_dir: directory in which to write JSON
        :param nms_method: NMS method ("hard" or "blend")
        :param iou_threshold: IoU threshold for detection
        :param use_2007_metric: whether to use PASCAL VOC 2007 AP computation
        :param device: torch device
        """
        self.dataset = dataset
        self.net = net
        self.iou_threshold = iou_threshold
        self.use_2007_metric = use_2007_metric

        self.eval_path = pathlib.Path(eval_dir)
        self.eval_path.mkdir(parents=True, exist_ok=True)

        # prepare ground‑truth
        (
            self.true_case_stat,
            self.all_gb_boxes,
            self.all_difficult_cases,
        ) = self.group_annotation_by_class(self.dataset)

        # choose predictor
        if arch == "vgg16-ssd":
            self.predictor = create_vgg_ssd_predictor(
                net, nms_method=nms_method, device=device
            )
        elif arch == "mb1-ssd":
            self.predictor = create_mobilenetv1_ssd_predictor(
                net, nms_method=nms_method, device=device
            )
        elif arch == "mb1-ssd-lite":
            self.predictor = create_mobilenetv1_ssd_lite_predictor(
                net, nms_method=nms_method, device=device
            )
        elif arch == "sq-ssd-lite":
            self.predictor = create_squeezenet_ssd_lite_predictor(
                net, nms_method=nms_method, device=device
            )
        elif arch == "mb2-ssd-lite":
            self.predictor = create_mobilenetv2_ssd_lite_predictor(
                net, nms_method=nms_method, device=device
            )
        else:
            raise ValueError(
                f"Invalid architecture '{arch}'; "
                "choose from vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, sq-ssd-lite."
            )

    def compute(self):
        """
        Compute mean AP and per-class AP entirely in memory. Does not write intermediate files.

        :returns: (mean_ap, [ap_class1, ap_class2, ...])
        """
        is_test = self.net.is_test
        self.net.is_test = True

        # collect all detections
        results = []
        for idx in range(len(self.dataset)):
            logging.debug(
                f"Evaluating image {idx + 1}/{len(self.dataset)}"
            )
            image = self.dataset.get_image(idx)
            boxes, labels, probs = self.predictor.predict(image)
            # matlab format offset +1 for later subtract
            det = torch.cat(
                [
                    torch.full((labels.size(0), 1), idx, dtype=torch.float32),
                    labels.float().unsqueeze(1),
                    probs.unsqueeze(1),
                    boxes + 1.0,
                ],
                dim=1,
            )
            results.append(det)

        self.net.is_test = is_test
        results = torch.cat(results, dim=0)

        # compute AP per class
        aps = []
        for cls_idx, cls_name in enumerate(self.dataset.class_names):
            if cls_idx == 0:
                # skip background
                continue
            sub = results[results[:, 1] == cls_idx]
            if sub.numel() == 0:
                aps.append(0.0)
                continue
            # sort by score descending
            sorted_idx = torch.argsort(sub[:, 2], descending=True)
            sub = sub[sorted_idx]

            # prepare inputs for in‑memory AP
            image_ids = [self.dataset.ids[int(i)] for i in sub[:, 0]]
            scores = sub[:, 2].numpy()
            # subtract 1.0 to go back to zero‑based coords
            boxes = [
                (sub[i, 3:] - 1.0).unsqueeze(0) for i in range(sub.shape[0])
            ]

            ap = self._compute_ap(
                self.true_case_stat[cls_idx],
                self.all_gb_boxes[cls_idx],
                self.all_difficult_cases[cls_idx],
                image_ids,
                boxes,
                scores,
                self.iou_threshold,
                self.use_2007_metric,
            )
            aps.append(ap)

        mean_ap = float(sum(aps) / len(aps)) if aps else 0.0
        return mean_ap, aps

    def _compute_ap(
        self,
        num_true_cases,
        gt_boxes,
        difficult_cases,
        image_ids,
        boxes,
        scores,
        iou_threshold,
        use_2007_metric,
    ):
        """
        Compute average precision for a single class given predictions in memory.

        :param num_true_cases: number of ground‑truth positives for this class
        :param gt_boxes: dict[image_id] -> Tensor[N,4] of GT boxes
        :param difficult_cases: dict[image_id] -> list of bool for each GT
        :param image_ids: list of image_id strings for each detection
        :param boxes: list of Tensor[1,4] for each detection
        :param scores: ndarray of scores
        :param iou_threshold: IoU cutoff
        :param use_2007_metric: whether to use VOC2007 method
        :returns: average precision (float)
        """
        tp = np.zeros(len(image_ids), dtype=np.float32)
        fp = np.zeros(len(image_ids), dtype=np.float32)
        matched = set()

        for i, img_id in enumerate(image_ids):
            box = boxes[i]
            if img_id not in gt_boxes:
                fp[i] = 1.0
                continue

            gt = gt_boxes[img_id]
            ious = box_utils.iou_of(box, gt)
            max_iou = float(torch.max(ious))
            max_idx = int(torch.argmax(ious))

            if max_iou > iou_threshold:
                if not difficult_cases[img_id][max_idx]:
                    if (img_id, max_idx) not in matched:
                        tp[i] = 1.0
                        matched.add((img_id, max_idx))
                    else:
                        fp[i] = 1.0
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (num_true_cases + 1e-8)

        if use_2007_metric:
            return measurements.compute_voc2007_average_precision(
                precision, recall
            )
        return measurements.compute_average_precision(precision, recall)

    def group_annotation_by_class(self, dataset):
        """
        Organize all ground‑truth by class.

        :returns: (true_case_stat, all_gt_boxes, all_difficult_cases)
        """
        true_case_stat = {}
        all_gt_boxes = {}
        all_difficult_cases = {}

        for idx in range(len(dataset)):
            image_id, annotation = dataset.get_annotation(idx)
            gt_boxes, classes, is_difficult = annotation
            gt_boxes = torch.from_numpy(gt_boxes)

            for j, cls in enumerate(classes):
                cls_idx = int(cls)
                if not is_difficult[j]:
                    true_case_stat[cls_idx] = (
                        true_case_stat.get(cls_idx, 0) + 1
                    )
                all_gt_boxes.setdefault(cls_idx, {}).setdefault(
                    image_id, []
                ).append(gt_boxes[j])
                all_difficult_cases.setdefault(cls_idx, {}).setdefault(
                    image_id, []
                ).append(is_difficult[j])

        # stack boxes
        for cls_idx in all_gt_boxes:
            for img_id in all_gt_boxes[cls_idx]:
                all_gt_boxes[cls_idx][img_id] = torch.stack(
                    all_gt_boxes[cls_idx][img_id]
                )

        return true_case_stat, all_gt_boxes, all_difficult_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC/Open Images."
    )
    parser.add_argument(
        "--net",
        default="vgg16-ssd",
        help="Network: mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, vgg16-ssd, sq-ssd-lite.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to .pth checkpoint."
    )
    parser.add_argument(
        "--dataset_type",
        default="voc",
        choices=["voc", "open_images"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Root directory of VOC or OpenImages test set.",
    )
    parser.add_argument(
        "--label_file", type=str, required=True, help="Path to labels.txt."
    )
    parser.add_argument(
        "--use_cuda", type=str2bool, default=True, help="Use CUDA if available."
    )
    parser.add_argument(
        "--use_2007_metric",
        type=str2bool,
        default=True,
        help="Use VOC2007 11‑point AP metric.",
    )
    parser.add_argument(
        "--nms_method", default="hard", help="NMS method: hard or blend."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for detection.",
    )
    parser.add_argument(
        "--eval_dir",
        default="models/eval_results",
        help="Directory to store the JSON result.",
    )
    parser.add_argument(
        "--mb2_width_mult",
        type=float,
        default=1.0,
        help="Width multiplier for MobilenetV2‑SSD‑Lite.",
    )

    args = parser.parse_args()

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available() and args.use_cuda
        else torch.device("cpu")
    )
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # load dataset
    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    else:
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    # create network
    num_classes = len(dataset.class_names)
    if args.net == "vgg16-ssd":
        net = create_vgg_ssd(num_classes, is_test=True)
    elif args.net == "mb1-ssd":
        net = create_mobilenetv1_ssd(num_classes, is_test=True)
    elif args.net == "mb1-ssd-lite":
        net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
    elif args.net == "sq-ssd-lite":
        net = create_squeezenet_ssd_lite(num_classes, is_test=True)
    elif args.net == "mb2-ssd-lite":
        net = create_mobilenetv2_ssd_lite(
            num_classes, width_mult=args.mb2_width_mult, is_test=True
        )
    else:
        logging.fatal(f"Unknown network {args.net}")
        sys.exit(1)

    # load weights
    logging.info(f"Loading model {args.model}")
    net.load(args.model)
    net.to(device)
    logging.info("Model loaded")

    # evaluate
    evaluator = MeanAPEvaluator(
        dataset,
        net,
        arch=args.net,
        eval_dir=args.eval_dir,
        nms_method=args.nms_method,
        iou_threshold=args.iou_threshold,
        use_2007_metric=args.use_2007_metric,
        device=device,
    )
    mean_ap, _ = evaluator.compute()

    # extract epoch number from filename
    m = re.search(r"Epoch-(\d+)", args.model)
    if not m:
        logging.error("Could not parse epoch from model filename; defaulting to 0")
        epoch = 0
    else:
        epoch = int(m.group(1))

    # write JSON
    out = {str(epoch): mean_ap}
    json_path = pathlib.Path(args.eval_dir) / f"map_epoch_{epoch}.json"
    with open(json_path, "w") as f:
        json.dump(out, f)

    logging.info(f"Saved mean AP for epoch {epoch} to {json_path}")
