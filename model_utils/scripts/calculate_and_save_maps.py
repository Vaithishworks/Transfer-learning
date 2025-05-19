#!/usr/bin/env python3
"""
SSD batch evaluation with parallel processing using spawn start method.
Script greps all model of the model folder and calculates the map for each model (epoch)
and saves the numbers into a common json file.
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import logging
import pathlib
import re
import sys
import concurrent.futures

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
        nms_method="hard",
        iou_threshold=0.5,
        use_2007_metric=True,
        device="cuda:0",
    ):
        """
        Initialize evaluator.
        """
        self.dataset = dataset
        self.net = net
        self.iou_threshold = iou_threshold
        self.use_2007_metric = use_2007_metric

        (
            self.true_case_stat,
            self.all_gb_boxes,
            self.all_difficult_cases,
        ) = self._group_annotation_by_class(dataset)

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
                f"Invalid architecture '{arch}' – choose from "
                "vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, sq-ssd-lite"
            )

    def compute(self):
        """
        Compute mean AP and per-class AP in memory.
        """
        is_test = self.net.is_test
        self.net.is_test = True

        detections = []
        for idx in range(len(self.dataset)):
            logging.debug(f"Evaluating image {idx+1}/{len(self.dataset)}")
            image = self.dataset.get_image(idx)
            boxes, labels, probs = self.predictor.predict(image)
            batch_idx = torch.full((labels.size(0), 1), idx, dtype=torch.float32)
            det = torch.cat(
                [
                    batch_idx,
                    labels.float().unsqueeze(1),
                    probs.unsqueeze(1),
                    boxes + 1.0,
                ],
                dim=1,
            )
            detections.append(det)

        self.net.is_test = is_test
        all_det = torch.cat(detections, dim=0)

        aps = []
        for cls_idx in range(1, len(self.dataset.class_names)):
            sub = all_det[all_det[:, 1] == cls_idx]
            if sub.numel() == 0:
                aps.append(0.0)
                continue
            order = torch.argsort(sub[:, 2], descending=True)
            sub = sub[order]

            image_ids = [self.dataset.ids[int(x)] for x in sub[:, 0]]
            scores = sub[:, 2].numpy()
            boxes = [(sub[i, 3:] - 1.0).unsqueeze(0) for i in range(sub.size(0))]

            ap = self._compute_ap(
                self.true_case_stat.get(cls_idx, 0),
                self.all_gb_boxes.get(cls_idx, {}),
                self.all_difficult_cases.get(cls_idx, {}),
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
        num_true,
        gt_boxes,
        difficult,
        image_ids,
        boxes,
        scores,
        iou_threshold,
        use_2007_metric,
    ):
        """
        Compute AP for a single class from in-memory predictions.
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
            argmax = int(torch.argmax(ious))

            if max_iou > iou_threshold and not difficult[img_id][argmax]:
                if (img_id, argmax) not in matched:
                    tp[i] = 1.0
                    matched.add((img_id, argmax))
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (num_true + 1e-8)

        if use_2007_metric:
            return measurements.compute_voc2007_average_precision(precision, recall)
        return measurements.compute_average_precision(precision, recall)

    def _group_annotation_by_class(self, dataset):
        """
        Organize ground-truth boxes and difficulty flags by class.
        """
        true_stat = {}
        all_boxes = {}
        all_diff = {}

        for idx in range(len(dataset)):
            image_id, (boxes, classes, is_diff) = dataset.get_annotation(idx)
            boxes = torch.from_numpy(boxes)
            for j, cls in enumerate(classes):
                ci = int(cls)
                if not is_diff[j]:
                    true_stat[ci] = true_stat.get(ci, 0) + 1
                all_boxes.setdefault(ci, {}).setdefault(image_id, []).append(boxes[j])
                all_diff.setdefault(ci, {}).setdefault(image_id, []).append(is_diff[j])

        for ci in all_boxes:
            for img in all_boxes[ci]:
                all_boxes[ci][img] = torch.stack(all_boxes[ci][img])

        return true_stat, all_boxes, all_diff


def evaluate_checkpoint(ckpt_path, args):
    """
    Worker function to evaluate a single checkpoint.
    """
    import torch
    from vision.datasets.voc_dataset import VOCDataset
    from vision.datasets.open_images import OpenImagesDataset

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    else:
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    num_classes = len(dataset.class_names)
    if args.net == "vgg16-ssd":
        net = create_vgg_ssd(num_classes, is_test=True)
    elif args.net == "mb1-ssd":
        net = create_mobilenetv1_ssd(num_classes, is_test=True)
    elif args.net == "mb1-ssd-lite":
        net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
    elif args.net == "sq-ssd-lite":
        net = create_squeezenet_ssd_lite(num_classes, is_test=True)
    else:
        net = create_mobilenetv2_ssd_lite(
            num_classes, width_mult=args.mb2_width_mult, is_test=True
        )

    net.load(ckpt_path)
    net.to(device)

    evaluator = MeanAPEvaluator(
        dataset,
        net,
        arch=args.net,
        nms_method=args.nms_method,
        iou_threshold=args.iou_threshold,
        use_2007_metric=args.use_2007_metric,
        device=device,
    )
    mean_ap, _ = evaluator.compute()

    m = re.search(r"Epoch-(\d+)", ckpt_path)
    epoch = int(m.group(1)) if m else -1
    return epoch, mean_ap


def main():
    """
    Entry point: parses arguments, finds all models, evaluates in parallel, writes JSON.
    """
    parser = argparse.ArgumentParser(description="SSD parallel batch evaluation.")
    parser.add_argument("--net", required=True, help="Network architecture")
    parser.add_argument("--model", required=True, help="Path to any checkpoint in target dir")
    parser.add_argument(
        "--dataset_type", default="voc", choices=["voc", "open_images"], help="Dataset type"
    )
    parser.add_argument("--dataset", required=True, help="Root directory of test set")
    parser.add_argument("--label_file", required=True, help="Path to labels.txt")
    parser.add_argument("--use_cuda", type=str2bool, default=True, help="Use CUDA")
    parser.add_argument(
        "--use_2007_metric", type=str2bool, default=True, help="Use VOC2007 metric"
    )
    parser.add_argument("--nms_method", default="hard", help="NMS method")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument(
        "--eval_dir", default="models/eval_results", help="Directory for map.json"
    )
    parser.add_argument(
        "--mb2_width_mult", type=float, default=1.0, help="MobilenetV2 width multiplier"
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(Y-%m-%d %H:%M:%S) - %(message)s",
    )

    model_dir = pathlib.Path(args.model).parent
    pattern = f"{args.net}-Epoch-*.pth"
    files = sorted(
        model_dir.glob(pattern),
        key=lambda p: int(re.search(r"Epoch-(\d+)", p.name).group(1)),
    )

    max_workers = min(len(files), multiprocessing.cpu_count())
    spawn_ctx = multiprocessing.get_context("spawn")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=spawn_ctx
    ) as executor:
        futures = {executor.submit(evaluate_checkpoint, str(ckpt), args): ckpt for ckpt in files}
        for future in concurrent.futures.as_completed(futures):
            epoch, mean_ap = future.result()
            results[str(epoch)] = mean_ap

    out_dir = pathlib.Path(args.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "map.json"
    with open(json_path, "w") as f:
        json.dump(results, f)

    logging.info(f"Wrote combined mAP JSON to {json_path}")


if __name__ == "__main__":
    main()
