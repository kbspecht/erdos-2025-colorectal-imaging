import json
import os
import tempfile

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# # @torch.no_grad()
# # def coco_map(model, loader, device, coco_gt_json):
# #     model.eval()
# #     results = []
# #     for images, targets in tqdm(loader, desc="COCO eval", ncols=100):
# #         images = [im.to(device) for im in images]
# #         outputs = model(images)
# #         for out, tgt in zip(outputs, targets):
# #             img_id = int(tgt["image_id"].item())
# #             b = out["boxes"].detach().cpu().numpy()
# #             s = out["scores"].detach().cpu().numpy()
# #             l = out["labels"].detach().cpu().numpy()

# #             if b.shape[0] > 0:
# #                 xywh = b.copy()
# #                 xywh[:, 2] = b[:, 2] - b[:, 0]  # w
# #                 xywh[:, 3] = b[:, 3] - b[:, 1]  # h
# #                 xywh[:, 0] = b[:, 0]  # x
# #                 xywh[:, 1] = b[:, 1]  # y
# #             else:
# #                 xywh = b

# #             for bb, ss, ll in zip(xywh, s, l):
# #                 results.append(
# #                     {
# #                         "image_id": img_id,
# #                         "category_id": int(ll),
# #                         "bbox": [
# #                             float(bb[0]),
# #                             float(bb[1]),
# #                             float(bb[2]),
# #                             float(bb[3]),
# #                         ],
# #                         "score": float(ss),
# #                     }
# #                 )

# #     coco_gt = COCO(coco_gt_json)
# #     coco_dt = coco_gt.loadRes(results) if len(results) else coco_gt.loadRes([])
# #     ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
# #     ev.evaluate()
# #     ev.accumulate()
# #     ev.summarize()
# #     return ev.stats
# #     # [AP, AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100, AR_s, AR_m, AR_l]


# @torch.no_grad()
# def coco_map(model, loader, device, coco_gt_json):
#     """
#     Compute COCO-style mAP given:
#       - model: Faster R-CNN model
#       - loader: DataLoader yielding (images, targets)
#       - device: torch.device
#       - coco_gt_json: path to GT annotations in (almost) COCO format

#     This function is robust to missing 'info' / 'licenses' in the GT JSON.
#     """
#     model.eval()
#     results = []

#     # -------- collect detection results --------
#     for images, targets in tqdm(loader, desc="COCO eval", ncols=100):
#         images = [im.to(device) for im in images]
#         outputs = model(images)

#         for out, tgt in zip(outputs, targets):
#             img_id = int(tgt["image_id"].item())
#             b = out["boxes"].detach().cpu().numpy()
#             s = out["scores"].detach().cpu().numpy()
#             l = out["labels"].detach().cpu().numpy()

#             if b.shape[0] > 0:
#                 xywh = b.copy()
#                 xywh[:, 2] = b[:, 2] - b[:, 0]  # w
#                 xywh[:, 3] = b[:, 3] - b[:, 1]  # h
#                 xywh[:, 0] = b[:, 0]  # x
#                 xywh[:, 1] = b[:, 1]  # y
#             else:
#                 xywh = b

#             for bb, ss, ll in zip(xywh, s, l):
#                 results.append(
#                     {
#                         "image_id": img_id,
#                         "category_id": int(ll),
#                         "bbox": [
#                             float(bb[0]),
#                             float(bb[1]),
#                             float(bb[2]),
#                             float(bb[3]),
#                         ],
#                         "score": float(ss),
#                     }
#                 )

#     # -------- load and sanitize GT JSON --------
#     with open(coco_gt_json, "r") as f:
#         data = json.load(f)

#     # ensure keys pycocotools expects
#     if "info" not in data:
#         data["info"] = {"description": "polyp dataset"}
#     if "licenses" not in data:
#         data["licenses"] = []

#     # write a temporary fixed JSON for pycocotools
#     with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
#         json.dump(data, tmp)
#         tmp_anno_path = tmp.name

#     # -------- run COCO eval --------
#     try:
#         coco_gt = COCO(tmp_anno_path)
#         coco_dt = coco_gt.loadRes(results) if len(results) else coco_gt.loadRes([])
#         ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
#         ev.evaluate()
#         ev.accumulate()
#         ev.summarize()
#         return (
#             ev.stats
#         )  # [AP, AP50, AP75, AP_s, AP_m, AP_l, AR1, AR10, AR100, AR_s, AR_m, AR_l]
#     finally:
#         # clean up temp file
#         if os.path.exists(tmp_anno_path):
#             os.remove(tmp_anno_path)

# import json
# import os
# import tempfile

# import torch
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from tqdm import tqdm


# @torch.no_grad()
# def coco_map(model, loader, device, coco_gt_json, verbose=True):
#     """
#     Run COCO-style evaluation and return a compact metrics dict:

#         {
#             "precision":  <mean precision>,
#             "recall":     <AR@[.5:.95], maxDets=100>,
#             "mAP_50":     <AP@0.50>,
#             "mAP_50_95":  <AP@[.5:.95]>,
#         }

#     Assumes:
#       - `loader` yields (images, targets)
#       - targets contain "image_id" (int-like) per sample
#       - model outputs torchvision-style dicts with "boxes", "scores", "labels"
#       - boxes are in absolute xyxy pixel coords matching the COCO GT JSON
#     """
#     model.eval()
#     results = []

#     # -------- collect detection results --------
#     for images, targets in tqdm(loader, desc="COCO eval", ncols=100):
#         images = [im.to(device) for im in images]
#         outputs = model(images)

#         for out, tgt in zip(outputs, targets):
#             img_id = int(tgt["image_id"].item())
#             b = out["boxes"].detach().cpu().numpy()  # [N, 4] xyxy
#             s = out["scores"].detach().cpu().numpy()  # [N]
#             l = out["labels"].detach().cpu().numpy()  # [N]

#             # convert xyxy -> xywh (COCO format)
#             if b.shape[0] > 0:
#                 xywh = b.copy()
#                 xywh[:, 2] = b[:, 2] - b[:, 0]  # w
#                 xywh[:, 3] = b[:, 3] - b[:, 1]  # h
#                 xywh[:, 0] = b[:, 0]  # x
#                 xywh[:, 1] = b[:, 1]  # y
#             else:
#                 xywh = b

#             for bb, ss, ll in zip(xywh, s, l):
#                 results.append(
#                     {
#                         "image_id": img_id,
#                         "category_id": int(ll),
#                         "bbox": [
#                             float(bb[0]),
#                             float(bb[1]),
#                             float(bb[2]),
#                             float(bb[3]),
#                         ],
#                         "score": float(ss),
#                     }
#                 )

#     # -------- load and sanitize GT JSON --------
#     with open(coco_gt_json, "r") as f:
#         data = json.load(f)

#     # ensure keys pycocotools expects
#     if "info" not in data:
#         data["info"] = {"description": "dataset"}
#     if "licenses" not in data:
#         data["licenses"] = []

#     # write a temporary fixed JSON for pycocotools
#     with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
#         json.dump(data, tmp)
#         tmp_anno_path = tmp.name

#     try:
#         coco_gt = COCO(tmp_anno_path)
#         coco_dt = coco_gt.loadRes(results) if len(results) else coco_gt.loadRes([])
#         ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
#         ev.evaluate()
#         ev.accumulate()
#         if verbose:
#             ev.summarize()

#         # COCO 'stats' vector:
#         # stats[0] = AP@[.5:.95] (mAP_50_95)
#         # stats[1] = AP@0.5      (mAP_50)
#         # stats[8] = AR@[.5:.95], maxDets=100 (recall)
#         mAP_50_95 = float(ev.stats[0])
#         mAP_50 = float(ev.stats[1])
#         recall = float(ev.stats[8])

#         # COCO precision tensor: [T, R, K, A, M]
#         # T: IoU thresholds, R: recall thresholds, K: classes,
#         # A: areas, M: maxDets
#         # we average over all IoUs, recalls, classes, area=all (0), maxDet=100 (last index)
#         precisions = ev.eval["precision"]  # shape [T, R, K, A, M]
#         if precisions.size == 0:
#             precision = float("nan")
#         else:
#             p = precisions[..., 0, -1]  # all t, r, k; area=all; maxDet=100
#             p = p[p > -1]  # drop invalid entries
#             precision = float(p.mean()) if p.size > 0 else float("nan")

#         return {
#             "precision": precision,
#             "recall": recall,
#             "mAP_50": mAP_50,
#             "mAP_50_95": mAP_50_95,
#         }

#     finally:
#         if os.path.exists(tmp_anno_path):
#             os.remove(tmp_anno_path)


@torch.no_grad()
def coco_map(model, loader, device, coco_gt_json, verbose=True, log_every=10):
    """
    Run COCO-style evaluation and return a compact metrics dict:

        {
            "precision":  <mean precision>,
            "recall":     <AR@[.5:.95], maxDets=100>,
            "mAP_50":     <AP@0.50>,
            "mAP_50_95":  <AP@[.5:.95]>,
        }

    Also prints:
      - tqdm batch progress bar
      - every `log_every` batches: batch index, #detections so far, top scores

    Assumes:
      - `loader` yields (images, targets)
      - targets contain "image_id" (int-like) per sample
      - model outputs torchvision-style dicts with "boxes", "scores", "labels"
      - boxes are in absolute xyxy pixel coords matching the COCO GT JSON
    """
    model.eval()
    results = []

    # -------- main loop: collect detection results --------
    num_batches = len(loader)

    for i, (images, targets) in enumerate(tqdm(loader, desc="COCO eval", ncols=100)):
        images = [im.to(device) for im in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            b = out["boxes"].detach().cpu().numpy()  # [N, 4] xyxy
            s = out["scores"].detach().cpu().numpy()  # [N]
            l = out["labels"].detach().cpu().numpy()  # [N]

            # convert xyxy -> xywh (COCO format)
            if b.shape[0] > 0:
                xywh = b.copy()
                xywh[:, 2] = b[:, 2] - b[:, 0]  # w
                xywh[:, 3] = b[:, 3] - b[:, 1]  # h
                xywh[:, 0] = b[:, 0]  # x
                xywh[:, 1] = b[:, 1]  # y
            else:
                xywh = b

            for bb, ss, ll in zip(xywh, s, l):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(ll),
                        "bbox": [
                            float(bb[0]),
                            float(bb[1]),
                            float(bb[2]),
                            float(bb[3]),
                        ],
                        "score": float(ss),
                    }
                )

        # -------- safe per-batch logging (no COCOeval here) --------
        if (i + 1) % log_every == 0 or (i + 1) == num_batches:
            print(
                f"[batch {i+1}/{num_batches}] " f"cumulative detections: {len(results)}"
            )
            if outputs:
                # just to get a feel for confidence levels
                scores = outputs[0]["scores"].detach().cpu().numpy()
                if scores.size > 0:
                    top = scores[:3]
                    print(f"   top scores this batch: {top}")

    # -------- handle no detections case safely --------
    if len(results) == 0:
        if verbose:
            print("No detections collected; returning NaN metrics.")
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "mAP_50": float("nan"),
            "mAP_50_95": float("nan"),
        }

    # -------- load and sanitize GT JSON --------
    with open(coco_gt_json, "r") as f:
        data = json.load(f)

    if "info" not in data:
        data["info"] = {"description": "polyp dataset"}
    if "licenses" not in data:
        data["licenses"] = []

    # write a temporary fixed JSON for pycocotools
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp)
        tmp_anno_path = tmp.name

    try:
        coco_gt = COCO(tmp_anno_path)
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        if verbose:
            ev.summarize()

        # COCO 'stats' vector:
        # stats[0] = AP@[.5:.95] (mAP_50_95)
        # stats[1] = AP@0.5      (mAP_50)
        # stats[8] = AR@[.5:.95], maxDets=100 (recall)
        mAP_50_95 = float(ev.stats[0])
        mAP_50 = float(ev.stats[1])
        recall = float(ev.stats[8])

        # COCO precision tensor: [T, R, K, A, M]
        precisions = ev.eval["precision"]
        if precisions.size == 0:
            precision = float("nan")
        else:
            # all IoUs, all recalls, all classes, area=all (0), maxDet=100 (last)
            p = precisions[..., 0, -1]
            p = p[p > -1]  # drop invalid entries
            precision = float(p.mean()) if p.size > 0 else float("nan")

        return {
            "precision": precision,
            "recall": recall,
            "mAP_50": mAP_50,
            "mAP_50_95": mAP_50_95,
        }

    finally:
        if os.path.exists(tmp_anno_path):
            os.remove(tmp_anno_path)
