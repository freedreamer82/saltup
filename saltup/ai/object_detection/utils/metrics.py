import numpy as np

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using the precision-recall curve.

    Args:
        recall: Array of recall values.
        precision: Array of precision values.

    Returns:
        Average Precision (AP) as a float.
    """
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate the area under the PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_ap_range(gt_bboxes: List[BBox], pred_bboxes_scores: List[Tuple[BBox, float]]) -> float:
    """
    Compute AP over a range of IoU thresholds (0.5 to 0.95 with step 0.05).

    Args:
        gt_bboxes: List of ground truth bounding boxes.
        pred_bboxes_scores: List of predicted bounding boxes and their confidence scores.

    Returns:
        AP averaged over IoU thresholds from 0.5 to 0.95.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for threshold in iou_thresholds:
        ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold)
        aps.append(ap)

    return np.mean(aps)


def compute_ap_for_threshold(gt_bboxes: List[BBox], pred_bboxes_scores: List[Tuple[BBox, float]], threshold: float) -> float:
    """
    Compute AP for a specific IoU threshold.

    Args:
        gt_bboxes: List of ground truth bounding boxes.
        pred_bboxes_scores: List of predicted bounding boxes and their confidence scores.
        threshold: IoU threshold.

    Returns:
        AP for the given threshold.
    """
    # Sort predictions by confidence score (descending)
    pred_bboxes_scores.sort(key=lambda x: x[1], reverse=True)
    pred_bboxes = [x[0] for x in pred_bboxes_scores]

    # Initialize TP and FP arrays
    tp = np.zeros(len(pred_bboxes))
    fp = np.zeros(len(pred_bboxes))

    # Match predictions to ground truth
    gt_matched = [False] * len(gt_bboxes)

    for i, pred_bbox in enumerate(pred_bboxes):
        max_iou = 0
        best_match_idx = -1

        # Find the ground truth box with the highest IoU
        for j, gt_bbox in enumerate(gt_bboxes):
            iou = pred_bbox.compute_iou(gt_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match_idx = j

        # If IoU > threshold and the ground truth box is not already matched, it's a TP
        if max_iou >= threshold and not gt_matched[best_match_idx]:
            tp[i] = 1
            gt_matched[best_match_idx] = True
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / len(gt_bboxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP
    ap = compute_ap(recall, precision)
    return ap