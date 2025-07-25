import numpy as np
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from typing import List, Tuple, Union
 

class Metric:
    def __init__(self):
        """
        Initialize the class Metric.
        """
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives

    def addTP(self, count=1):
        """Add a specified number of True Positives (TP)."""
        self.tp += count

    def addFP(self, count=1):
        """Add a specified number of False Positives (FP)."""
        self.fp += count

    def addFN(self, count=1):
        """Add a specified number of False Negatives (FN)."""
        self.fn += count
        
    def addTN(self, count=1):
        """Add a specified number of True Negatives (TN)."""
        self.tn += count

    def getTP(self):
        """Returns the number of True Positives (TP)."""
        return self.tp

    def getFP(self):
        """Returns the number of False Positives (FP)."""
        return self.fp
    
    def getTN(self):
        """Returns the number of True Negatives (TN)."""
        return self.tn
    
    def getFN(self):
        """Returns the number of False Negatives (FN)."""
        return self.fn
    
    def accuracy(self):
        """Calculate the accuracy."""
        total_samples = self.tp + self.fp + self.fn + self.tn
        return self.tp / total_samples if total_samples != 0 else 0
    
    def precision(self):
        """Calculate the precision."""
        denom = self.tp + self.fp
        return self.tp / denom if denom != 0 else 0

    def recall(self):
        """Calculate the recall."""
        denom = self.tp + self.fn
        return self.tp / denom if denom != 0 else 0

    def f1_score(self):
        """Calculate the F1-score."""
        p = self.precision()
        r = self.recall()
        denom = p + r
        return 2 * (p * r) / denom if denom != 0 else 0
    
    def getAccuracy(self):
        """Return the accuracy."""
        return self.accuracy()
    
    def getPrecision(self):
        """Return the precision."""
        return self.precision()

    def getRecall(self):
        """Return the recall."""
        return self.recall()

    def getF1Score(self):
        """Return the F1-score."""
        return self.f1_score()

    def get_metrics(self):
        """Returns a dictionary with all calculated metrics."""
        return {
            'tp': self.getTP(),
            'fp': self.getFP(),
            'tn': self.getTN(),
            'fn': self.getFN(),
            'accuracy': self.getAccuracy(),
            'precision': self.getPrecision(),
            'recall': self.getRecall(),
            'f1_score': self.getF1Score(),
        }

    def __add__(self, other):
        if not isinstance(other, Metric):
            return NotImplemented
        result = Metric()
        result.tp = self.tp + other.tp
        result.fp = self.fp + other.fp
        result.tn = self.tn + other.tn
        result.fn = self.fn + other.fn
        return result

    def __sub__(self, other):
        if not isinstance(other, Metric):
            return NotImplemented
        result = Metric()
        result.tp = self.tp - other.tp
        result.fp = self.fp - other.fp
        result.tn = self.tn - other.tn
        result.fn = self.fn - other.fn
        return result

    def __iadd__(self, other):
        if not isinstance(other, Metric):
            return NotImplemented
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        return self

    def __isub__(self, other):
        if not isinstance(other, Metric):
            return NotImplemented
        self.tp -= other.tp
        self.fp -= other.fp
        self.tn -= other.tn
        self.fn -= other.fn
        return self

    def __str__(self):
        """Return a string representation for pretty printing."""
        metrics = self.get_metrics()
        return (
            f"Metric(\n"
            f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}\n"
            f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
            f"F1-score: {metrics['f1_score']:.4f}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using the precision-recall curve.

    Args:
        recall: Array of recall values.
        precision: Array of precision values.

    Returns:
        Average Precision (AP) as a float.
    """
    # Append sentinel values at the beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = np.maximum(mpre[i], mpre[i + 1])

    # Identify indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Calculate the area under the PR curve using the trapezoidal rule
    ap = np.sum((mrec[i + 1] - mrec[i]) * (mpre[i + 1] + mpre[i]) / 2)
    return ap




def compute_map_50_95(gt_bboxes: List[BBox], pred_bboxes_scores: List[Tuple[BBox, float]]) -> float:
    """
    Compute AP over a range of IoU thresholds (0.5 to 0.95 with step 0.05).

    Args:
        gt_bboxes: List of ground truth bounding boxes.
        pred_bboxes_scores: List of predicted bounding boxes and their confidence scores.

    Returns:
        AP averaged over IoU thresholds from 0.5 to 0.95.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # IoU thresholds from 0.50 to 0.95
    aps = []

    for threshold in iou_thresholds:
        ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, threshold)
        aps.append(ap)

    return np.mean(aps)  # Mean of APs for all IoU thresholds

def compute_map_50(gt_bboxes: List[BBox], pred_bboxes_scores: List[Tuple[BBox, float]]) -> float:
    """
    Compute AP at IoU threshold 0.5.

    Args:
        gt_bboxes: List of ground truth bounding boxes.
        pred_bboxes_scores: List of predicted bounding boxes and their confidence scores.

    Returns:
        AP at IoU threshold 0.5.
    """

    ap = compute_ap_for_threshold(gt_bboxes, pred_bboxes_scores, 0.5)

    return np.mean(ap)  # Mean of APs for all IoU thresholds


def compute_ap_for_threshold(gt_bboxes: List[BBox], pred_bboxes_scores: List[Tuple[BBox, float]], threshold: float) -> float:
    # Sort predictions by confidence score (descending)
    pred_bboxes_scores.sort(key=lambda x: x[1], reverse=True)
    pred_bboxes = [x[0] for x in pred_bboxes_scores]

    # Initialize TP and FP arrays
    tp = np.zeros(len(pred_bboxes))
    fp = np.zeros(len(pred_bboxes))

    # Initialize array to keep track of already matched ground truths
    gt_matched = [False] * len(gt_bboxes)

    # Match predictions to ground truths
    for i, pred_bbox in enumerate(pred_bboxes):
        max_iou = 0
        best_match_idx = -1

        # Find the ground truth with the highest IoU
        for j, gt_bbox in enumerate(gt_bboxes):
            if gt_matched[j]:
                continue  # Skip already matched ground truths
            iou = pred_bbox.compute_iou(gt_bbox)
            if iou > max_iou:
                max_iou = iou
                best_match_idx = j

        # If IoU > threshold and the ground truth is not already matched, it's a TP
        if max_iou >= threshold and best_match_idx != -1:
            tp[i] = 1
            gt_matched[best_match_idx] = True  # Mark the ground truth as matched
        else:
            fp[i] = 1  # Otherwise, it's a FP

    # Compute cumulative recall and precision
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute recall
    recall = tp_cumsum / len(gt_bboxes) if len(gt_bboxes) > 0 else np.zeros_like(tp_cumsum)

    # Compute precision
    denominator = tp_cumsum + fp_cumsum
    precision = np.where(denominator > 0, tp_cumsum / denominator, np.zeros_like(tp_cumsum))

    # Compute AP using the compute_ap function
    ap = compute_ap(recall, precision)
    return ap