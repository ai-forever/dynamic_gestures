import numpy as np


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """
    Convert regression location results of SSD into boxes in the form of (center_x, center_y, h, w).
    Parameters
    ----------
    locations: numpy.ndarray
        Regression location results, sized [num_priors,4].
    priors: numpy.ndarray
        Prior boxes in center-offset form, sized [num_priors,4].
    center_variance: float
        The center variance for decoding.
    size_variance: float
        The size variance for decoding.

    Returns
    -------
    boxes: numpy.ndarray
        Boxes in corner form, sized [num_priors,4].
    """
    # priors can have one dimension less.
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate(
        [
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:],
        ],
        axis=len(locations.shape) - 1,
    )


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    """
    Convert boxes to locations with respect to priors, which are encoded as (cx, cy, w, h).
    Parameters
    ----------
    center_form_boxes: numpy.ndarray
        Boxes to be converted to locations, sized [num_priors,4].
    center_form_priors: numpy.ndarray
        Prior boxes in center-form, sized [num_priors,4].
    center_variance: float
        The center variance for encoding.
    size_variance: float
        The size variance for encoding.

    Returns
    -------
    locations: numpy.ndarray
        Encoded locations, sized [num_priors,4].
    """
    if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
        center_form_priors = np.expand_dims(center_form_priors, 0)
    return np.concatenate(
        [
            (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
            np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance,
        ],
        axis=len(center_form_boxes.shape) - 1,
    )


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Parameters
    ----------
    left_top: numpy.ndarray
        Left top corner of the rectangles, sized [N,2].
    right_bottom: numpy.ndarray
        Right bottom corner of the rectangles, sized [N,2].

    Returns
    -------
    area: numpy.ndarray
        Computed areas, sized [N,].
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def center_form_to_corner_form(locations):
    """
    Convert center-form boxes to corner-form.
    Parameters
    ----------
    locations: numpy.ndarray
        Center-form boxes to be converted to corner-form, sized [num_priors,4].

    Returns
    -------
    boxes: numpy.ndarray
        Corner-form boxes, sized [num_priors,4].
    """
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2],
        len(locations.shape) - 1,
    )


def corner_form_to_center_form(boxes):
    """
    Convert corner-form boxes to center-form.
    Parameters
    ----------
    boxes: numpy.ndarray
        Corner-form boxes to be converted to center-form, sized [num_priors,4].

    Returns
    -------
    locations: numpy.ndarray
        Center-form boxes, sized [num_priors,4].
    """
    return np.concatenate(
        [(boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]], len(boxes.shape) - 1
    )


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Parameters
    ----------
    box_scores: numpy.ndarray
        boxes in corner-form and probabilities.
    iou_threshold: float
        intersection over union threshold.
    top_k: int
        keep top_k results. If k <= 0, keep all the results.
    candidate_size: int
        only consider the candidates with the highest scores.

    Returns
    -------
    picked: numpy.ndarray
        a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]
