import numpy as np

from ocsort import (
    KalmanBoxTracker,
    associate,
    ciou_batch,
    ct_dist,
    diou_batch,
    giou_batch,
    iou_batch,
    linear_assignment,
)
from onnx_models import HandClassification, HandDetection
from utils import Deque, Drawer, Hand

ASSO_FUNCS = {"iou": iou_batch, "giou": giou_batch, "ciou": ciou_batch, "diou": diou_batch, "ct_dist": ct_dist}


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


class MainController:
    """
    Main tracking function.
    Class contains a list of tracks, each track contains a KalmanBoxTracker object and a Deque object with Hand objects.
    """

    def __init__(
        self, detection_model, classification_model, max_age=30, min_hits=3, iou_threshold=0.3, maxlen=30, min_frames=20
    ):
        """
        Parameters
        ----------
        detection_model : str
            Path to detection model.
        classification_model : str
            Path to classification model.
        max_age : int
            Maximum age of track.
        min_hits : int
            Minimum number of hits to confirm track.
        iou_threshold : float
            IOU threshold for track association.
        maxlen : int
            Maximum length of deque in track.
        min_frames : int
            Minimum number of frames to confirm track.
        """
        self.maxlen = maxlen
        self.min_frames = min_frames
        self.max_age = max_age
        self.min_hits = min_hits
        self.delta_t = 3
        self.iou_threshold = iou_threshold
        self.inertia = 0.2
        self.asso_func = ASSO_FUNCS["giou"]
        self.tracks = []
        self.frame_count = 0
        self.detection_model = HandDetection(detection_model)
        self.classification_model = HandClassification(classification_model)
        self.drawer = Drawer()

    def update(self, dets=np.empty((0, 5)), labels=None):
        """
        Parameters
        ----------
        dets : np.array
            Bounding boxes with shape [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...] .
            Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        labels : np.array
            Labels with shape (N, 1) where N is number of bounding boxes.

        Returns
        -------
        np.array
            Returns the similar array, where the last column is the object ID.

        Notes
        -----
        The number of objects returned may differ from the number of detections provided.

        """
        if len(dets) == 0:
            for trk in self.tracks:
                trk["hands"].append(Hand(bbox=None, gesture=None))
            return

        self.frame_count += 1

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        lbs = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t]["tracker"].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        velocities = np.array(
            [
                trk["tracker"].velocity if trk["tracker"].velocity is not None else np.array((0, 0))
                for trk in self.tracks
            ]
        )
        last_boxes = np.array([trk["tracker"].last_observation for trk in self.tracks])
        k_observations = np.array(
            [k_previous_obs(trk["tracker"].observations, trk["tracker"].age, self.delta_t) for trk in self.tracks]
        )

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )

        for m in matched:
            self.tracks[m[1]]["tracker"].update(dets[m[0], :])
            self.tracks[m[1]]["hands"].append(Hand(bbox=dets[m[0], :4], gesture=labels[m[0]]))

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.tracks[trk_ind]["tracker"].update(dets[det_ind, :])
                    self.tracks[trk_ind]["hands"].append(Hand(bbox=dets[det_ind, :4], gesture=labels[det_ind]))
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.tracks[m]["tracker"].update(None)
            self.tracks[m]["hands"].append(Hand(bbox=None, gesture=None))

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            self.tracks.append(
                {
                    "hands": Deque(self.maxlen, self.min_frames),
                    "tracker": KalmanBoxTracker(dets[i, :], delta_t=self.delta_t),
                }
            )
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            if trk["tracker"].last_observation.sum() < 0:
                d = trk["tracker"].get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk["tracker"].last_observation[:4]
            if (trk["tracker"].time_since_update < 1) and (
                trk["tracker"].hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk["tracker"].id + 1])).reshape(1, -1))
                if len(trk["hands"]) > 0:
                    lbs.append(trk["hands"][-1].gesture)
                else:
                    lbs.append(None)

            i -= 1
            # remove dead tracklet
            if trk["tracker"].time_since_update > self.max_age:
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret), lbs
        return np.empty((0, 5)), np.empty((0, 1))

    def __call__(self, frame):
        """
        Parameters
        ----------
        frame : np.array
            Image frame with shape (H, W, 3).

        Returns
        -------
        list of np.array


        """
        bboxes, probs = self.detection_model(frame)
        if len(bboxes):
            labels = self.classification_model(frame, bboxes)
            bboxes = np.concatenate((bboxes, np.expand_dims(probs, axis=1)), axis=1)
            new_bboxes, labels = self.update(dets=bboxes, labels=labels)
            return new_bboxes[:, :-1], new_bboxes[:, -1], labels
        else:
            self.update(np.empty((0, 5)), None)
            return None, None, None
