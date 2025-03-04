import argparse
import time

import cv2
import numpy as np

from main_controller import MainController
from utils import Drawer, Event, targets


def run(args):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    controller = MainController(args.detector, args.classifier)
    drawer = Drawer()
    debug_mode = args.debug
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            start_time = time.time()
            bboxes, ids, labels = controller(frame)
            if debug_mode:
                if bboxes is not None:
                    bboxes = bboxes.astype(np.int32)
                    for i in range(bboxes.shape[0]):
                        box = bboxes[i, :]
                        gesture = targets[labels[i]] if labels[i] is not None else "None"
                        
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                        cv2.putText(
                            frame,
                            f"ID {ids[i]} : {gesture}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"fps {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if len(controller.tracks) > 0:
                count_of_zoom = 0
                thumb_boxes = []
                for trk in controller.tracks:
                    if trk["tracker"].time_since_update < 1:
                        if len(trk['hands']):
                            count_of_zoom += (trk['hands'][-1].gesture == 3)

                            thumb_boxes.append(trk['hands'][-1].bbox)
                            if len(trk['hands']) > 3 and [trk['hands'][-1].gesture, trk['hands'][-2].gesture, trk['hands'][-3].gesture] == [23, 23, 23]:
                                x, y, x2, y2 = map(int, trk['hands'][-1].bbox)
                                x, y, x2, y2 = max(x, 0), max(y, 0), max(x2, 0), max(y2, 0)
                                bbox_area = frame[y:y2, x:x2]
                                blurred_bbox = cv2.GaussianBlur(bbox_area, (51, 51), 10)
                                frame[y:y2, x:x2] = blurred_bbox

                        if trk["hands"].action is not None:
                            if Event.SWIPE_LEFT == trk["hands"].action or  Event.SWIPE_LEFT2 == trk["hands"].action or  Event.SWIPE_LEFT3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_RIGHT == trk["hands"].action or Event.SWIPE_RIGHT2 == trk["hands"].action or Event.SWIPE_RIGHT3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_UP == trk["hands"].action or Event.SWIPE_UP2 == trk["hands"].action or Event.SWIPE_UP3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_DOWN == trk["hands"].action or Event.SWIPE_DOWN2 == trk["hands"].action or Event.SWIPE_DOWN3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DRAG == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                ...
                            elif Event.DROP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.FAST_SWIPE_DOWN == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.FAST_SWIPE_UP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.ZOOM_IN == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.ZOOM_OUT == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DOUBLE_TAP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DRAG2 == trk["hands"].action or Event.DRAG3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                ...
                            elif Event.DROP2 == trk["hands"].action or Event.DROP3 == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.TAP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.COUNTERCLOCK == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.CLOCKWISE == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                                
                if count_of_zoom == 2:
                    drawer.draw_two_hands(frame, thumb_boxes)
            if debug_mode:
                frame = drawer.draw(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument(
        "--detector",
        default='models/hand_detector.onnx',
        type=str,
        help="Path to detector onnx model"
    )

    parser.add_argument(
        "--classifier",
        default='models/crops_classifier.onnx',
        type=str,
        help="Path to classifier onnx model",
    )

    parser.add_argument("--debug", required=False, action="store_true", help="Debug mode")
    args = parser.parse_args()
    run(args)
