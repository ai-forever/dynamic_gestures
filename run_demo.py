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
                for trk in controller.tracks:
                    if trk["tracker"].time_since_update < 1:
                        if trk["hands"].action is not None:
                            if Event.SWIPE_LEFT == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_RIGHT == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_UP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.SWIPE_DOWN == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None
                                ...
                            elif Event.DRAG == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                ...
                            elif Event.DROP == trk["hands"].action:
                                drawer.set_action(trk["hands"].action)
                                trk["hands"].action = None

            if debug_mode:
                frame = drawer.draw(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run demo")
    parser.add_argument("--detector", required=True, type=str, help="Path to detector onnx model")
    parser.add_argument(
        "--classifier",
        required=True,
        type=str,
        help="Path to classifier onnx model",
    )
    parser.add_argument("--debug", required=False, action="store_true", help="Debug mode")
    args = parser.parse_args()
    run(args)
