import cv2

from utils import Event


class Drawer:
    def __init__(self):
        self.height = self.width = None
        self.action = None
        self.show_delay = 0

    def set_action(self, action):
        """
        Set action to draw

        Parameters
        ----------
        action : Event
            Action to draw
        """
        self.action = action
        self.show_delay = 0

    def draw(self, frame):
        """
        Draw action on frame

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on
        x : int
            X coordinate of hand center
        y : int
            Y coordinate of hand center

        Returns
        -------
        frame : np.ndarray
            Frame with action

        """
        if self.height is None:
            self.height, self.width, _ = frame.shape
        if self.action is not None:
            if self.action == Event.SWIPE_LEFT:
                frame = cv2.arrowedLine(
                    frame,
                    (int(self.width * 0.6), self.height // 2),
                    (int(self.width * 0.4), self.height // 2),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.SWIPE_RIGHT:
                frame = cv2.arrowedLine(
                    frame,
                    (int(self.width * 0.4), self.height // 2),
                    (int(self.width * 0.6), self.height // 2),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.SWIPE_UP:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.6)),
                    (self.width // 2, int(self.height * 0.4)),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.SWIPE_DOWN:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.4)),
                    (self.width // 2, int(self.height * 0.6)),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.DRAG:
                frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (0, 255, 0), 9)
            elif self.action == Event.DROP:
                frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (0, 0, 255), -1)
            self.show_delay += 1
            if self.show_delay > 10:
                self.show_delay = 0
                self.action = None
                self.x = self.y = None

        return frame
