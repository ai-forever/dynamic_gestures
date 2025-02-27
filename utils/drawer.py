import cv2

from .enums import Event


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

    def draw_two_hands(self, frame, bboxes):
        self.height, self.width, _ = frame.shape
        center_x1, center_y1 = bboxes[0][0] + (bboxes[0][2] - bboxes[0][0]) // 2, bboxes[0][1] + (bboxes[0][3] - bboxes[0][1]) // 2
        center_x2, center_y2 = bboxes[1][0] + (bboxes[1][2] - bboxes[1][0]) // 2, bboxes[1][1] + (bboxes[1][3] - bboxes[1][1]) // 2
        # frame = cv2.circle(frame, (int(center_x1), int(center_y1)), 50, (255, 0, 0), 9)
        # frame = cv2.circle(frame, (int(center_x2), int(center_y2)), 50, (255, 0, 0), 9)

        diff = int(center_x1 - center_x2)

        frame = cv2.rectangle(frame,
                              (int(center_x1), int(center_y1 - diff * 0.3)),
                              (int(center_x2), int(center_y2 + diff * 0.3)),
                              (255, 0, 0), 5)

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
            if self.action in [Event.SWIPE_LEFT, Event.SWIPE_LEFT2, Event.SWIPE_LEFT3]:
                frame = cv2.arrowedLine(
                    frame,
                    (int(self.width * 0.6), self.height // 2),
                    (int(self.width * 0.4), self.height // 2),
                    (0, 255, 0),
                    9,
                )
            elif self.action in [Event.SWIPE_RIGHT, Event.SWIPE_RIGHT2, Event.SWIPE_RIGHT3]:
                frame = cv2.arrowedLine(
                    frame,
                    (int(self.width * 0.4), self.height // 2),
                    (int(self.width * 0.6), self.height // 2),
                    (0, 255, 0),
                    9,
                )
            elif self.action in [Event.SWIPE_UP, Event.SWIPE_UP2, Event.SWIPE_UP3]:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.6)),
                    (self.width // 2, int(self.height * 0.4)),
                    (0, 255, 0),
                    9,
                )
            elif self.action in [Event.SWIPE_DOWN, Event.SWIPE_DOWN2, Event.SWIPE_DOWN3]:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.4)),
                    (self.width // 2, int(self.height * 0.6)),
                    (0, 255, 0),
                    9,
                )

            elif self.action == Event.FAST_SWIPE_DOWN:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.4)),
                    (self.width // 2, int(self.height * 0.6)),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.FAST_SWIPE_UP:
                frame = cv2.arrowedLine(
                    frame,
                    (self.width // 2, int(self.height * 0.6)),
                    (self.width // 2, int(self.height * 0.4)),
                    (0, 255, 0),
                    9,
                )
            elif self.action == Event.ZOOM_OUT:
                center_x, center_y = self.width // 2, self.height // 2
                square_size = 200

                top_left = (center_x - square_size // 2, center_y - square_size // 2)
                top_right = (center_x + square_size // 2, center_y - square_size // 2)
                bottom_left = (center_x - square_size // 2, center_y + square_size // 2)
                bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                frame = cv2.arrowedLine(frame, top_left, (center_x - 20, center_y - 20), (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, top_right, (center_x + 20, center_y - 20), (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, bottom_left, (center_x - 20, center_y + 20), (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, bottom_right, (center_x + 20, center_y + 20), (0, 255, 0), 3)
            elif self.action == Event.ZOOM_IN:
                center_x, center_y = self.width // 2, self.height // 2
                square_size = 200
                arrow_length = 50
                top_left = (center_x - square_size // 2, center_y - square_size // 2)
                top_right = (center_x + square_size // 2, center_y - square_size // 2)
                bottom_left = (center_x - square_size // 2, center_y + square_size // 2)
                bottom_right = (center_x + square_size // 2, center_y + square_size // 2)

                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                top_left_end = (top_left[0] - arrow_length, top_left[1] - arrow_length)
                top_right_end = (top_right[0] + arrow_length, top_right[1] - arrow_length)
                bottom_left_end = (bottom_left[0] - arrow_length, bottom_left[1] + arrow_length)
                bottom_right_end = (bottom_right[0] + arrow_length, bottom_right[1] + arrow_length)

                frame = cv2.arrowedLine(frame, top_left, top_left_end, (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, top_right, top_right_end, (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, bottom_left, bottom_left_end, (0, 255, 0), 3)
                frame = cv2.arrowedLine(frame, bottom_right, bottom_right_end, (0, 255, 0), 3)

            elif self.action in [Event.DRAG, Event.DRAG2, Event.DRAG3]:
                frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (0, 255, 0), 9)
            elif self.action == Event.DOUBLE_TAP:
                frame = cv2.putText(frame, 'DOUBLE CLICK', (self.width // 2, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 5, cv2.LINE_AA) 
            elif self.action == Event.TAP:
                frame = cv2.putText(frame, 'CLICK', (self.width // 2, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 5, cv2.LINE_AA) 
            elif self.action in [Event.DROP, Event.DROP2, Event.DROP3]:
                frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (0, 0, 255), -1)
            elif self.action == Event.COUNTERCLOCK:
                frame = cv2.putText(frame, 'COUNTERCLOCK', (self.width // 2, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 255, 0) , 5, cv2.LINE_AA) 
            elif self.action == Event.CLOCKWISE:
                frame = cv2.putText(frame, 'CLOCKWISE', (self.width // 2, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 255, 0) , 5, cv2.LINE_AA) 
            # elif self.action == Event.DRAG2:
            #     frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (255, 0, 0), 9)
            # elif self.action == Event.DROP2:
            #     frame = cv2.circle(frame, (self.width // 2, self.height // 2), 50, (0, 0, 255), -1)
            self.show_delay += 1
            if self.show_delay > 10:
                self.show_delay = 0
                self.action = None
                self.x = self.y = None

        return frame
