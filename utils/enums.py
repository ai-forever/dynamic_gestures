from enum import Enum


# Hand position enum.
class HandPosition(Enum):
    UNKNOWN = -1
    LEFT_START = 1
    RIGHT_START = 2
    LEFT_END = 3
    RIGHT_END = 4
    UP_START = 5
    UP_END = 6
    DOWN_START = 7
    DOWN_END = 8


# Events for action controller
class Event(Enum):
    UNKNOWN = -1
    SWIPE_RIGHT = 0
    SWIPE_LEFT = 1
    SWIPE_UP = 2
    SWIPE_DOWN = 3
    DRAG = 4
    DROP = 5


# Target classes
targets = [
    "down",  # 0
    "right",  # 1
    "left",  # 2
    "call",  # 3
    "dislike",  # 4
    "fist",  # 5
    "four",  # 6
    "like",  # 7
    "mute",  # 8
    "ok",  # 9
    "one",  # 10
    "palm",  # 11
    "peace",  # 12
    "rock",  # 13
    "stop",  # 14
    "stop_inverted",  # 15
    "three",  # 16
    "two_up",  # 17
    "two_up_inverted",  # 18
    "three2",  # 19
    "peace_inverted",  # 20
]
