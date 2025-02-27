class Hand:
    def __init__(self, bbox, hand_id=None, gesture=None):
        """
        Hand class

        Parameters
        ----------
        bbox : np.ndarray
            Bounding box of hand

        hand_id : int
            Id of hand

        gesture : int
            Current gesture of hand
        """
        self.bbox = bbox
        self.hand_id = hand_id
        if self.bbox is not None:
            self.center = self._get_center()
            self.size = self.bbox[2] - self.bbox[0]
        self.position = None
        self.gesture = gesture

    def _get_center(self):
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    def __repr__(self):
        return f"Hand({self.center}, {self.size}, {self.position}, {self.gesture})"
