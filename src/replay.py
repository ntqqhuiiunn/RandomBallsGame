import cv2
import numpy


class ReplayProgram:
    def __init__(self) -> None:
        self.replay = cv2.imread(r"images\replay.png")
        self.temporaryReplay = cv2.imread(r"images\replay_touched.png")
        self.temporaryReplay = cv2.resize(
            self.temporaryReplay, (self.replay.shape[1], self.replay.shape[0]))

    def _setUpBackground(self, size):
        temp = numpy.full((size[1], size[0], 3), fill_value=(
            23, 127, 222), dtype=numpy.uint8)
        x, y = size[0] // 2, size[1] // 2
        obj = self.replay.copy()
        temp[y - obj.shape[0] // 2: y + (obj.shape[0] + 1) // 2,
             x - obj.shape[1] // 2: x + (obj.shape[1] + 1) // 2] = obj
        return temp

    def _moveOver(self, frame, cursor, clicked):
        selected = False
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        obj = self.replay.copy()
        lowerBoundX = x - obj.shape[1] // 2
        upperBoundX = x + (obj.shape[1] + 1) // 2
        lowerBoundY = y - obj.shape[0] // 2
        upperBoundY = y + (obj.shape[0] + 1) // 2
        if upperBoundX > cursor[0] > lowerBoundX and upperBoundY > cursor[1] > lowerBoundY:
            frame[lowerBoundY: upperBoundY,
                  lowerBoundX: upperBoundX] = self.temporaryReplay
            selected = True
            if selected and clicked:
                return True
        return False
