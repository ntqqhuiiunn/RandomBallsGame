import cv2
import sys


class ButtonManager:
    def __init__(self, config) -> None:
        self.config = config
        self.buttonWidth, self.buttonHeight = self.config["global"]["buttonSize"]
        self._setup()
        self.itemPosition = []

    def _setup(self):
        self.start = cv2.imread(r"images\original_start.png")
        self.temporaryStart = cv2.imread(r"images\after_start.png")
        self.quit = cv2.imread(r"images\exit.png")
        self.temporaryStart = cv2.resize(
            self.temporaryStart, (self.start.shape[1], self.start.shape[0]))
        self.title = cv2.imread(r"images\title.png")
        self.quit = cv2.resize(self.quit, self.config["global"]["quitSize"])


    def displayStartButton(self, frame):
        w, h = self.start.shape[1], self.start.shape[0]
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        frame[y - h // 2: y + (h + 1) // 2, x -
              w // 2: x + (w + 1) // 2] = self.start

    def moveOverStartButton(self, frame, cursor, clicked):
        selected = False
        w, h = self.start.shape[1], self.start.shape[0]
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        lowerBoundX, upperBoundX = x - w // 2, x + (w + 1) // 2
        lowerBoundY, upperBoundY = y - h // 2, y + (h + 1) // 2
        if upperBoundX > cursor[0] > lowerBoundX and upperBoundY > cursor[1] > lowerBoundY:
            frame[lowerBoundY: upperBoundY,
                  lowerBoundX: upperBoundX] = self.temporaryStart
            selected = True
        if selected and clicked:
            return True
        return False

    def moveToQuitButton(self, frame, cursor, reversed=False, *args):
        if not reversed:
            x, y = frame.shape[1] - 5 - \
                self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        else:
            x, y = 5 + self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        w, h = self.quit.shape[1], self.quit.shape[0]
        lowerBoundX, upperBoundX = x - w // 2, x + w // 2 + 1
        lowerBoundY, upperBoundY = y - h // 2, y + h // 2 + 1
        check = True
        for arg in args:
            if arg:
                check = False
        if upperBoundX > frame.shape[1] - cursor[0] > lowerBoundX and upperBoundY > cursor[1] > lowerBoundY and check:
            sys.exit()

    def setupTitle(self, frame):
        x, y = frame.shape[1] // 2, frame.shape[0] // 3 - 80
        w, h = self.title.shape[1], self.title.shape[0]
        frame[y - h // 2: y + (h + 1) // 2, x -
              w // 2: x + (w + 1) // 2] = self.title

    def setupQuit(self, frame, reversed=False):
        obj = self.quit.copy()
        if not reversed:
            x, y = frame.shape[1] - 5 - \
                self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        else:
            x, y = 5 + self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        w, h = self.quit.shape[1], self.quit.shape[0]
        selection = frame[y - h // 2: y + h // 2 + 1, x -
                          w // 2: x + w // 2 + 1]
        gray = cv2.cvtColor(self.quit, cv2.COLOR_BGR2GRAY)
        selection[gray > 0] = obj[gray > 0]

