import os
import cv2
import numpy
import sys
from src.utils import loadConfig


class ButtonManager:
    def __init__(self, configPath) -> None:
        self.config = loadConfig(configPath)
        self.buttonWidth, self.buttonHeight = self.config["global"]["buttonSize"]
        self._setup()
        self._setUpItem()
        self.itemPosition = []

    def _setup(self):
        self.start = cv2.imread(r"images\original_start.png")
        self.temporaryStart = cv2.imread(r"images\after_start.png")
        self.title = cv2.imread(r"images\title.png")
        self.mode1 = cv2.imread(r"images\mode1.png")
        self.mode2 = cv2.imread(r"images\mode2.png")
        self.temporaryMode1 = cv2.imread(r"images\mode1touched.png")
        self.temporaryMode2 = cv2.imread(r"images\mode2touched.png")
        modeSize = self.config["global"]["modeSize"]
        self.mode1 = cv2.resize(self.mode1, modeSize)
        self.mode2 = cv2.resize(self.mode2, modeSize)
        self.temporaryMode1 = cv2.resize(self.temporaryMode1, modeSize)
        self.temporaryMode2 = cv2.resize(self.temporaryMode2, modeSize)
        self.quit = cv2.imread(r"images\exit.png")
        self.start = cv2.resize(
            self.start, (self.buttonWidth, int(self.buttonWidth / 2.2039)))
        self.temporaryStart = cv2.resize(
            self.temporaryStart, (self.buttonWidth, int(self.buttonWidth / 2.2039)))
        self.title = cv2.resize(self.title, (401, int(401 / 2.96618)))
        self.quit = cv2.resize(self.quit, self.config["global"]["quitSize"])
        self.selectitem = cv2.imread(r"images\select_item.png")

    def _setUpItem(self):
        self.items = []
        self.items.append(cv2.imread(r"images\ball.png"))
        self.items.append(cv2.imread(r"images\bomb.png"))
        self.items.append(cv2.imread(r"images\egg.png"))
        self.nameItems = ["ball", "bomb", "egg"]

    def displayStartButton(self, frame):
        w, h = self.start.shape[1], self.start.shape[0]
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        frame[y - h // 2: y + h // 2 + 1, x -
              w // 2: x + w // 2 + 1] = self.start

    def moveOverStartButton(self, frame, cursor, clicked):
        selected = True
        w, h = self.start.shape[1], self.start.shape[0]
        x, y = frame.shape[1] // 2, frame.shape[0] // 2
        lowerBoundX, upperBoundX = x - w // 2, x + w // 2
        lowerBoundY, upperBoundY = y - h // 2, y + h // 2
        if upperBoundX > cursor[0] > lowerBoundX and upperBoundY > cursor[1] > lowerBoundY:
            frame[lowerBoundY: upperBoundY + 1,
                  lowerBoundX: upperBoundX + 1] = self.temporaryStart
            selected = True
        if selected and clicked:
            return True
        return False

    def moveToQuitButton(self, frame, cursor):
        x, y = frame.shape[1] - 5 - \
            self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        w, h = self.quit.shape[1], self.quit.shape[0]
        lowerBoundX, upperBoundX = x - w // 2, x + w // 2 + 1
        lowerBoundY, upperBoundY = y - h // 2, y + h // 2 + 1
        if upperBoundX > frame.shape[1] - cursor[0] > lowerBoundX and upperBoundY > cursor[1] > lowerBoundY:
            sys.exit()

    def setupTitle(self, frame):
        x, y = frame.shape[1] // 2, frame.shape[0] // 3 - 80
        w, h = self.title.shape[1], self.title.shape[0]
        frame[y - h // 2: y + h // 2 + 1, x -
              w // 2: x + w // 2 + 1] = self.title

    def setupQuit(self, frame, reversed=False):
        if not reversed:
            x, y = frame.shape[1] - 5 - \
                self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        else:
            x, y = 5 + self.quit.shape[1] // 2, self.quit.shape[0] // 2 + 5
        w, h = self.quit.shape[1], self.quit.shape[0]
        frame[y - h // 2: y + h // 2 + 1, x -
              w // 2: x + w // 2 + 1] = self.quit

    def displaySelectModeButton(self, frame):
        x0, y0 = frame.shape[1] // 2, frame.shape[0] // 2
        y0 += 30
        w1, h1, w2, h2 = self.mode1.shape[1], self.mode1.shape[0], self.mode2.shape[1], self.mode2.shape[0]
        frame[y0 - h1 // 2: y0 + h1 // 2 + 1,
              x0 - w1 - 75: x0 - 75] = self.mode1

        frame[y0 - h2 // 2: y0 + h2 // 2 + 1,
              x0 + 75: x0 + w2 + 75] = self.mode2

    def selectMode1(self, frame, cursor, clicked):
        selected = False
        x = frame.shape[1] - 1 - cursor[0]
        y = cursor[1]
        x0, y0 = frame.shape[1] // 2, frame.shape[0] // 2
        y0 += 30
        w1, h1 = self.mode1.shape[1], self.mode1.shape[0]
        lowerBoundY = y0 - h1 // 2
        upperBoundY = y0 + h1 // 2 + 1
        lowerBoundX = x0 - w1 - 75
        upperBoundX = x0 - 75
        if upperBoundX > x >= lowerBoundX and upperBoundY > y >= lowerBoundY:
            frame[lowerBoundY: upperBoundY, lowerBoundX:
                  upperBoundX] = self.temporaryMode1
            selected = True
        if selected and clicked:
            return True
        return False

    def selectMode2(self, frame, cursor, clicked):
        selected = False
        x = frame.shape[1] - 1 - cursor[0]
        y = cursor[1]
        x0, y0 = frame.shape[1] // 2, frame.shape[0] // 2
        y0 += 30
        w2, h2 = self.mode2.shape[1], self.mode2.shape[0]
        lowerBoundY = y0 - h2 // 2
        upperBoundY = y0 + h2 // 2 + 1
        lowerBoundX = x0 + 75
        upperBoundX = x0 + w2 + 75
        if upperBoundX > x >= lowerBoundX and upperBoundY > y >= lowerBoundY:
            frame[lowerBoundY: upperBoundY,
                  lowerBoundX: upperBoundX] = self.temporaryMode2
            selected = True
        if selected and clicked:
            return True
        return False

    def displaySelectItem(self, frame):
        x, y = frame.shape[1] // 2, frame.shape[0] // 3 - 80
        w, h = self.selectitem.shape[1], self.selectitem.shape[0]
        frame[y - h // 2: y + (h + 1) // 2, x -
              w // 2: x + (w + 1) // 2] = self.selectitem
        itemRange = 150
        y0 = frame.shape[0] // 2
        for i in range(len(self.items)):
            h0, w0 = 100, int(
                self.items[i].shape[1] / self.items[0].shape[0] * 100)
            obj = cv2.resize(self.items[i], (w0, h0))
            lowerY = y0 - h0 // 2
            upperY = y0 + (h0 + 1) // 2
            lowerX = (i + 1) * itemRange + 30 * i - w0 // 2
            upperX = (i + 1) * itemRange + 30 * i + (w0 + 1) // 2
            gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            selection = frame[lowerY: upperY, lowerX: upperX]
            selection[gray > 0] = obj[gray > 0]
            self.itemPosition.append([lowerY, upperY, lowerX, upperX])

    def getItem(self, hand, frame, clicked):
        selected = False
        w = frame.shape[1]
        handX, handY = hand[0], hand[1]
        handX = w - 1 - handX
        for i, position in enumerate(self.itemPosition):
            if position[0] < handY < position[1] and position[2] < handX < position[3]:
                cv2.rectangle(frame, (position[2], position[0]), (position[3], position[1]), color=(
                    0, 0, 255), thickness=10)
                selected = True
                if selected and clicked:
                    i = len(self.items) - 1 - i
                    return self.items[i], self.nameItems[i]
        return None, None


