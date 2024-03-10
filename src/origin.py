import cv2
import numpy
import random
import sys
import os
from src.buttons_manager import ButtonManager
from PIL import ImageFont


class OriginalMode:
    def __init__(self, config, video, detector) -> None:
        self.fixedSize = (1100, 640)
        self.video = video
        self.WIDTH = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.config = config
        self.itemWidth = 80
        self.detector = detector
        self.lowerBoundX, self.upperBoundX = self.config["global"][
            "lowerBoundX"], self.config["global"]["upperBoundX"]
        self.lowerBoundY, self.upperBoundY = self.config["global"][
            "lowerBoundY"], self.config["global"]["upperBoundY"]
        self._setUpItems()
        self._setUpBackground()
        self._setUpSplash()
        self.buttons = ButtonManager(self.config)
        self.winname = "Pamdom Balls"
        self.wrongItems = []
        self._setUpPieces()
        self._setUpFont()
        self.side1 = cv2.imread(r"asset\side1.png")
        self.side2 = cv2.imread(r"asset\side2.png")
        self.side3 = cv2.imread(r"asset\side3.png")
        self.setUpGraphics()

    def _setUpFont(self):
        font_path = r"fonts\33713_SerpentineBoldItalic.ttf"
        font_size = 30
        self.font = ImageFont.truetype(font_path, font_size)

    def _setUpItems(self):
        self.fruits = ["apple", "banana", "lemon", "orange", "watermelon"]
        self.numItems = 7
        self.fruitsSource = []
        self.coordinates = numpy.empty([self.numItems, 4], dtype=numpy.uint16)
        self.source = r"asset\objects"
        self.maxHeight = 0
        for fruit in self.fruits:
            image = cv2.imread(os.path.join(self.source, fruit + ".png"))
            height, width = image.shape[0], image.shape[1]
            new_w = self.itemWidth
            new_h = int(new_w * height / width)
            obj = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)
            if new_h > self.maxHeight:
                self.maxHeight = new_h
            self.fruitsSource.append(obj)
        for i in range(self.numItems):
            item = self.fruitsSource[i % 5]
            w = item.shape[1]
            h = item.shape[0]
            x, y = self._generateItem((w, h))
            self.coordinates[i] = [x, y, w, h]

    def _setUpBackground(self):
        self.background = cv2.imread(
            r"asset\back.jpg")

        self.background = cv2.resize(self.background, self.fixedSize)

    def _setUpSplash(self):
        self.splashs = []
        for path in os.listdir(r"asset\splash"):
            image = cv2.imread(os.path.join(r"asset\splash", path))
            image = cv2.resize(
                image, (int(1.5 * self.itemWidth), int(1.5 * self.itemWidth)))
            image = cv2.flip(image, 1)
            self.splashs.append(image)

    def _setUpSide(self, frame, level=1):
        if level == 1:
            frame[:, :400] = self.side1.copy()
        elif level == 2:
            frame[:, :400] = self.side2.copy()
        else:
            frame[:, :400] = self.side3.copy()
        return frame

    def _setUpPieces(self):
        self.pieces = []
        self.piecesPosition = []
        for fruit in self.fruits:
            path1 = os.path.join(self.source, fruit + "1.png")
            path2 = os.path.join(self.source, fruit + "2.png")
            piece1 = cv2.imread(path1)
            piece1 = cv2.resize(
                piece1, (76, int(76 / piece1.shape[1] * piece1.shape[0])))
            piece2 = cv2.imread(path2)
            piece2 = cv2.resize(
                piece2, (76, int(76 / piece2.shape[1] * piece2.shape[0])))
            self.pieces.append((piece1, piece2))
        for _ in range(self.numItems):
            self.piecesPosition.append([])

    def _generateItem(self, size):
        fromX, toX = self.lowerBoundX, self.upperBoundX - 2 * self.itemWidth
        fromY, toY = self.lowerBoundY, (self.upperBoundY - self.maxHeight) // 2
        x = random.randint(fromX, toX)
        y = random.randint(fromY, toY)
        while self._checkOverlap((x, y), size, self.coordinates):
            x = random.randint(fromX, toX)
            y = random.randint(fromY, toY)
        return x, y

    def _checkOverlap(self, point, size, rectangles):
        x = point[0]
        y = point[1]
        rect_width = size[0]
        rect_height = size[1]
        overlap = any(x < rect[0] + rect_width and
                      x + rect_width > rect[0] and
                      y < rect[1] + rect_height and
                      y + rect_height > rect[1]
                      for rect in rectangles)
        return overlap

    def displayItem(self, index, frame):
        item = self.fruitsSource[index % 5].copy()
        coordinate = self.coordinates[index]
        gray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
        selection = frame[coordinate[1]: coordinate[1] + coordinate[3],
                          coordinate[0]: coordinate[0] + coordinate[2]]
        selection[gray > 0] = item[gray > 0]

    def findIntersectedRect(self, cursor):
        x = cursor[0]
        y = cursor[1]
        # Extract coordinates of top-left and bottom-right corners of all rectangles
        rect_x = self.coordinates[:, 0]
        rect_y = self.coordinates[:, 1]
        rect_width = self.coordinates[:, 2]
        rect_height = self.coordinates[:, 3]

        # Check if the point is inside any rectangle using broadcasting
        inside_x = numpy.logical_and(x >= rect_x, x <= rect_x + rect_width)
        inside_y = numpy.logical_and(y >= rect_y, y <= rect_y + rect_height)
        inside_any_rectangle = numpy.logical_and(inside_x, inside_y)

        # Find the index of the first rectangle containing the point
        first_index = numpy.argmax(inside_any_rectangle)

        if inside_any_rectangle[first_index]:
            return first_index
        else:
            return -1

    def dropItem(self, index):
        coordinate = self.coordinates[index]
        x, y, w, h = coordinate
        self.wrongItems[index] = [x, y]
        self.piecesPosition[index] = [x, y, x + 51, y + 51]
        x, y = self._generateItem((w, h))
        self.coordinates[index] = [x, y, w, h]

    def showUser(self, image, frame):
        x0 = 65 + 700
        y0 = 387  # 300
        image = cv2.resize(
            image, (260, 195), interpolation=cv2.INTER_NEAREST)
        image = cv2.flip(image, 1)
        frame[y0: 195 + y0, x0: x0 + 260] = image

    def setUpTarget(self, frame, indices, targets):
        targetFruits = []
        names = []
        for index in indices:
            targetFruits.append(self.fruitsSource[index])
            names.append(self.fruits[index])
        displayedItems = []
        grayItems = []
        for i in range(len(indices)):
            item = targetFruits[i]
            displayedItem = cv2.resize(
                item, (50, int(50 * item.shape[0] / item.shape[1])), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(displayedItem, cv2.COLOR_BGR2GRAY)
            displayedItems.append(displayedItem)
            grayItems.append(gray)

        y0 = 125
        for i in range(3):
            x0 = frame.shape[1] - 1 - (105 + 94 * i)
            item = displayedItems[i]
            gray = grayItems[i]
            h, w = item.shape[0], item.shape[1]
            frame[y0 - h // 2: y0 + (h + 1) // 2, x0 - w //
                  2: x0 + (w + 1) // 2][gray > 0] = item[gray > 0]
            cv2.putText(frame, str(targets[i]),
                        (x0 - w // 2 + 5, y0 + 120), 4, 1.5, (0, 255, 0), 2)

    def setUpGraphics(self):
        self.loadingSource1 = cv2.VideoCapture(r"images\loading.mp4")
        self.loadingSource2 = cv2.VideoCapture(r"images\loading.mp4")
        self.loadingSource3 = cv2.VideoCapture(r"images\loading.mp4")
        self.loadingSource0 = r"images\loading.mp4"
        self.nextLevelSource1 = cv2.VideoCapture(r"images\next_level.mp4")
        self.nextLevelSource2 = cv2.VideoCapture(r"images\next_level.mp4")
        self.nextLevelSource3 = cv2.VideoCapture(r"images\next_level.mp4")
        self.nextLevelSource0 = r"images\next_level.mp4"
        self.loseSource1 = cv2.VideoCapture(r"images\lose.mp4")
        self.loseSource2 = cv2.VideoCapture(r"images\lose.mp4")
        self.loseSource3 = cv2.VideoCapture(r"images\lose.mp4")
        self.loseSource0 = r"images\lose.mp4"
        self.mouseSource = cv2.imread(r"images\mouse.png")
        self.bombSource = cv2.imread(r"images\bombey.png")
        self.bombExplosionSource = cv2.imread(r"images\bomb_drop.png")
        self.winSource = cv2.VideoCapture(r"images\win.mp4")
        self.winSource0 = r"images\win.mp4"
