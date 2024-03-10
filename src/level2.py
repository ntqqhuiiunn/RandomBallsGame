import cv2
import sys
import random
import time
from PIL import Image, ImageDraw
from cvzone.HandTrackingModule import HandDetector
from src.origin import OriginalMode
from src.utils import *
from src.level3 import Level3
from src.replay import ReplayProgram



class Level2(OriginalMode):
    def __init__(self, config, video, detector) -> None:
        super().__init__(config, video, detector)
        self.config = config
        self.mouse = self.mouseSource.copy()
        self.mouse = cv2.resize(self.mouse, (41, 41),
                                interpolation=cv2.INTER_NEAREST)
        self.delays = numpy.zeros(self.numItems, dtype=numpy.uint8)
        self.wrongItems = []
        for _ in range(self.numItems):
            self.wrongItems.append([])

        self.setUpBomb()
        self.nextGame = Level3(self.config, self.video, self.detector)
        self.replaymode = ReplayProgram()

    def addText(self, frame, text, position, touchedBomb):
        otherFrame = Image.fromarray(frame)
        draw = ImageDraw.Draw(otherFrame)
        color = (0, 0, 255) if touchedBomb else (0, 255, 0)
        draw.text(position, text, fill=color, font=self.font)
        return numpy.array(otherFrame)

    def setUpBomb(self):
        self.bomb = self.bombSource.copy()
        h, w = self.bomb.shape[0], self.bomb.shape[1]
        self.bomb = cv2.resize(
            self.bomb, (self.itemWidth, int(self.itemWidth / w * h)), interpolation=cv2.INTER_NEAREST)
        self.bombHeight, self.bombWidth = self.bomb.shape[0], self.bomb.shape[1]
        self.bombs = []
        for _ in range(self.numItems // 2):
            x, y = self._generateItem((self.bombWidth, self.bombHeight))
            self.bombs.append([x, y])
        self.bombs = numpy.array(self.bombs)
        self.bombDropIndices = []
        for _ in range(len(self.bombs)):
            self.bombDropIndices.append([])
        self.delayBombs = numpy.zeros(len(self.bombs), dtype=numpy.uint8)
        self.bum = self.bombExplosionSource.copy()
        self.bum = cv2.resize(
            self.bum, (3 * self.itemWidth, 3 * self.itemWidth), interpolation=cv2.INTER_NEAREST)

    def generateBomb(self, size):
        fromX, toX = self.lowerBoundX + self.itemWidth, self.upperBoundX - 2 * self.itemWidth
        fromY, toY = self.lowerBoundY + \
            self.itemWidth, (self.upperBoundY -
                             self.maxHeight) // 2 - 2 * self.itemWidth
        x = random.randint(fromX, toX)
        y = random.randint(fromY, toY)
        while self._checkOverlap((x, y), size, self.bombs):
            x = random.randint(fromX, toX)
            y = random.randint(fromY, toY)
        return x, y

    def displayBomb(self, frame):
        gray = cv2.cvtColor(self.bomb, cv2.COLOR_BGR2GRAY)
        for position in self.bombs:
            x, y = position
            selection = frame[y: y + self.bombHeight, x: x + self.bombWidth]
            selection[gray < 200] = self.bomb[gray < 200]

    def findTouchedBomb(self, cursor):
        x = cursor[0]
        y = cursor[1]
        # Extract coordinates of top-left and bottom-right corners of all rectangles
        rect_x = self.bombs[:, 0]
        rect_y = self.bombs[:, 1]
        rect_width = self.bombWidth
        rect_height = self.bombHeight

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

    def dropBomb(self, index):
        x, y = self.bombs[index]
        self.bombDropIndices[index] = [x, y]

        x, y = self.generateBomb((self.bombWidth, self.bombHeight))
        self.bombs[index] = [x, y]

    def up(self):
        for i in range(self.numItems):
            if self.coordinates[i][1] + self.coordinates[i][3] >= self.upperBoundY - self.maxHeight:
                w = self.coordinates[i][2]
                h = self.coordinates[i][3]
                x, y = self._generateItem((w, h))
                self.coordinates[i] = [x, y, w, h]
            else:
                self.coordinates[i][1] += 4
        for i in range(len(self.bombs)):
            if self.bombs[i][1] + 5 >= self.upperBoundY - self.maxHeight - self.itemWidth:
                x, y = self.generateBomb((self.bombWidth, self.bombHeight))
                self.bombs[i] = [x, y]
            else:
                self.bombs[i][1] += 5

    def run(self):
        remainingTime = 1
        setTime = True
        pause = False
        showCamera = True
        startVideo = cv2.VideoCapture(self.loadingSource0)
        endVideo = cv2.VideoCapture(self.nextLevelSource0)
        loseVideo = cv2.VideoCapture(self.loseSource0)
        targetIndices = random.sample([0, 1, 2, 3, 4], k=3)
        upperBoundItems = self.config["numberOfItems"]["level2"]
        targetScore = [random.randint(
            5, upperBoundItems), random.randint(5, upperBoundItems), random.randint(5, upperBoundItems)]
        while True:
            touchedBomb = False
            brokenBomb = 0
            ret, frame = self.video.read()
            retStart, frameStart = startVideo.read()
            check = all(value == 0 for value in targetScore)
            checkLose = remainingTime <= 0.001
            if not check and not checkLose:
                if retStart:
                    mask = cv2.resize(frameStart, self.fixedSize)
                    image = frame.copy()
                    showCamera = False
                elif not retStart and ret:
                    if setTime:
                        timeUp = time.time() + \
                            self.config["timeLimit"]["level2"]
                        setTime = False
                    else:
                        if not pause:
                            remainingTime = timeUp - time.time()
                    showCamera = True
                    mask = self.background.copy()
                    mask = self._setUpSide(mask, level=2)
                    frame = cv2.resize(frame, self.fixedSize)
                    hand, image = self.detector.findHands(frame, draw=True)
                    if not pause:
                        self.up()
                    for i in range(self.numItems):
                        self.displayItem(i, mask)
                    self.displayBomb(mask)
                    if len(hand) > 0:
                        landmarks = hand[0]["lmList"]
                        landmarks = numpy.array(landmarks)
                        cursor = landmarks[8]
                        self.buttons.moveToQuitButton(
                            mask, cursor, reversed=True)
                        showMouse(mask, cursor, self.mouse)
                        index = self.findIntersectedRect(cursor)
                        if index >= 0:
                            self.dropItem(index)
                            categoryIndex = index % 5
                            if categoryIndex in targetIndices:
                                tempId = targetIndices.index(
                                    categoryIndex)
                                if targetScore[tempId] > 0:
                                    targetScore[tempId] -= 1
                                else:
                                    targetScore[tempId] = 0
                        bombIndex = self.findTouchedBomb(cursor)
                        if bombIndex >= 0:
                            value = self.bombs[bombIndex]
                            cv2.putText(mask, str(value), value,
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            self.dropBomb(bombIndex)
                    for i, bombCoor in enumerate(self.bombDropIndices):
                        if len(bombCoor) > 0:
                            x, y = bombCoor
                            if self.delayBombs[i] < 5:
                                selection = mask[y - self.itemWidth: y + 2 * self.itemWidth,
                                                 x - self.itemWidth: x + 2 * self.itemWidth]
                                gray = cv2.cvtColor(
                                    self.bum, cv2.COLOR_BGR2GRAY)
                                selection[gray > 0] = self.bum[gray > 0]
                                self.delayBombs[i] += 1
                                touchedBomb = True
                            else:
                                brokenBomb += 1
                                self.delayBombs[i] = 0
                                self.bombDropIndices[i] = []
                    for i, wrong in enumerate(self.wrongItems):
                        if len(wrong) > 0:
                            x, y = wrong
                            if self.delays[i] < 5:
                                splash = self.splashs[i % 5]
                                selection = mask[y: y + splash.shape[0],
                                                 x: x + splash.shape[1]]
                                gray = cv2.cvtColor(splash, cv2.COLOR_BGR2GRAY)
                                selection[gray > 0] = splash[gray > 0]
                                self.delays[i] += 1
                            else:
                                self.wrongItems[i] = []
                                self.delays[i] = 0
                    for i, position in enumerate(self.piecesPosition):
                        if len(position) > 0:
                            piece1, piece2 = self.pieces[i % 5]
                            height = max(
                                position[1] + piece1.shape[0], position[3] + piece2.shape[0])
                            if height >= self.upperBoundY - self.maxHeight:
                                self.piecesPosition[i] = []
                            else:
                                x1, y1, x2, y2 = self.piecesPosition[i]
                                gray1 = cv2.cvtColor(
                                    piece1, cv2.COLOR_BGR2GRAY)
                                gray2 = cv2.cvtColor(
                                    piece2, cv2.COLOR_BGR2GRAY)
                                selection1 = mask[y1: y1 +
                                                  gray1.shape[0], x1: x1 + gray1.shape[1]]
                                selection1[gray1 > 0] = piece1[gray1 > 0]
                                selection2 = mask[y2: y2 +
                                                  gray2.shape[0], x2: x2 + gray2.shape[1]]
                                selection2[gray2 > 0] = piece2[gray2 > 0]
                                self.piecesPosition[i][1] += 9
                                self.piecesPosition[i][3] += 9
                    cv2.line(mask, (400, self.upperBoundY - self.maxHeight),
                             (self.upperBoundX - 1, self.upperBoundY - self.maxHeight), (0, 255, 0), 2)
                    mask = cv2.flip(mask, 1)
                    timeUp -= float(brokenBomb)
                    mask = self.addText(
                        mask, str(round(remainingTime, 2)), (945, 15), touchedBomb)
                    self.buttons.setupQuit(mask, reversed=True)
                    self.setUpTarget(mask, targetIndices,
                                     targetScore)
            elif not checkLose and check:
                retEnd, frameEnd = endVideo.read()
                if retEnd:
                    mask = cv2.resize(frameEnd, self.fixedSize)
                    image = frame.copy()
                    showCamera = False
                else:
                    return self.nextGame.run()
            elif checkLose:
                retLose, frameLose = loseVideo.read()
                if retLose:
                    mask = cv2.resize(frameLose, self.fixedSize)
                    image = frame.copy()
                    showCamera = False
                else:
                    mask = self.replaymode._setUpBackground(self.fixedSize)
                    self.buttons.setupQuit(mask)
                    showCamera = False
                    hand, image = self.detector.findHands(frame, draw=True)
                    if len(hand) > 0:
                        landmarks = hand[0]['lmList']
                        cursor = landmarks[8]

                        first = (landmarks[4][1] - landmarks[5][1]) ** 2 + \
                            (landmarks[4][0] - landmarks[5][0]) ** 2
                        second = (landmarks[0][1] - landmarks[5][1]) ** 2 + \
                            (landmarks[0][0] - landmarks[5][0]) ** 2 + 0.001
                        ratio = first / second
                        clicked = ratio < 0.07
                        touched = self.replaymode._moveOver(
                            mask, cursor, clicked)
                        showMouse(mask, cursor, self.mouse, mode="main")
                        self.buttons.moveToQuitButton(
                            mask, cursor, reversed=False)
                        if touched:
                            return True
            cv2.namedWindow(self.winname, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(self.winname, 50, 10)
            if showCamera:
                self.showUser(image, mask)
            cv2.imshow(self.winname, mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()


if __name__ == "__main__":
    config = loadConfig(r"config\config.yaml")
    video = cv2.VideoCapture(0)
    detector = HandDetector(
        maxHands=1, detectionCon=0.8, minTrackCon=0.7)
    game = Level2(config, video, detector)
    game.run()
