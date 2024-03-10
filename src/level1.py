import cv2
import sys
import random
import time
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw
from src.origin import OriginalMode
from src.level2 import Level2
from src.replay import ReplayProgram
from src.utils import *



class Level1(OriginalMode):
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

        self.nextGame = Level2(self.config, self.video, self.detector)
        self.replaymode = ReplayProgram()

    def addText(self, frame, text, position):
        otherFrame = Image.fromarray(frame)
        draw = ImageDraw.Draw(otherFrame)
        draw.text(position, text, fill=(0, 255, 0), font=self.font)
        return numpy.array(otherFrame)

    def up(self):
        for i in range(self.numItems):
            if self.coordinates[i][1] + self.coordinates[i][3] >= self.upperBoundY - self.maxHeight:
                w = self.coordinates[i][2]
                h = self.coordinates[i][3]
                x, y = self._generateItem((w, h))
                self.coordinates[i] = [x, y, w, h]
            else:
                self.coordinates[i][1] += 2

    def run(self):
        remainingTime = 1
        setTime = True
        pause = False
        showCamera = True
        startVideo = cv2.VideoCapture(self.loadingSource0)
        endVideo = cv2.VideoCapture(self.nextLevelSource0)
        loseVideo = cv2.VideoCapture(self.loseSource0)
        targetIndices = random.sample([0, 1, 2, 3, 4], k=3)
        upperBoundItems = self.config["numberOfItems"]["level1"]
        targetScore = [random.randint(
            5, upperBoundItems), random.randint(5, upperBoundItems), random.randint(5, upperBoundItems)]
        while True:
            cursor = None
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
                            self.config["timeLimit"]["level1"]
                        setTime = False
                    else:
                        if not pause:
                            remainingTime = timeUp - time.time()
                    showCamera = True
                    mask = self.background.copy()
                    mask = self._setUpSide(mask, level=1)
                    frame = cv2.resize(frame, self.fixedSize)
                    hand, image = self.detector.findHands(frame, draw=True)
                    if not pause:
                        self.up()
                    for i in range(self.numItems):
                        self.displayItem(i, mask)
                    if len(hand) > 0:
                        if hand[0]["type"] == "Right":
                            landmarks = hand[0]["lmList"]
                            landmarks = numpy.array(landmarks)
                            cursor = landmarks[8]
                            self.buttons.moveToQuitButton(mask, cursor, reversed=True)
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
                                self.piecesPosition[i][1] += 7
                                self.piecesPosition[i][3] += 7
                    cv2.line(mask, (400, self.upperBoundY - self.maxHeight),
                             (self.upperBoundX - 1, self.upperBoundY - self.maxHeight), (0, 255, 0), 2)
                    mask = cv2.flip(mask, 1)
                    mask = self.addText(
                        mask, str(round(remainingTime, 2)), (945, 15))
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
                        self.buttons.moveToQuitButton(mask, cursor, reversed=False)
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
    game = Level1(config, video, detector)
    game.run()
