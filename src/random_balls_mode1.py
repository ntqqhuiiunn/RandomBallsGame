import cv2
import numpy
import random
import sys
from cvzone.HandTrackingModule import HandDetector
from src.buttons_manager import ButtonManager
from src.utils import loadConfig, showMouse


class RandomBallsMode1:
    def __init__(self, video, configPath, item, itemName) -> None:
        self.config = loadConfig(configPath)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video = video
        self.detector = HandDetector(
            maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.numBalls = self.config["mode1"]["numBalls"]
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.balls = numpy.empty([self.numBalls, 3], dtype=numpy.uint16)
        self.score = 0
        self.ballRadius = self.config["mode1"]["radius"]
        self.lowerBoundX, self.upperBoundX = self.config["global"][
            "lowerBoundX"], self.config["global"]["upperBoundX"]
        self.lowerBoundY, self.upperBoundY = self.config["global"][
            "lowerBoundY"], self.config["global"]["upperBoundY"]
        self.currentColor = random.randint(0, 3)
        colorsRange = list(range(0, 4))
        colorsRange.remove(self.currentColor)
        self.wrongColor = random.choice(colorsRange)
        for i in range(self.numBalls):
            x, y = self._generateBall()
            self.balls[i] = numpy.array([x, y, i % 4])
        self.balls = numpy.array(self.balls)
        self.introVideo = cv2.VideoCapture(r"images\countdown.mp4")
        self.limit = self.config["global"]["limit"]
        self._setUpItem(item)
        self.win = cv2.VideoCapture(r"images\win.mp4")
        self._setUpMinus(itemName)
        self.itemName = itemName
        self.plus = self._setUpImage(r"images\plus.png")
        self.buttonsManager = ButtonManager(configPath)
        self.fixedSize = self.config["global"]["fixedFrameSize"]
        self._setUpBackground()

    def _setUpImage(self, imagePath):
        obj = cv2.imread(imagePath)
        obj = cv2.resize(
            obj, (2 * int(self.ballRadius / 1.4142) + 1, 2 * int(self.ballRadius / 1.4142) + 1))
        obj = cv2.flip(obj, 1)
        return obj

    def _setUpBackground(self):
        temp = cv2.imread(r"images\back1.png")
        self.background = cv2.resize(temp, (self.width, self.height))

    def _setUpItem(self, item):
        obj = cv2.resize(
            item, (2 * int(self.ballRadius / 1.4142) + 1, 2 * int(self.ballRadius / 1.4142) + 1))
        obj = cv2.flip(obj, 1)
        self.item = obj

    def _setUpMinus(self, name):
        if name == "bomb":
            self.minus = cv2.imread(r"images\bomb_drop.png")
        elif name == "egg":
            self.minus = cv2.imread(r"images\egg_drop.png")
        else:
            self.minus = cv2.imread(r"images\minus.png")
        self.minus = cv2.resize(
            self.minus, (4 * self.ballRadius + 1, 4 * self.ballRadius + 1))
        self.minus = cv2.flip(self.minus, 1)

    def _up(self) -> None:
        for ball in self.balls:
            ball[1] += self.config["mode1"]["velocity"]

    def _generateBall(self):
        fromX, toX = self.lowerBoundX + 4 * self.ballRadius + \
            1, self.upperBoundX - 4 * self.ballRadius - 1
        fromY, toY = self.lowerBoundY, self.upperBoundY
        x = random.randint(fromX, toX)
        y = random.randint(fromY, toY)
        while self._isCollision(x, y, self.balls):
            x = random.randint(fromX, toX)
            y = random.randint(fromY, toY)
        return x, y

    def _dropBall(self, index: int) -> None:
        x, y = self._generateBall()
        self.balls[index] = [x, y, index % 4]

    def _isCollision(self, x, y, existingBalls):
        distances = numpy.sqrt(
            (existingBalls[:, 0] - x) ** 2 + (existingBalls[:, 1] - y) ** 2)
        return numpy.any(distances < self.ballRadius * 2 + 4)

    def _searchBall(self, x, y):
        rightBalls = []
        wrongItems = []
        for i, ball in enumerate(self.balls):
            if abs(x - ball[0]) <= self.ballRadius:
                if abs(y - ball[1]) <= self.ballRadius:
                    if ball[2] == self.currentColor:
                        rightBalls.append([ball[0], ball[1]])
                        self.score += 1
                    else:
                        wrongItems.append([ball[0], ball[1]])
                        self.score -= 1
                    self._dropBall(i)
                    self.currentColor = random.randint(0, 3)
                    colorsRange = list(range(0, 4))
                    colorsRange.remove(self.currentColor)
                    self.wrongColor = random.choice(colorsRange)
        return rightBalls,  wrongItems

    def _getNameColor(self, colorIndex: int):
        if colorIndex == 0:
            return "Blue"
        elif colorIndex == 1:
            return "Green"
        elif colorIndex == 2:
            return "Red"
        else:
            return "Yellow"

    def _displayItem(self, frame, coordinate, item, isBall=False, color=None):
        x, y = coordinate[0], coordinate[1]
        obj = item.copy()
        mask = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
        size = obj.shape[0]

        mask_selection = mask > 0
        frame_selection = frame[y - size // 2: y +
                                (size + 1) // 2, x - size // 2: x + (size + 1) // 2]
        frame_selection[mask_selection] = item[mask_selection]

        if isBall:
            assert color is not None, "Color must be not null"
            cv2.circle(frame, (x, y), self.ballRadius, color, 2)

    def _displayHand(self, landmarks, size):
        height, width = size[0], size[1]
        tempLandmarks = [(width - landmark[0], landmark[1])
                         for landmark in landmarks]
        return tempLandmarks

    def _displayUserScreen(self, frame, userScreen):
        ratio = userScreen.shape[0] / userScreen.shape[1]
        newWidth = self.lowerBoundX
        newHeight = int(ratio * newWidth)
        screen = cv2.resize(userScreen, (newWidth, newHeight))
        bottomRightX = newWidth
        bottomRightY = self.height - 1
        frame[bottomRightY - newHeight: bottomRightY,
              bottomRightX - newWidth: bottomRightX] = screen

    def run(self):
        while True:
            cursorX = -1
            cursorY = -1
            ret, frame = self.video.read()
            image = frame
            mask = self.background.copy()
            self.buttonsManager.setupQuit(mask, reversed=True)
            retIntro, frameIntro = self.introVideo.read()

            if self.score <= 10:
                if not ret:
                    break
                if ret and retIntro:
                    frameIntro = cv2.resize(
                        frameIntro, (frame.shape[1], frame.shape[0]))
                    mask = frameIntro
                if ret and not retIntro:
                    hand, image = self.detector.findHands(frame, draw=True)
                    if len(hand) > 0:
                        landmarks = hand[0]['lmList']
                        cursor = landmarks[8]
                        self.buttonsManager.moveToQuitButton(mask, cursor)
                        cursorX, cursorY = cursor[0], cursor[1]
                        showMouse(mask, cursorX, cursorY)
                    if cursorX >= 0 and cursorY >= 0:
                        rightBalls, wrongItems = self._searchBall(
                            cursorX, cursorY)
                        if len(rightBalls) > 0:
                            for ball in rightBalls:
                                self._displayItem(
                                    mask, ball, self.plus)
                        if len(wrongItems) > 0:
                            for stone in wrongItems:
                                self._displayItem(
                                    mask, stone, self.minus)
                    self._up()
                    for i in range(len(self.balls)):
                        if self.balls[i][1] >= self.limit:
                            self._dropBall(i)
                    for i in range(len(self.balls)):
                        c = self.colors[self.balls[i][2]]
                        self._displayItem(
                            mask, self.balls[i][:2], self.item, isBall=True, color=c)
                    mask = cv2.flip(mask, 1)
                    cv2.line(mask, (0, self.limit + self.ballRadius), (self.width -
                             self.lowerBoundX, self.limit + self.ballRadius), (0, 0, 255), thickness=2)
                    cv2.line(mask, (0, self.lowerBoundY - self.ballRadius), (self.width,
                             self.lowerBoundY - self.ballRadius), (0, 255, 255), thickness=2)
                    cv2.line(mask, (self.width - self.lowerBoundX, self.lowerBoundY - self.ballRadius),
                             (self.width - self.lowerBoundX, self.limit + self.ballRadius), (255, 0, 255), 2)
                    cv2.putText(mask, "Color: " + self._getNameColor(self.currentColor), (0, 25), fontScale=1.0,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=self.colors[self.wrongColor], thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(mask, "Score: " + str(self.score), (0, 50), fontScale=1.0,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=self.colors[self.wrongColor], thickness=2, lineType=cv2.LINE_AA)

            else:
                retWin, frameWin = self.win.read()
                if not retWin:
                    sys.exit()
                frameWin = cv2.resize(
                    frameWin, (frame.shape[1], frame.shape[0]))
                mask = frameWin
            mask = cv2.resize(mask, self.fixedSize)
            winname = "Pamdom Balls"
            cv2.namedWindow(winname, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(winname, 10, 10)
            cv2.imshow(winname, mask)
            userScreen = "User Cam"
            cv2.namedWindow(userScreen, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(userScreen, 1 + self.fixedSize[0], 10)
            image = cv2.flip(image, 1)
            cv2.imshow(userScreen, image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    app = RandomBallsMode1(cap, r"config\config.yaml")
    app.run()
