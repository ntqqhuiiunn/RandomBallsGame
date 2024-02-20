import cv2
import numpy
import random
import time
import sys
from cvzone.HandTrackingModule import HandDetector
from buttons_manager import ButtonManager
from utils import loadConfig, drawSkeleton, setUpBackground


class RandomBallsMode2:
    def __init__(self, video, configPath) -> None:
        self.config = loadConfig(configPath)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video = video
        self.detector = HandDetector(
            maxHands=1, detectionCon=0.8, minTrackCon=0.7)

        self.ballRadius = self.config["mode2"]["radius"]
        self.lowerBoundX, self.upperBoundX = self.config["global"][
            "lowerBoundX"], self.config["global"]["upperBoundX"]
        self.lowerBoundY, self.upperBoundY = self.config["global"][
            "lowerBoundY"], self.config["global"]["upperBoundY"]
        self._initialBalls(self.config["mode2"]["numBalls"])
        self.introVideo = cv2.VideoCapture(r"images\countdown.mp4")
        self.limit = self.config["global"]["limit"]
        self.item = self._setUpImage(r"images\item.png")
        self.stone = self._setUpImage(r"images\stone.png")
        self.win = cv2.VideoCapture(r"images\win.mp4")
        self.nextLevel = cv2.VideoCapture(r"images\next_level.mp4")
        self.lose = cv2.VideoCapture(r"images\lose.mp4")
        self.outOfTime = cv2.VideoCapture(r"images\lose.mp4")
        self.score = 0
        self.level = 0
        self.totalScore = 0
        self._initialStone()
        self.minus = self._setUpImage(r"images\minus.png")
        self.plus = self._setUpImage(r"images\plus.png")
        self.buttonsManager = ButtonManager(configPath)

    def _up(self, velocity: int) -> None:
        for ball in self.balls:
            ball[1] += velocity
        if self.level >= 2:
            for stone in self.stones:
                stone[1] += velocity

    def _setUpImage(self, imagePath):
        obj = cv2.imread(imagePath)
        obj = cv2.resize(
            obj, (2 * int(self.ballRadius / 1.4142) + 1, 2 * int(self.ballRadius / 1.4142) + 1))
        obj = cv2.flip(obj, 1)
        return obj

    def _initialBalls(self, numBalls: int):
        self.balls = []
        for _ in range(numBalls):
            y = random.randint(self.lowerBoundY, self.upperBoundY)
            x = random.randint(self.lowerBoundX + self.ballRadius,
                               self.upperBoundX - self.ballRadius)
            self.balls.append([x, y])
        self.balls = numpy.array(self.balls)

    def _dropBall(self, index: int) -> None:
        x = random.randint(self.lowerBoundX + self.ballRadius,
                           self.upperBoundX - self.ballRadius)
        y = random.randint(self.lowerBoundY, self.upperBoundY)
        while self._isCollision(x, y, self.balls):
            x = random.randint(self.lowerBoundX + self.ballRadius,
                               self.upperBoundX - self.ballRadius)
            y = random.randint(self.lowerBoundY, self.upperBoundY)
        self.balls[index] = [x, y]

    def _initialStone(self):
        self.stones = []
        limitY = self.limit - self.ballRadius - \
            self.config["mode2"]["limitToWin"]
        for _ in range(len(self.balls) * 2):
            y = random.randint(self.lowerBoundY, limitY)
            x = random.randint(self.lowerBoundX + self.ballRadius,
                               self.upperBoundX + self.ballRadius)
            while self._isCollision(x, y, self.balls):
                y = random.randint(self.lowerBoundY, limitY)
                x = random.randint(
                    self.lowerBoundX + self.ballRadius, self.upperBoundX + self.ballRadius)
            self.stones.append([x, y])
        self.stones = numpy.array(self.stones)

    def _dropStones(self, index: int) -> None:
        limitY = self.limit - self.ballRadius - \
            self.config["mode2"]["limitToWin"]
        x = random.randint(self.lowerBoundX + self.ballRadius,
                           self.upperBoundX - self.ballRadius)
        y = random.randint(self.lowerBoundY, limitY)
        while self._isCollision(x, y, self.stones) and self._isCollision(x, y, self.balls):
            x = random.randint(self.lowerBoundX + self.ballRadius,
                               self.upperBoundX - self.ballRadius)
            y = random.randint(self.lowerBoundY, limitY)
        self.stones[index] = [x, y]

    def _isCollision(self, x, y, existingBalls):
        distances = numpy.sqrt(
            (existingBalls[:, 0] - x) ** 2 + (existingBalls[:, 1] - y) ** 2)
        return numpy.any(distances < self.ballRadius * 2 + 2)

    def _displayItem(self, frame, x, y, item):
        obj = item.copy()
        assert obj.shape[0] == obj.shape[1], "Item not a square" + \
            str(obj.shape)
        size = obj.shape[0]
        frame[y - size // 2: y + (size + 1) // 2,
              x - size // 2: x + (size + 1) // 2] = obj

    def _setUpBackground(self, frame):
        temp = numpy.full(frame.shape, fill_value=(
            23, 127, 222), dtype=frame.dtype)
        return temp

    def _searchBall(self, x, y):
        rightBalls = []
        wrongItems = []
        for i, ball in enumerate(self.balls):
            if ball[0] - self.ballRadius <= x and ball[0] + self.ballRadius >= x:
                if ball[1] - self.ballRadius <= y and ball[1] + self.ballRadius >= y:
                    rightBalls.append([ball[0], ball[1]])
                    self._dropBall(i)
                    self.score += 1
        if self.level >= 2:
            for i, stone in enumerate(self.stones):
                if stone[0] - self.ballRadius <= x and stone[0] + self.ballRadius >= x:
                    if stone[1] - self.ballRadius <= y and stone[1] + self.ballRadius >= y:
                        wrongItems.append([stone[0], stone[1]])
                        self._dropStones(i)
                        self.score -= 1
        return rightBalls, wrongItems

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
        nextLevel = False
        handLocation = None
        finalRound = False
        timeUp = 0
        remainTime = 1
        image = None
        while True:
            cursorX = -1
            cursorY = -1
            ret, frame = self.video.read()

            if not ret:
                break
            mask = setUpBackground(frame)
            self.buttonsManager.setupQuit(mask, reversed=True)
            retIntro, frameIntro = self.introVideo.read()
            if retIntro:
                frameIntro = cv2.resize(
                    frameIntro, (frame.shape[1], frame.shape[0]))
                mask = frameIntro
            else:
                if int(remainTime) <= 0:
                    retLose, frameLose = self.outOfTime.read()
                    if not retLose:
                        break
                    frameLose = cv2.resize(
                        frameLose, (frame.shape[1], frame.shape[0]))
                    mask = frameLose
                else:
                    if self.totalScore <= 30:
                        if self.score < -10:
                            retLose, frameLose = self.lose.read()
                            if not retLose:
                                break
                            frameLose = cv2.resize(
                                frameLose, (frame.shape[1], frame.shape[0]))
                            mask = frameLose
                        elif -10 <= self.score < 10:
                            if nextLevel:
                                retNext, frameNext = self.nextLevel.read()
                                if retNext:
                                    frameNext = cv2.resize(
                                        frameNext, (frame.shape[1], frame.shape[0]))
                                    mask = frameNext
                                else:
                                    if self.level < 2:
                                        self.nextLevel = cv2.VideoCapture(
                                            r"images\next_level.mp4")
                                    else:
                                        self.nextLevel = cv2.VideoCapture(
                                            r"images\final.mp4")
                                        timeUp = time.time() + \
                                            self.config["mode2"]["timeLimit"]
                                    nextLevel = False

                            else:
                                hand, image = self.detector.findHands(
                                    frame, draw=True)

                                if len(hand) > 0:
                                    landmarks = hand[0]['lmList']
                                    cursor = landmarks[8]
                                    self.buttonsManager.moveToQuitButton(
                                        mask, cursor)
                                    displayedHand = self._displayHand(
                                        landmarks, (frame.shape[0], frame.shape[1]))
                                    handLocation = drawSkeleton(
                                        displayedHand, frame)
                                    cursorX, cursorY = cursor[0], cursor[1]
                                else:
                                    cursorX = -1
                                    cursorY = -1
                                    handLocation = None
                                if cursorX >= 0 and cursorY >= 0:
                                    rightBalls, wrongItems = self._searchBall(
                                        cursorX, cursorY)
                                    if len(rightBalls) > 0:
                                        for ball in rightBalls:
                                            self._displayItem(
                                                mask, ball[0], ball[1], self.plus)
                                    if len(wrongItems) > 0:
                                        for stone in wrongItems:
                                            self._displayItem(
                                                mask, stone[0], stone[1], self.minus)
                                if self.level == 0:
                                    velocity = self.config["mode2"]["velocity"] // 2
                                elif self.level == 1:
                                    velocity = self.config["mode2"]["velocity"]
                                else:
                                    velocity = 2 * \
                                        self.config["mode2"]["velocity"]
                                self._up(velocity=velocity)
                                for i in range(len(self.balls)):
                                    if self.balls[i][1] >= self.limit:
                                        self._displayItem(
                                            mask, self.balls[i][0], self.limit, self.minus)
                                        self._dropBall(i)
                                        self.score -= 1
                                for ball in self.balls:
                                    self._displayItem(
                                        mask, ball[0], ball[1], self.item)
                                if self.level >= 2:
                                    for i in range(len(self.stones)):
                                        if self.stones[i][1] >= self.height - self.config["global"]["limit"]:
                                            self._dropStones(i)
                                    for stone in self.stones:
                                        self._displayItem(
                                            mask, stone[0], stone[1],  self.stone)
                                self._displayUserScreen(mask, image)
                                mask = cv2.flip(mask, 1)
                                if handLocation is not None:
                                    for x, y in list(zip(handLocation[0], handLocation[1])):
                                        mask[x][y] = (255, 255, 255)
                                cv2.line(mask, (0, self.limit + self.ballRadius), (self.width -
                                         self.lowerBoundX, self.limit + self.ballRadius), (0, 0, 255), thickness=2)
                                cv2.putText(mask, "Score: " + str(self.score) + "/10", (0, 25), fontScale=1.0,
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                                if self.level == 3:
                                    remainTime = timeUp - time.time()
                                    cv2.putText(mask, "Remaining Time: " + str(round(remainTime, 2)), (0, 50), fontScale=1,
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA
                                                )
                        elif self.score == 10:
                            nextLevel = True
                            self.score -= 10
                            self.totalScore += 10
                            self.level += 1
                    else:
                        retWin, frameWin = self.win.read()
                        if not retWin:
                            break
                        frameWin = cv2.resize(
                            frameWin, (frame.shape[1], frame.shape[0]))
                        mask = frameWin
            winname = "Pamdom Balls"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 360, 30)
            cv2.imshow(winname, mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    app = RandomBallsMode2(cap, r"config\config.yaml")
    app.run()