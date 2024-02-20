import cv2
import numpy
import time
import sys
from cvzone.HandTrackingModule import HandDetector
from src.buttons_manager import ButtonManager
from src.random_balls_mode2 import RandomBallsMode2
from src.random_balls_mode1 import RandomBallsMode1
from src.utils import setUpBackground

class Game:
    def __init__(self, configPath : str):
        self.video = cv2.VideoCapture(0)
        self.detector = HandDetector(
            maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.configPath = configPath
        self.buttonManager = ButtonManager(configPath)
        self.loading_path = r"images\loading.mp4"

    def createSkeleton(self, landmarks, frame):
        mask = numpy.zeros_like(frame)
        root = landmarks[0]
        points = [(root[0], root[1])]
        for i in range(5):
            cv2.line(frame, (root[0], root[1]), (landmarks[4 * i + 1][0],
                                                 landmarks[4 * i + 1][1]), color=(0, 255, 0), thickness=2)
            cv2.line(mask, (root[0], root[1]), (landmarks[4 * i + 1][0],
                                                landmarks[4 * i + 1][1]), color=(255, 255, 255), thickness=2)
            for j in range(1, 4):
                start = landmarks[4 * i + j]
                end = landmarks[4 * i + j + 1]
                cv2.line(frame, (start[0], start[1]), (end[0],
                                                       end[1]), color=(0, 255, 0), thickness=2)
                cv2.line(mask, (start[0], start[1]), (end[0],
                                                      end[1]), color=(255, 255, 255), thickness=2)

            if i > 0:
                points.append(
                    (landmarks[4 * i + 1][0], landmarks[4 * i + 1][1]))
        cv2.fillPoly(
            frame, [numpy.array(points, dtype=numpy.int32)], (0, 255, 0))
        cv2.fillPoly(
            mask, [numpy.array(points, dtype=numpy.int32)], (255, 255, 255))

        image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thMask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        handLocation = numpy.where(thMask > 0)
        return handLocation

    def displayHand(self, landmarks, size):
        height, width = size[0], size[1]
        tempLandmarks = [(width - landmark[0], landmark[1])
                         for landmark in landmarks]
        return tempLandmarks

    def run(self):
        touchStart = False
        touchMode1 = False
        touchMode2 = False
        loadingVideo = cv2.VideoCapture(self.loading_path)
        loading = False
        while True:
            handLocation = None
            ret, frame = self.video.read()
            mask = setUpBackground(frame)
            self.buttonManager.setupTitle(mask)
            self.buttonManager.setupQuit(mask)
            if not ret:
                break

            if touchStart:
                if loading:
                    retLoading, loadingFrame = loadingVideo.read()
                    if retLoading:
                        loadingFrame = cv2.resize(loadingFrame, (640, 480))
                        mask = loadingFrame
                    else:
                        loading = False
                else:
                    self.buttonManager.displaySelectModeButton(mask)
            else:
                self.buttonManager.displayStartButton(mask)
            if touchMode1:
                cv2.waitKey(75)
                app = RandomBallsMode1(self.video, self.configPath)
                app.run()
            if touchMode2:
                cv2.waitKey(75)
                app = RandomBallsMode2(self.video, self.configPath)
                app.run()
            hand, image = self.detector.findHands(frame, draw=False)
            if len(hand) > 0:
                landmarks = hand[0]['lmList']
                cursor = landmarks[8]
                self.buttonManager.moveToQuitButton(mask, cursor)
                if not touchStart:
                    touch = self.buttonManager.moveOverStartButton(
                        mask, cursor)
                    if touch:
                        touchStart = True
                        loading = True
                else:
                    if (not touchMode1 or not touchMode2) and not loading:
                        touched1 = self.buttonManager.selectMode1(mask, cursor)
                        touched2 = self.buttonManager.selectMode2(mask, cursor)
                        if touched1:
                            touchMode1 = True
                        if touched2:
                            touchMode2 = True
                displayedHand = self.displayHand(
                    landmarks, (frame.shape[0], frame.shape[1]))
                handLocation = self.createSkeleton(displayedHand, frame)
            if handLocation is not None:
                for x, y in list(zip(handLocation[0], handLocation[1])):
                    mask[x][y] = (255, 255, 255)
            winname = "Pamdom Balls"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 360, 30)
            cv2.imshow(winname, mask)
            frame = cv2.flip(frame, 1)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()


if __name__ == "__main__":
    game = Game(r"config\config.yaml")
    game.run()
