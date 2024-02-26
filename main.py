import cv2
import numpy
import time
import sys
from cvzone.HandTrackingModule import HandDetector
from src.buttons_manager import ButtonManager
from src.random_balls_mode2 import RandomBallsMode2
from src.random_balls_mode1 import RandomBallsMode1
from src.utils import loadConfig, showMouse


class Game:
    def __init__(self, configPath: str):
        self.video = cv2.VideoCapture(0)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector = HandDetector(
            maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.configPath = configPath
        self.config = loadConfig(self.configPath)
        self.buttonManager = ButtonManager(configPath)
        self.loading_path = r"images\loading.mp4"
        self.item = None
        self.nameItem = None
        self.fixedSize = self.config["global"]["fixedFrameSize"]

    def displayHand(self, landmarks, size):
        height, width = size[0], size[1]
        tempLandmarks = [(width - landmark[0], landmark[1])
                         for landmark in landmarks]
        return tempLandmarks

    def _setUpBackground(self, frame):
        temp = numpy.full(frame.shape, fill_value=(
            23, 127, 227), dtype=frame.dtype)
        return temp

    def run(self):
        touchStart = False
        touchMode1 = False
        touchMode2 = False
        selectedItem = False
        loadingVideo = cv2.VideoCapture(self.loading_path)
        loadingVideo2 = cv2.VideoCapture(self.loading_path)
        loading1 = False
        loading2 = False
        clicked = False
        cursorX = -1
        cursorY = -1
        while True:
            handLocation = None
            ret, frame = self.video.read()
            mask = self._setUpBackground(frame)
            self.buttonManager.setupTitle(mask)
            self.buttonManager.setupQuit(mask)
            if not ret:
                break

            if touchStart:
                if loading1:
                    retLoading, loadingFrame = loadingVideo.read()
                    if retLoading:
                        loadingFrame = cv2.resize(loadingFrame, (640, 480))
                        mask = loadingFrame
                    else:
                        loading1 = False
                else:
                    if not selectedItem:
                        mask = self._setUpBackground(frame)
                        self.buttonManager.displaySelectItem(mask)
                    else:
                        retLoading2, loadingFrame2 = loadingVideo2.read()
                        if retLoading2:
                            loadingFrame2 = cv2.resize(
                                loadingFrame2, (640, 480))
                            mask = loadingFrame2
                        else:
                            loading2 = True
                            self.buttonManager.displaySelectModeButton(mask)
            else:
                self.buttonManager.displayStartButton(mask)
            if touchMode1:
                app = RandomBallsMode1(
                    self.video, self.configPath, self.item, self.nameItem)
                app.run()
            if touchMode2:
                app = RandomBallsMode2(
                    self.video, self.configPath, self.item, self.nameItem)
                app.run()
            hand, image = self.detector.findHands(frame, draw=True)
            if len(hand) > 0:
                landmarks = hand[0]['lmList']
                cursor = landmarks[8]
                cursorX = cursor[0]
                cursorY = cursor[1]
                ratio = (landmarks[6][1] - cursor[1]) / \
                    (landmarks[5][1] - cursor[1] + 0.0001)
                clicked = ratio > 1
                self.buttonManager.moveToQuitButton(mask, cursor)
                if not touchStart:
                    touch = self.buttonManager.moveOverStartButton(
                        mask, cursor, clicked)
                    if touch:
                        touchStart = True
                        loading1 = True
                else:
                    if not selectedItem:
                        result, name = self.buttonManager.getItem(
                            cursor, mask, clicked)
                        if result is not None:
                            selectedItem = True
                            self.item = result
                            self.nameItem = name
                    else:
                        if (not touchMode1 or not touchMode2) and not loading1 and loading2:
                            touched1 = self.buttonManager.selectMode1(
                                mask, cursor, clicked)
                            touched2 = self.buttonManager.selectMode2(
                                mask, cursor, clicked)
                            if touched1:
                                touchMode1 = True
                            if touched2:
                                touchMode2 = True
            #     displayedHand = self.displayHand(
            #         landmarks, (frame.shape[0], frame.shape[1]))
            #     handLocation = drawSkeleton(displayedHand, frame)
            # if handLocation is not None:
            #     for x, y in list(zip(handLocation[0], handLocation[1])):
            #         mask[x][y] = (255, 255, 255)

            else:
                clicked = False
                cursorX = -1
                cursorY = -1
            if cursorX >= 0 and cursorY >= 0:
                showMouse(mask, cursorX, cursorY, "main")
            print(mask.shape)
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
    game = Game(r"config\config.yaml")
    game.run()
