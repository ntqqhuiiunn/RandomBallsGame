import cv2
import numpy
import sys
from cvzone.HandTrackingModule import HandDetector
from src.buttons_manager import ButtonManager
from src.utils import loadConfig, showMouse
from src.level1 import Level1


class Game:
    def __init__(self, configPath: str):
        self.video = cv2.VideoCapture(0)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector = HandDetector(
            maxHands=1, detectionCon=0.8, minTrackCon=0.7)
        self.configPath = configPath
        self.config = loadConfig(self.configPath)
        self.buttonManager = ButtonManager(self.config)
        self.loading_path = r"images\loading.mp4"
        self.item = None
        self.nameItem = None
        self.fixedSize = (1100, 640)
        self.continueMode1 = False
        self.continueMode2 = False
        self.mouse = cv2.imread(r"images\mouse.png")
        self.mouse = cv2.resize(self.mouse, (41, 41),
                                interpolation=cv2.INTER_NEAREST)
        self._setUpBackground()
        self.program = Level1(self.config, self.video, self.detector)
        self.userFrame = (280, 210)

    def _setUpBackground(self):
        self.background = numpy.full((self.fixedSize[1], self.fixedSize[0], 3), fill_value=(
            23, 127, 227), dtype=numpy.uint8)

    def run(self):
        touchStart = False
        clicked = False
        retLoading = False
        retLoading2 = False
        cursorX = -1
        cursorY = -1
        numberOfReplay = 1
        reLoadingVideo = cv2.VideoCapture(self.loading_path)
        while True:
            ret, frame = self.video.read()
            mask = self.background.copy()
            self.buttonManager.setupTitle(mask)
            self.buttonManager.setupQuit(mask)
            if not ret:
                break

            if touchStart:
                if numberOfReplay > 0:
                    numberOfReplay -= 1
                    alpha = self.program.run()
                retReload, frameReload = reLoadingVideo.read()
                if alpha and not retReload:
                    self.run()
                else:
                    mask = cv2.resize(frameReload, self.fixedSize)
            else:
                self.buttonManager.displayStartButton(mask)
            hand, image = self.detector.findHands(frame, draw=True)

            if len(hand) > 0:
                if hand[0]["type"] == "Right":
                    landmarks = hand[0]['lmList']
                    cursor = landmarks[8]
                    cursorX = cursor[0]
                    cursorY = cursor[1]
                    first = (landmarks[4][1] - landmarks[5][1]) ** 2 + \
                        (landmarks[4][0] - landmarks[5][0]) ** 2
                    second = (landmarks[0][1] - landmarks[5][1]) ** 2 + \
                        (landmarks[0][0] - landmarks[5][0]) ** 2 + 0.001
                    ratio = first / second
                    clicked = ratio < 0.1
                    self.buttonManager.moveToQuitButton(
                        mask, cursor, retLoading, retLoading2)
                    if not touchStart:
                        touch = self.buttonManager.moveOverStartButton(
                            mask, cursor, clicked)
                        if touch:
                            touchStart = True
            else:
                clicked = False
                cursorX = -1
                cursorY = -1
            if cursorX >= 0 and cursorY >= 0:
                showMouse(mask, (cursorX, cursorY), self.mouse, "main")
            winname = "Pamdom Balls"
            cv2.namedWindow(winname, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(winname, 50, 10)
            cv2.imshow(winname, mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                sys.exit()


if __name__ == "__main__":
    game = Game(r"config\config.yaml")
    game.run()
