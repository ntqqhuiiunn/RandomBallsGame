import yaml
import random
import numpy
import cv2


def loadConfig(path: str):
    with open(path, mode="r") as file:
        configLoader = yaml.load(file, Loader=yaml.Loader)
    return configLoader


def drawSkeleton(landmarks, frame):
    mask = numpy.zeros_like(frame)
    root = landmarks[0]
    points = [(root[0], root[1])]
    for i in range(5):
        cv2.line(mask, (root[0], root[1]), (landmarks[4 * i + 1][0],
                 landmarks[4 * i + 1][1]), color=(255, 255, 255), thickness=2)
        for j in range(1, 4):
            start = landmarks[4 * i + j]
            end = landmarks[4 * i + j + 1]
            cv2.line(mask, (start[0], start[1]), (end[0],
                     end[1]), color=(255, 255, 255), thickness=2)

        if i > 0:
            points.append((landmarks[4 * i + 1][0], landmarks[4 * i + 1][1]))
    cv2.fillPoly(
        mask, [numpy.array(points, dtype=numpy.int32)], (255, 255, 255))

    image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thMask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    handLocation = numpy.where(thMask > 0)
    return handLocation
def setUpBackground(frame):
    temp = numpy.full(frame.shape, fill_value=(
            23, 127, 222), dtype=frame.dtype)
    return temp