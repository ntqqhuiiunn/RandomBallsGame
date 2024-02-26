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


def showMouse(frame, x, y, mode="mode"):
    mouse = cv2.imread(r"images\mouse.png")
    mouse = cv2.resize(mouse, (41, 41))
    if mode == "mode":
        mouse = cv2.flip(mouse, 1)
    gray_mouse = cv2.cvtColor(mouse, cv2.COLOR_BGR2GRAY)
    if mode != "mode":
        x = frame.shape[1] - 1 - x
    if x <= 20:
        x = 20
    if x >= frame.shape[1] - 21:
        x = frame.shape[1] - 21
    if y <= 20:
        y = 20
    if y >= frame.shape[0] - 21:
        y = frame.shape[0] - 21
    selection = frame[y - 20: y + 21, x - 20: x + 21]
    selection[gray_mouse > 0] = mouse[gray_mouse > 0]


# def setUpBackground(frame, mode="main"):
#     if mode == "main":
#         temp = numpy.full(frame.shape, fill_value=(
#         23, 127, 227), dtype=frame.dtype)
#     elif mode == "mode1":

#     else:
#         temp = cv2.imread(r"images\back2.png")
#         temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))
#     return temp


def showCursorPoint(frame, cursor, mode="mode"):
    cursorX, cursorY = cursor[:2]
    if mode != "mode":
        cursorX = frame.shape[1] - 1 - cursorX
    cv2.circle(frame, (cursorX, cursorY), 5, (0, 0, 255), thickness=-1)


def createMask(item):
    def compare(array):
        return array[0] == 0 and array[1] == 0 and array[2] == 0
    height, width = item.shape[0], item.shape[1]
    mask = numpy.zeros((height, width), dtype=item.dtype)
    for h in range(height):
        row = item[h]
        start_index = 0
        end_index = width - 1
        while start_index < width and compare(row[start_index]):
            start_index += 1
        while end_index > 0 and compare(row[end_index]):
            end_index -= 1
        mask[h, start_index: end_index + 1] = 1
    return mask


item = cv2.imread(r"images\ball.png")
item = cv2.resize(item, (41, 41))
mask = createMask(item)
print(mask.shape)
frame = numpy.zeros((640, 480),  dtype=item.dtype)
size = item.shape[0]
x = 60
y = 70
selection = frame[y - size // 2: y +
                  (size + 1) // 2, x - size // 2: x + (size + 1) // 2]
print(selection.shape)
print(item[mask > 0])
print(selection[mask > 0])

a = numpy.array([[0, 0, 0, 0],
                 [0, 0, 1, 2],
                 [0, 2, 3, 0],
                 [0, 1, 0, 0]])
d = numpy.array([[0, 0, 0, 0],
                 [0, 0, 1, 2],
                 [0, 2, 3, 0],
                 [0, 1, 0, 0]])
b = a > 0
c = numpy.array([[-1, -1, -1, -1],
                 [-1, -1, -1, -1],
                 [-1, -1, -1, -1],
                 [-1, -1, -1, -1]])
print(b)
c[a > 1] = d[a > 1]
print(c)
