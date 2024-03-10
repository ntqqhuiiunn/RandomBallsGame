import yaml
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


def showMouse(frame, cursor, mouse, mode="mode"):
    x = cursor[0]
    y = cursor[1]
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

# def getRatioPause(landmarks: numpy.ndarray):
#     first = landmarks[[7, 11, 15, 19]]
#     second = landmarks[[6, 10, 14, 18]]
#     anchor = landmarks[0]
#     second_number = numpy.add(numpy.square(
#         second[:, 1] - anchor[1]), numpy.square(second[:, 0] - anchor[0]))

#     first_number = numpy.add(numpy.square(
#         first[:, 1] - anchor[1]), numpy.square(first[:, 0] - anchor[0]))
#     first_number = numpy.multiply(first_number, [100, 1, 1, 1])
#     second_number = numpy.multiply(second_number, [100, 1, 1, 1])

#     ratio = numpy.sum(first_number) / (numpy.sum(second_number) + 0.001)
#     return ratio


def getRatioPause(landmarks: numpy.ndarray):
    first = landmarks[[7, 11, 15, 19]]
    second = landmarks[[6, 10, 14, 18]]
    distance = numpy.subtract(first[:, 1], second[:, 1])
    return numpy.all(distance > 0)
# def getRatioOpen(landmarks: numpy.ndarray):
#     first = landmarks[[11, 15, 19]]
#     second = landmarks[[10, 14, 18]]
#     anchor = landmarks[0]
#     second_number = numpy.add(numpy.square(
#         second[:, 1] - anchor[1]), numpy.square(second[:, 0] - anchor[0]))

#     first_number = numpy.add(numpy.square(
#         first[:, 1] - anchor[1]), numpy.square(first[:, 0] - anchor[0]))
#     first_number = numpy.sum(first_number)
#     second_number = numpy.sum(second_number)

#     ratio1 = numpy.sqrt(first_number) / numpy.sqrt(second_number)
#     a = numpy.subtract(landmarks[4][:2], landmarks[8][:2])
#     a = numpy.sum(numpy.square(a))
#     b = numpy.subtract(landmarks[1][:2], landmarks[2][:2])
#     b = numpy.sum(numpy.square(b))
#     ratio2 = a / (b + 0.001)
#     return ratio1 > 1 and ratio2 < 0.5


def getRatioOpen(landmarks: numpy.ndarray):
    first = landmarks[[7, 11, 15, 19]]
    second = landmarks[[6, 10, 14, 18]]
    distance = numpy.subtract(first[:, 1], second[:, 1])
    remainder = distance[1:]
    return distance[0] > 0 and numpy.all(remainder < 0)
