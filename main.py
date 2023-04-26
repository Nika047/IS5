import cv2
import numpy as np
import imutils

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

colors = {
    'yellow': (np.array([20, 10, 30]), np.array([40, 255, 255]),np.array([10, 180, 200]), np.array([20, 255, 255])),
    'blue': (np.array([110, 10, 30]), np.array([130, 255, 255]),np.array([90, 180, 200]), np.array([110, 255, 255])),
    'green': (np.array([45, 10, 30]),  np.array([75, 255, 255])),
    'red': (np.array([0, 10, 30]), np.array([20, 255, 255]), np.array([160, 10, 30]), np.array([180, 255, 255]))
}

relatives = {
    0: ('Rhombus'),
    1: ('Star'),
    2: ('Cloud'),
    3: ('Lightning'),
    4: ('Arrow'),
    5: ('Heart'),
    6: ('Moon')
}

image = cv2.imread('C:/Users/1/Desktop/IS_img/IS5.png')
im = imutils.resize(image, width= 600)
output = im.copy()
cv2.imshow('Input', image)

def color_check():
    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    for col in colors:
        mask = cv2.inRange(hsv, colors.get(col)[0], colors.get(col)[1])
        if col == 'blue':
            mask += cv2.inRange(hsv, colors.get(col)[2], colors.get(col)[3])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [i for i in contours if cv2.contourArea(i) > 50]
        find_fig(contours, mask, col)

def find_fig(contours, mask, col):
    global amountRhombus
    global amountStar
    global amountCloud
    global amountLightning
    global amountArrow
    global amountHeart
    global amountMoon
    global colorRhombus
    global colorStar
    global colorCloud
    global colorLightning
    global colorArrow
    global colorHeart
    global colorMoon

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        cv2.drawContours(output, [cnt], -1, (0, 0, 0), 1)
        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = mask[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            num = int(results[0])
            result = relatives[num]

            if result == 'Rhombus':
                amountRhombus += 1
                if colorRhombus[0] == " ":
                    colorRhombus[0] = col
                elif colorRhombus[1] == " ":
                    colorRhombus[1] = col
                elif colorRhombus[2] == " ":
                    colorRhombus[2] = col
                elif colorRhombus[3] == " ":
                    colorRhombus[3] = col
            elif result == 'Star':
                amountStar += 1
                if colorStar[0] == " ":
                    colorStar[0] = col
                elif colorStar[1] == " ":
                    colorStar[1] = col
                elif colorStar[2] == " ":
                    colorStar[2] = col
                elif colorStar[3] == " ":
                    colorStar[3] = col
            elif result == 'Cloud':
                amountCloud += 1
                if colorCloud[0] == " ":
                    colorCloud[0] = col
                elif colorCloud[1] == " ":
                    colorCloud[1] = col
                elif colorCloud[2] == " ":
                    colorCloud[2] = col
                elif colorCloud[3] == " ":
                    colorCloud[3] = col
            elif result == 'Lightning':
                amountLightning += 1
                if colorLightning[0] == " ":
                    colorLightning[0] = col
                elif colorLightning[1] == " ":
                    colorLightning[1] = col
                elif colorLightning[2] == " ":
                    colorLightning[2] = col
                elif colorLightning[3] == " ":
                    colorLightning[3] = col
            elif result == 'Arrow':
                amountArrow += 1
                if colorArrow[0] == " ":
                    colorArrow[0] = col
                elif colorArrow[1] == " ":
                    colorArrow[1] = col
                elif colorArrow[2] == " ":
                    colorArrow[2] = col
                elif colorArrow[3] == " ":
                    colorArrow[3] = col
            elif result == 'Heart':
                amountHeart += 1
                if colorHeart[0] == " ":
                    colorHeart[0] = col
                elif colorHeart[1] == " ":
                    colorHeart[1] = col
                elif colorHeart[2] == " ":
                    colorHeart[2] = col
                elif colorHeart[3] == " ":
                    colorHeart[3] = col
            elif result == 'Moon':
                amountMoon += 1
                if colorMoon[0] == " ":
                    colorMoon[0] = col
                elif colorMoon[1] == " ":
                    colorMoon[1] = col
                elif colorMoon[2] == " ":
                    colorMoon[2] = col
                elif colorMoon[3] == " ":
                    colorMoon[3] = col

            text1 = "{} {}".format(col, result)
            cv2.putText(output, text1, (x + w // 2 - 50, y + h // 2), 0, 0.6, (0, 0, 0), 1)

cp = im.copy()
gray = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
cnt = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

text2 = "Group 1044. Hon Veronika Vladislavovna"
cv2.putText(output, text2, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

amountRhombus = 0
amountStar = 0
amountCloud = 0
amountLightning = 0
amountArrow = 0
amountHeart = 0
amountMoon = 0
colorRhombus = [" "] * 4
colorStar = [" "] * 4
colorCloud = [" "] * 4
colorLightning = [" "] * 4
colorArrow = [" "] * 4
colorHeart = [" "] * 4
colorMoon = [" "] * 4

color_check()

print ("\nFigures:\n")
print ("Rhombus: {}".format(amountRhombus))
print ("Colors: {}".format(colorRhombus))
print ("Star: {}".format(amountStar))
print ("Colors: {}".format(colorStar))
print ("Cloud: {}".format(amountCloud))
print ("Colors: {}".format(colorCloud))
print ("Lightning: {}".format(amountLightning))
print ("Colors: {}".format(colorLightning))
print ("Arrow: {}".format(amountArrow))
print ("Colors: {}".format(colorArrow))
print ("Heart: {}".format(amountHeart))
print ("Colors: {}".format(colorHeart))
print ("Moon: {}".format(amountMoon))
print ("Colors: {}\n".format(colorMoon))
print ("Total figures: {}".format(len(cnt)))

cv2.imshow('Output', output)
cv2.waitKey(0)