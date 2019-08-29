import cv2 as cv
import numpy as np

cap = cv.VideoCapture('worktrain.mp4')
bg = cv.createBackgroundSubtractorMOG2()

while(cap.isOpened()):
    ret, frame = cap.read()
    bg_frame = bg.apply(frame)
    ret, thresh = cv.threshold(bg_frame, 127, 255, 0)
    contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv.contourArea(c)
        if area >= 22000:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
        else:
            continue
    cv.imshow('video',frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()