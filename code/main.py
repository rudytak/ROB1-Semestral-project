from ctu_crs import CRS97
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

robot = CRS97()
robot.initialize(home=True)

q = robot.get_q()
print(q)


img = robot.grab_image()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
detector = cv.aruco.ArucoDetector(aruco_dict)

corners, ids, rej = detector.detectMarkers(gray)
imgc = img.copy()
cv.aruco.drawDetectedMarkers(imgc,corners,ids)

plt.figure(figsize=(10,5))
plt.imshow(imgc)
plt.show()

robot.close()

input()