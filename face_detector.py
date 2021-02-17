import cv2
from random import randrange

tfd = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

cv2.namedWindow("fit",cv2.WINDOW_FREERATIO)

while True:
	ans, frame = cam.read()

	gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	fc = tfd.detectMultiScale(gs_img)

	for i in range(0,len(fc)):
		start = (fc[i][0],fc[i][1])
		end = (fc[i][2]+start[0],fc[i][3]+start[1])

		color = (0, 255 ,0)

		thick = 5

		cv2.rectangle(frame,start,end,color,thick)

	cv2.imshow("fit",frame)

	key = cv2.waitKey(1)
	if key == 81:
		break