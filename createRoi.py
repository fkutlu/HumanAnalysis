# import the necessary packages
import argparse
import cv2

import os
import json
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def createRoi(cap):

	def click_and_crop(event, x, y, flags, param):
		# grab references to the global variables
		global refPt, cropping

		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed
		if event == cv2.EVENT_LBUTTONDOWN:
			refPt = [(x, y)]
			cropping = True

		# check to see if the left mouse button was released
		elif event == cv2.EVENT_LBUTTONUP:
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			refPt.append((x, y))
			cropping = False

			# draw a rectangle around the region of interest
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", image)


			#		cap = cv2.VideoCapture('Relaxinghighwaytraffic640480.avi')

	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)

	while(cap.isOpened()):
		image = cap.read()[1]

		#rotate image as desired (deg)
		#show image
		cv2.imshow("image", image)

		#wait until a key is pressed if desired roi is obtained press 'c'
		key = cv2.waitKey(-1) & 0xFF
		if key == ord("c"):
			break

	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(refPt) == 2:
		with open('roi.json', 'w') as outfile:
			json.dump(refPt, outfile)
		cv2.destroyAllWindows()
	else:
		print ("ERROR: Roi is not selected!!!")
