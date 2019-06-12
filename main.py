from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import dlib
import cv2

def sound_alarm(path):
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	A=dist.euclidean(eye[1],eye[5])
	B=dist.euclidean(eye[2],eye[4])
	C=dist.euclidean(eye[0],eye[3])
	ear=(A+B)/(2.0*C)
	return ear


ap=argparse.ArgumentParser()
ap.add_argument("-s","--shapePredictorPath",required=True,help="This is the path to the shape predictor file of the dlib library")
ap.add_argument("-a","--alarmPath",required=True,help="This is the path to the alarm which will sound in case the driver's eye aspect ration is found to be below the threshold")
args=vars(ap.parse_args())


EYE_AR_THRESH=0.25
EYE_AR_CONSEC_FRAMES=50
counter=0
alarm_on=False

print("[INFO] Loading Facial Landmarks detector")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shapePredictorPath"])

print("[INFO] Loading Camera for Monitoring")
vs=VideoStream(-1).start()

(lstart,lend)= face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rend)= face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while True:
	frame=vs.read()
	frame=imutils.resize(frame,width=500)
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	rects=detector(gray,0)
	for rect in rects:
		shape=predictor(gray,rect)
		shape=face_utils.shape_to_np(shape)
		leftEye=shape[lstart:lend]
		rightEye=shape[rstart:rend]
		leftEAR=eye_aspect_ratio(leftEye)
		rightEAR=eye_aspect_ratio(rightEye)
		ear=(leftEAR+rightEAR)/2.0
		leftEyeHull= cv2.convexHull(leftEye)
		rightEyeHull= cv2.convexHull(rightEye)
		cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
		if ear<EYE_AR_THRESH:
			counter+=1
			if counter>=EYE_AR_CONSEC_FRAMES:
				if not alarm_on:
					alarm_on=True
					t=Thread(target=sound_alarm,args=(args["alarm"],))
					t.deamon=True
					t.start()
			cv2.putText(frame,"DROWSINESS ALERT!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)
		else:
			counter=0
			alarm_on=False
		cv2.putText(frame,"EAR: {: .2f}".format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
	cv2.imshow("Frame",frame)
	key=cv2.waitKey(1) & 0xFF
	if key==ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
