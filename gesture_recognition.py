#coding=utf-8

"""
File name:gesture_recognition.py
Version:v1.0

The last update time:2025/8/21
Code by AshenMajo
"""

import cv2
import mediapipe as mp
import math
from collections import deque

cap=cv2.VideoCapture(0)
mhands=mp.solutions.hands
hands=mhands.Hands(max_num_hands=1)
handsdraw=mp.solutions.drawing_utils

prev_angle=None
ANGLE_THRESHOLD = 0.05

angle_buffer=deque(maxlen=5)

def isfist(handlandmarks):
   tips=[8,12,16,20]
   pips=[6,10,14,18]

   for tip,pip in zip(tips,pips):
      if handlandmarks.landmark[tip].y<handlandmarks.landmark[pip].y:
         return False
   return True

def getAngle(handlandmarks):

   index_mcp=handlandmarks.landmark[5]
   pinky_mcp=handlandmarks.landmark[17]

   vx=pinky_mcp.x-index_mcp.x
   vy=pinky_mcp.y-index_mcp.y

   angle=math.atan2(vy,vx)

   return angle

def rotation_direction(handlandmarks):

   global prev_angle
   angle=getAngle(handlandmarks)
   direction='Stable'

   angle_buffer.append(angle)
   smoothed_angle=sum(angle_buffer)/len(angle_buffer)

   if prev_angle is not None:
      delta=smoothed_angle-prev_angle

      if delta>math.pi:
         delta=delta-2*math.pi
      elif delta <-math.pi:
         delta=delta+2*math.pi

      if abs(delta)<ANGLE_THRESHOLD:
         direction=''
      elif delta<0:
         direction='Counterclockwise'
      elif delta>0:
         direction='Clockwise'
          
   prev_angle=smoothed_angle
   return direction,smoothed_angle

while(True):
   ret,img=cap.read()
   img=cv2.flip(img,1)
   imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   result=hands.process(imgRGB)

   gestureText="Gesture:"
   direction,angle='None',0

   print(result.multi_hand_landmarks)
   if result.multi_hand_landmarks:
      for hand_location in result.multi_hand_landmarks:
         handsdraw.draw_landmarks(img,hand_location,mhands.HAND_CONNECTIONS)

         if isfist(hand_location):
            gestureText='Gesture:fist'

         direction,angle=rotation_direction(hand_location)
   cv2.putText(img,gestureText,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
   cv2.putText(img,f'angle:{angle:.2f} direction:{direction}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

   if ret:
      cv2.imshow('Camera',img)
   if cv2.waitKey(1)==ord('q'):
      break

hands.close()
cap.release()
cv2.destroyAllWindows()