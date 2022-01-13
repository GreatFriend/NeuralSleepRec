from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import numpy as np
import dlib
from scipy.spatial import distance as dist

counter=0
THRESHOLD_EAR = 0.3
FRAMES_EAR = 48
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def findEAR(eye):
    A = dist.euclidean([eye[1].x,eye[1].y], [eye[5].x,eye[5].y])
    B = dist.euclidean([eye[2].x,eye[2].y], [eye[4].x,eye[4].y])
    C = dist.euclidean([eye[0].x,eye[0].y], [eye[3].x,eye[3].y])
    ear = (A + B) / (2.0 * C)

    return ear

def index(request):
    try:
        cam = VideoCamera()
        counter=0
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'edit/index.html')

class VideoCamera(object):
    counter=0
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        frame=image
        brightness = 50
        contrast = 30
        frame = np.int16(frame)
        frame = frame * (contrast/127+1) - contrast + brightness
        frame = np.clip(frame, 0, 255)
        frame = np.uint8(frame)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray1 = cv2.filter2D(gray, -1, kernel)
        faces=detector(gray1)
        for face in faces:
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

            landmarks=predictor(gray,face)
            for n in range(36,48):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(frame,(x,y),1,(0,0,255),-1)

            
            eyeR=float(findEAR((landmarks.part(36),landmarks.part(37),landmarks.part(38),landmarks.part(39),landmarks.part(40),landmarks.part(41))))
            eyeL=float(findEAR((landmarks.part(42),landmarks.part(43),landmarks.part(44),landmarks.part(45),landmarks.part(46),landmarks.part(47))))
            eyeR=float('{:.3f}'.format(eyeR))
            eyeL=float('{:.3f}'.format(eyeL))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, str(eyeR)+ " R", (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, str(eyeL)+ " L", (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)
            #cv2.putText(frame, str(counter)+ "", (50, 150), font, fontScale, color, thickness, cv2.LINE_AA)
            
            
            if eyeR<THRESHOLD_EAR and eyeL<THRESHOLD_EAR and self.counter<=FRAMES_EAR+10:
                self.counter+=1
            if self.counter>0 and eyeR>THRESHOLD_EAR and eyeL>THRESHOLD_EAR:
                self.counter=0
            if self.counter>=FRAMES_EAR:
                cv2.putText(frame, "Wake up!", (50, 200), font, fontScale, color, thickness, cv2.LINE_AA)

        
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

