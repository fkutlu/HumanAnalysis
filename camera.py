import threading
from fer.fer import FER
import cv2
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import os
import numpy as np
import face_recognition
import json
import collections
import time
import createRoi
def crop_face(imgarray, section, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()

class VideoCamera(object):

    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)
        self.totalFrame = 0

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

        WRN_WEIGHTS_PATH = "./pretrained_models/weights.18-4.06.hdf5"

        self.face_size = 64
        self.depth=16
        self.width=8
        self.model = WideResNet(self.face_size, depth=self.depth, k=self.width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)

        self.model.load_weights(fpath)
        self.detector = FER(mtcnn=True)

        self.known_face_encodings = []
        self.face_locations = []
        self.face_encodings = []
        self.totalFace = 0
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        self.totalFrame += 1
        ret, src = self.cap.read()

        if ret:
            frame = src.copy()
            frame_draw=frame.copy()
            rgb_frame = frame[:, :, ::-1]
            self.totalFrame=0

            res=self.detector.detect_emotions(frame)
            if len(res)>0:
                for i, face in enumerate(res):
                    maxValue=-1
                    emotion=""
                    if face["emotions"]["angry"]>maxValue :
                        maxValue=face["emotions"]["angry"]
                        emotion="angry"
                    if face["emotions"]["disgust"]>maxValue :
                        maxValue=face["emotions"]["disgust"]
                        emotion="disgust"
                    if face["emotions"]["fear"]>maxValue :
                        maxValue=face["emotions"]["fear"]
                        emotion="fear"
                    if face["emotions"]["happy"]>maxValue :
                        maxValue=face["emotions"]["happy"]
                        emotion="happy"
                    if face["emotions"]["sad"]>maxValue :
                        maxValue=face["emotions"]["sad"]
                        emotion="sad"
                    if face["emotions"]["surprise"]>maxValue :
                        maxValue=face["emotions"]["surprise"]
                        emotion="surprise"
                    if face["emotions"]["neutral"]>maxValue :
                        maxValue=face["emotions"]["neutral"]
                        emotion="neutral"
                    self.face_locations = []
                    self.face_locations.append((face['box'][1], face['box'][0]+face['box'][2],face['box'][1]+face['box'][3],face['box'][0]))
                    self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
                    if len(self.known_face_encodings)>15:
                        self.known_face_encodings.pop(0)
                    matches = face_recognition.compare_faces(self.known_face_encodings, self.face_encodings[0])
                    face_id=-1
                    if True in matches:
                        face_id=matches.index(True)
                    else:
                        self.known_face_encodings.append(self.face_encodings[0])
                    face_imgs = np.empty((1, self.face_size, self.face_size, 3))
                    face_img, cropped = crop_face(frame, face['box'], margin=40, size=self.face_size)
                    (x, y, w, h) = face['box']

                    face_imgs[0, :, :, :] = face_img

                    results = self.model.predict(face_imgs)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()
                    if predicted_genders[0][0] > 0.5:
                        #print("Female")
                        color=(0, 0, 255)
                    else:
                        #print("Male")
                        color = (255, 0,0)

                    cv2.rectangle(frame_draw, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame_draw, emotion, (x , y + h + 20 ), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
                    cv2.putText(frame_draw,"ID: "+ str(face_id), (x , y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
                    cv2.putText(frame_draw,"Age: "+ str(int(predicted_ages[0])), (x , y +25), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)


                    self.totalFace=self.totalFace+1
            ret, jpeg = cv2.imencode('.jpg', frame_draw)
            # Record video
            # if self.is_record:
            #     if self.out == None:
            #         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #         self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))

            #     ret, frame = self.cap.read()
            #     if ret:
            #         self.out.write(frame)
            # else:
            #     if self.out != None:
            #         self.out.release()
            #         self.out = None

            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
