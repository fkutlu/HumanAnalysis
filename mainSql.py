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
import pymysql
import datetime
db = pymysql.connect("localhost","root","12345678","emotion" )
cursor = db.cursor()

totalFrame = 0

ROI=False
Show=False
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

WRN_WEIGHTS_PATH = "./pretrained_models/weights.18-4.06.hdf5"

face_size = 64
depth=16
width=8
model = WideResNet(face_size, depth=depth, k=width)()
model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
fpath = get_file('weights.18-4.06.hdf5',
                 WRN_WEIGHTS_PATH,
                 cache_subdir=model_dir)

model.load_weights(fpath)


detector = FER()
cap = cv2.VideoCapture('rtsp://admin:Bullwark@172.21.39.126:554')#'rtsp://admin:Bullwark@195.87.215.0:554'

if ROI:
    createRoi.createRoi(cap)

with open('roi.json') as f:
    refPt = json.load(f)
print (refPt)

known_face_encodings = []

face_locations = []
face_encodings = []

while(cap.isOpened()):
    totalFrame += 1
    src = cap.read()[1]
    start_time = time.time()
    frame=src[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    #frame = cv2.resize(src, (0, 0), fx=0.4, fy=0.4)
    #frame = src.copy()

    frame_draw=frame.copy()
    rgb_frame = frame[:, :, ::-1]
    if totalFrame % 1 == 0:
        totalFrame=0
        res=detector.detect_emotions(frame)
        if len(res)>0:
            print("****************************************************************************")
            print(res)
            for i, face in enumerate(res):

                angry=face["emotions"]["angry"]
                disgust=face["emotions"]["disgust"]
                fear=face["emotions"]["fear"]
                happy=face["emotions"]["happy"]
                sad=face["emotions"]["sad"]
                surprise=face["emotions"]["surprise"]
                neutral=face["emotions"]["neutral"]

                if Show:
                    cv2.rectangle(frame_draw, (face['box'][0], face['box'][1]), (face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]), (255, 200, 0), 2)
                face_locations = []
                face_locations.append((face['box'][1], face['box'][0]+face['box'][2],face['box'][1]+face['box'][3],face['box'][0]))

                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                if len(known_face_encodings)>5:
                    known_face_encodings.pop(0)

                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])

                if True in matches:
                    continue

                else:
                    known_face_encodings.append(face_encodings[0])

                face_imgs = np.empty((1, face_size, face_size, 3))
                face_img, cropped = crop_face(frame, face['box'], margin=40, size=face_size)
                (x, y, w, h) = cropped
                if Show:
                    cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (255, 255, 0), 2)
                face_imgs[0, :, :, :] = face_img

                results = model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                if predicted_genders[0][0] > 0.5:
                    print("Female")
                    gender=0
                else:
                    print("Male")
                    gender = 1
                print("Age: ",int(predicted_ages[0]))
                age = int(predicted_ages[0])


                sql = """INSERT INTO detect(Cam_ID,angry,disgust,fear,happy,sad,surprise,neutral,age,gender,time) VALUES ('%d','%f','%f','%f','%f','%f','%f','%f','%d','%d','%s')""" % \
                      (0, angry, disgust, fear, happy, sad, surprise, neutral, age, gender,
                       datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))

                cursor.execute(sql)

                db.commit()
                if Show:
                    cv2.imshow('detect', face_img.copy())
                    #cv2.imwrite(str(totalFace)+'.jpg', face_img.copy())
                    #cv2.putText(frame_draw, str(int(predicted_ages[0])), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    print("FPS: ", 1.0 / (time.time() - start_time))

    if Show:
        cv2.imshow('frame_draw', frame_draw)
        cv2.waitKey(1)
db.close()
