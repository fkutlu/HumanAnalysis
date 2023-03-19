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
import keras
totalFrame = 0

jsonData = collections.OrderedDict()
data_result = collections.OrderedDict()
ROI=False
Show=True


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


detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)#'rtsp://admin:Bullwark@172.21.39.126:554'  '01.mp4'

if ROI:
    createRoi.createRoi(cap)

with open('roi.json') as f:
    refPt = json.load(f)
print (refPt)

known_face_encodings = []


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (640,480))

face_locations = []
face_encodings = []
totalFace=0
while(cap.isOpened()):
    totalFrame += 1
    src = cap.read()[1]
    start_time = time.time()
    if ROI:
        frame=src[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    #frame = cv2.resize(src, (0, 0), fx=0.4, fy=0.4)
    else:
        frame = src.copy()

    frame_draw=frame.copy()
    rgb_frame = frame[:, :, ::-1]
    if totalFrame % 1 == 0:
        totalFrame=0
        res=detector.detect_emotions(frame)
        if len(res)>0:
            print("****************************************************************************")
            print(res)
            for i, face in enumerate(res):
                data_result["id"] = str(totalFace)

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
                #face["emotions"]["angry"]=str(face["emotions"]["angry"])
                #face["emotions"]["disgust"]=str(face["emotions"]["disgust"])
                #face["emotions"]["fear"]=str(face["emotions"]["fear"])
                #face["emotions"]["happy"]=str(face["emotions"]["happy"])
                #face["emotions"]["sad"]=str(face["emotions"]["sad"])
                #face["emotions"]["surprise"]=str(face["emotions"]["surprise"])
                #face["emotions"]["neutral"]=str(face["emotions"]["neutral"])


                #data_result["emotions"] = face["emotions"]
                #cv2.rectangle(frame_draw, (face['box'][0], face['box'][1]), (face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]), (255, 200, 0), 2)
                face_locations = []
                face_locations.append((face['box'][1], face['box'][0]+face['box'][2],face['box'][1]+face['box'][3],face['box'][0]))

                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                if len(known_face_encodings)>15:
                    known_face_encodings.pop(0)


                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                face_id=-1
                if True in matches:
                    face_id=matches.index(True)
                else:
                    known_face_encodings.append(face_encodings[0])


                face_imgs = np.empty((1, face_size, face_size, 3))
                face_img, cropped = crop_face(frame, face['box'], margin=40, size=face_size)
                (x, y, w, h) = face['box']

                face_imgs[0, :, :, :] = face_img

                results = model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                if predicted_genders[0][0] > 0.5:
                    print("Female")
                    color=(0, 0, 255)
                    data_result["Gender"]="Female"
                else:
                    print("Male")
                    color = (255, 0,0)
                    data_result["Gender"] = "Male"
                print("Age: ",int(predicted_ages[0]))
                data_result["Age"] = int(predicted_ages[0])

                if Show:
                    cv2.rectangle(frame_draw, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame_draw, emotion, (x , y + h + 20 ), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
                    cv2.putText(frame_draw,"ID: "+ str(face_id), (x , y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)
                    cv2.putText(frame_draw,"Age: "+ str(int(predicted_ages[0])), (x , y +25), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 1)


                data_result["DataTime"]=time.asctime(time.localtime(time.time()))
                jsonData["result"]=data_result

                with open('result.json', 'a+') as outfile:
                   json.dump(jsonData, outfile)
                #cv2.imshow(str(totalFace) + '.jpg', face_img.copy())
                #cv2.imwrite(str(totalFace)+'.jpg', face_img.copy())
                totalFace=totalFace+1
                #cv2.putText(frame_draw, str(int(predicted_ages[0])), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    #print("FPS: ", 1.0 / (time.time() - start_time))

    if Show:
        cv2.imshow('frame_draw', frame_draw)
        out.write(frame_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
