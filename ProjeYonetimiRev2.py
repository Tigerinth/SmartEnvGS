'''''
Hareket ve Isik Tespit Uygulamasi + esp32
Tiger
'''''


import cv2
import numpy as np
import time
import datetime
import serial
import threading
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4 #guven thresholdu
model.classes = [0]  #insan

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, first_frame = cap.read()

grayframe = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

tempVal = 0.0
humdVal = 0.0
LDRVal = 0
LDRText = "Dusuk"
lightsAction = False

filename = datetime.datetime.now().strftime("%d_%m_%Y_%H%M") + "_LOG.txt"

threshold = 100
motiondetected = False
lightdetected = False
print("baslatiliyor...")
time.sleep(0.5)
#fps count
start = time.time()


hourVal = datetime.datetime.now().hour
print("saat degeri: "+ str(hourVal))

def read_serial():
    global tempVal, humdVal, lightsAction,LDRVal,LDRText
    last_sent = time.time()

    with serial.Serial('COM5', 115200, timeout=1) as ser:
        while True:
            if ser.in_waiting:
                data = ser.read_until(b'/')
                decodedData = data.decode(errors='ignore').strip()

                if decodedData.startswith("TEMP"):
                    try:
                        tempVal = float(decodedData[5:-1])
                        print("temp =", tempVal)
                    except ValueError:
                        pass
                elif decodedData.startswith("HUMD"):
                    try:
                        humdVal = float(decodedData[5:-1])
                        print("humd =", humdVal)
                    except ValueError:
                        pass
                elif decodedData.startswith("LDRL"):
                    try:
                        LDRVal = int(decodedData[5:-1])

                        if LDRVal >= 1800:
                            LDRText = "Yuksek"
                        elif LDRVal < 1800:
                            LDRText = "Dusuk"
                        print("ldr =", LDRVal)
                    except ValueError:
                        pass

            now = time.time()
            if now - last_sent >= 1.0:
                if lightsAction:
                    ser.write("1".encode())
                else:
                    ser.write("0".encode())

                print("led: " + str(lightsAction))
                with open(filename, "a") as file:
                    file.write(f"[{str(datetime.datetime.now().strftime("%d_%m_%Y_%H%M"))}] -> Nem: {str(humdVal)}, Sicaklik: {str(tempVal)}, isikSensor: {str(LDRVal)}, aydinlatmaDurum: {str(lightsAction)}\n")
                last_sent = now


thread = threading.Thread(target=read_serial, daemon=True)
thread.start()

while True:
    ret, frame = cap.read()

    results = model(frame)


    cv2.putText(frame, "Akilli Ortam Sistemi", (390, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 1)

    ####################
    #   Isik Tespiti   #
    ####################
    '''
    #ton doyma deger
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_light = np.array([0, 0, 245])
    upper_light = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_light, upper_light)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    '''
    if hourVal <= 6 or hourVal >= 19:
        cv2.putText(frame, "isik tespit aktif", (475, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        '''
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 100, 150), 2)
            lightdetected = True
        else:
            lightdetected = False
        '''
    else:
        cv2.putText(frame, "isik tespit deaktif", (475, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)


    ########################
    #   Hareket Tespiti    #
    ########################

    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(grayframe, gray)
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 15000:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(frame, 'Hareket', (x, y-8), cv2.FONT_HERSHEY_PLAIN, 1, (100, 50, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150,150), 2)
            motiondetected = True
        else:
            motiondetected = False
    '''
    motiondetected = False
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Insan {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        motiondetected = True

    #print(motiondetected)

    actions = []
    actioncount = 0
    if humdVal >= 60.:
        actions.append("ortami Havalandir")
    if tempVal >= 25.:
        actions.append("isiticiyi Kapat")
    cv2.putText(frame, 'Eylem Listesi', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 1)

    if motiondetected is False:
        actioncount = actioncount + 1
        actions.append("isiklar sonduluyor...")
        actions.append("bilgisayari uyku moduna al")
        #actions.append("musluklari kapat")
        lightsAction = False

    i = 0
    for action in actions:
            i = i+1
            cv2.putText(frame, action, (15, 40 + i*27), cv2.FONT_HERSHEY_PLAIN, 1, (200, 50, 0), 2)

    end = time.time()
    fps = 1 / (end - start)
    start = end


    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.putText(frame, "Nem   :" + str(humdVal) + "%", (475, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 1)
    cv2.putText(frame, "Sicaklik:" + str(tempVal) + "C", (475, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 1)
    cv2.putText(frame, "isik    :" + LDRText, (475, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 1)


    cv2.imshow('Motion & Light & BT + ESP32', frame)
    #cv2.imshow('Motion & Light Detection and Functions', frame)
    #cv2.imshow('Light Detection Visualization', res)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        print("isik")
        lightsAction = not lightsAction

cap.release()
cv2.destroyAllWindows()
