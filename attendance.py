import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 取得所有檔案中的照片＆取得照片的名稱當作 tag
path = 'Images'
images = []
classNames = []
if '.DS_Store' in os.listdir(path):
    myLists = os.listdir(path)[1:]
else:
    myLists = os.listdir(path)
# print(myLists)

for myList in myLists:
    curImg = cv2.imread(f'{path}/{myList}')
    images.append(curImg)                           # 取得所有檔案中的照片
    classNames.append(os.path.splitext(myList)[0])  # 取得照片的名稱


# 取得所有編碼測量值的影像
def findEncodings(images):
    encodeList = []
    for image in images:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


# 取的檔案中所有影像的編碼(128點測量值)
encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print("編碼檔案內所有影像的128點測量值完成")


# 標記登入到 CSV文件
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(mtDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            current_date = datetime.now()
            string_date = current_date.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{string_date}')


# 進行辨識
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    s_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    s_frame = cv2.cvtColor(s_frame, cv2.COLOR_BGR2RGB)

    # 取得辨識者臉部的座標
    faceCurrentFrame = face_recognition.face_locations(s_frame)
    # 取得辨識者臉部的128個編碼的測量值
    encodeCurrentFrame = face_recognition.face_encodings(
        s_frame, faceCurrentFrame)

    # 比較辨識者和檔案中的影像
    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)

        # 取得所有 faceDis 中最小的值就是與辨識者最相似的照片 - argmin 取最小索引值
        matchIndex = np.argmin(faceDis)

        # 取得比對為True的照片名稱
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)

            # 繪製辨識者臉部的矩形框
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4  # 因為前面將原始圖縮小為了加速比對，這裡要乘回等於1的倍數

            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1+35), (x2, y1),
                          (150, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y1+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # 將 tag標示在下方
            # cv2.rectangle(frame, (x1, y2-35), (x2, y2),
            #               (150, 255, 0), cv2.FILLED)
            # cv2.putText(frame, name, (x1+6, y2-6),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # 登入後紀錄到 csv文件中
            markAttendance(name)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(50) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
cv2.waitKey(1)
