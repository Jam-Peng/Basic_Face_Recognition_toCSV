import cv2
import face_recognition


img_musk = face_recognition.load_image_file('Images/Musk.jpg')
img_musk = cv2.cvtColor(img_musk, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file('Images/test.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# 取得 musk臉部的座標
faceLocMusk = face_recognition.face_locations(img_musk)[0]
# 取得 musk臉部的128個編碼的測量值
encodeMusk = face_recognition.face_encodings(img_musk)[0]
# 繪製 musk臉部的矩形外框
cv2.rectangle(img_musk, (faceLocMusk[3], faceLocMusk[0]),
              (faceLocMusk[1], faceLocMusk[2]), (255, 0, 255), 2)

# 取得 test臉部的座標
faceLocTest = face_recognition.face_locations(img_test)[0]
# 取得 test臉部的128個編碼的測量值
encodeTest = face_recognition.face_encodings(img_test)[0]
# 繪製 test臉部的矩形外框
cv2.rectangle(img_test, (faceLocTest[3], faceLocTest[0]),
              (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# 比對兩個影像的臉部是否相同 (True or False)
results = face_recognition.compare_faces([encodeMusk], encodeTest)
# 比對兩個影像的臉部距離差值 (數值越接近 0 代表兩個臉部越相似)
faceDis = face_recognition.face_distance([encodeMusk], encodeTest)
# print(results, faceDis)

# 標示比對的結果和相似值
cv2.putText(img_test, f'{results} {round(faceDis[0], 2)}', (faceLocTest[3], faceLocTest[0]-10),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)


cv2.imshow('Musk', img_musk)
cv2.imshow('test', img_test)
cv2.waitKey(0)
