import cv2
import os
import uuid
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    os.makedirs("dataset")
except:
    pass

count = 0

while(True):
    ret, frame = cap.read()
    text = str(count)
    img = frame.copy()
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', img)

    keyin = cv2.waitKey(1) & 0xFF
    if keyin == ord('q'):
        break
    elif keyin == ord('s'):
        name = str(uuid.uuid1())
        cv2.imwrite("dataset/{}.jpg".format(name.split('-')[0]), frame)
        count += 1

cap.release()
cv2.destroyAllWindows()
