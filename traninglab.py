import cv2
import cv2.data

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces=face_cascade.detectMultiScale(gray)

    for (x,y,z,h) in faces:
        gray=gray[y:y+h,x:x+z]
        eyes=eye_cascade.detectMultiScale(gray)

        for (a,b,c,d) in eyes:
            cv2.rectangle(frame,(x+a,y+b),(x+a+c,y+b+d),(0,255,0),2)
            smiles = smile_cascade.detectMultiScale(gray,1.8,20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)

    cv2.imshow('Smile Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()