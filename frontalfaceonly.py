#basic computer vision inspired by pythorch
#we start by importing the cv2 library which we will use

import cv2

#the we load the cascades ,which can contain may things like face,eye,smile
#but we are going to use the front face since its what we are doing

face_cascades=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


#now we will start capturing using our pc webcam
#cv2 give a very good platform to capure 

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #so now we can load those faces we want to detect 

    faces=face_cascades.detectMultiScale(gray)

    #spacfy on rectalgle you what it to be on your face
    #in (255,0,0)is is a color which we should see on our rectacle on faces 
    #you can change to your desired color using those values eg (0,0,255) is another color
    #the 2 is tho thickness and thiness the more you increase you will bolder rectagle

    for (x,y,a,b) in faces:
        cv2.rectangle(frame,(x,y),(x+a,y+b),(255,0,0),2)

        #this imshow it will display the results ,like a new window for livestreaming will be opened
        # the ' frontal face detection ' it will be the frame title

        cv2.imshow('front face detection',frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        #the ('q') it a function to ensure it is open
        #closes when clicked close
cap.release()
cv2.destroyAllWindows()


#we have succefully done a computer vision program 
#follow on x @skmusyoka_
#see on github
#"keep bulding tech is not the future ,tech revolution is ongoing now please board"
#...........samson musyoka...............