import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils
ptime = 0
ctime = 0
run = True
while run:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks :
        for handlms in result.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y *h)
                cv2.circle(img, (cx, cy), 10,(255,100,100), cv2.FILLED)
            mpdraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime 

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,
            (255,0,255),3)
    cv2.imshow("HAND DETECTOR", img)
    key = cv2.waitKey(1)
    if key == ord(" "):
        break


