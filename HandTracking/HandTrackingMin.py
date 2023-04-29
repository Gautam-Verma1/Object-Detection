import cv2
import mediapipe as mp
import time

# creating video object
cap = cv2.VideoCapture(0) # 1 indicates -> using webcam number 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #

while True:
    success, img = cap.read()

    # send RGB image to this object
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RBG bcz 'hands' only uses RGB
    results = hands.process(imgRGB) # process the frames fro us and give results

    # use results to extract the parameters within
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # extract info of each hand

            for id, lm, in enumerate(handLms.landmark): # fetch the id and lm of all landmarks
                h, w, c = img.shape # obtain width and height
                cx, cy = int(lm.x * w), int(lm.y * h) # obtain the dimension of lm in the frame
                print(id, cx, cy)
                if id==0: # for a particular landmark, here 0
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # draw circle on it

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)