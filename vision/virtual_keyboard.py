import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=.8, maxHands=2)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/", " "]]
finalText = ""

keyboard = Controller()

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (button.pos[0], button.pos[1],
                                                   button.size[0],button.size[0]), 20 ,rt=0)
        cv2.rectangle(img, button.pos, (x+w, y+h), (255,144,30), cv2.FILLED)
        cv2.putText(img, button.text, (x+20, y+80), 
        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
    return img


class Button():
    def __init__(self, pos, text, size=[80,80]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([100*j+50, 100*i+50], key))
        

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    img = drawAll(img, buttonList)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # list of 21 landmarks
        bbox1 = hand1["bbox"] # bounding box info x,y,w,h

        if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)


        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList1[8][0] < x+w and y < lmList1[8][1] < y+h:
                cv2.rectangle(img, button.pos, (x+w, y+h), (175,0,175), cv2.FILLED)
                cv2.putText(img, button.text, (x+20,y+65), 
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            
                l, _, _ = detector.findDistance(lmList1[8], lmList1[4], img)
                print(l)

                if l<30:
                    #keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x+w, y+h), (0,255,0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20,y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText += button.text
                    sleep(0.53)
                     
    cv2.rectangle(img, (50, 350), (700, 450), (175,0,175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), 
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv2.imshow("keyboard", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break


cap.release()
cv2.destroyAllWindows()
