from ultralytics import YOLO # import the YOLO module for loading the YOLO model.
import cv2  #OpenCV, a library used for image and video processing.
import cvzone # is an extension of OpenCV that simplifies some tasks like adding text or shapes.
import math #provides mathematical functions.
from playsound import playsound  #playsound is a library to play sound files.
import os  #os provides functions for interacting with the operating system.
import winsound  #winsound allows playing sound beeps on Windows systems.


def beep():
    try:
        # Replace these values with your desired frequency and duration
        winsound.Beep(440, 1000)  # Frequency (Hz), Duration (milliseconds)
    except Exception as e:
        print(f"Error playing beep sound: {e}")


def ppe_detection(file): 
    if file is None: 
        cap = cv2.VideoCapture(0)  # For Webcam
        cap.set(3, 1280)
        cap.set(4, 720)
    else: 
        cap = cv2.VideoCapture(file)  # For Video
    model = YOLO("best.pt")

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)
    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.5:
                    if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0, 255)  # blue
                        beep()
                    elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                        myColor = (0, 255, 0)  # green
                    else:
                        myColor = (255, 0, 0)  # red
                        beep()  # Trigger the beep sound

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                       colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #file = r"F:\Computer_vision\PPE_detection_YOLO\Videos\ppe-1.mp4"
    file = r"F:\Computer_vision\PPE_detection_YOLO\Videos\ppe-2.mp4"
    ppe_detection(file)

