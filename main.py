import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

good_reps=0
bad_reps=0
model = joblib.load("pushup_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)


prediction_buffer = deque(maxlen=5)

counter = 0
stage = "up"

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    feedback = "NO PERSON"
    color = (0, 0, 255)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        s = [landmarks[11].x, landmarks[11].y]
        e = [landmarks[13].x, landmarks[13].y]
        wr = [landmarks[15].x, landmarks[15].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]

      
        arm_angle = calculate_angle(s, e, wr)
        body_angle = calculate_angle(s, hip, knee)
        hip_offset = s[1] - hip[1]


        is_horizontal = abs(s[1] - ankle[1]) < 0.50
        hands_below_shoulder = wr[1] > s[1]
        body_straight = body_angle > 120

        if not (is_horizontal and hands_below_shoulder and body_straight):
            feedback = "GET INTO PUSHUP POSITION"
            color = (0, 0, 255)
            prediction_buffer.clear()  

        else:
#ml
            features = [[arm_angle, body_angle, hip_offset]]
            prediction = model.predict(features)[0]

            prediction_buffer.append(prediction)

            if sum(prediction_buffer) > len(prediction_buffer)//2:
                final_prediction = 1
            else:
                final_prediction = 0

#feedback
            if body_angle < 140:
                feedback = "KEEP BODY STRAIGHT"
                speak("keep body straight")

            elif arm_angle > 130 and final_prediction == 0:
                feedback = "GO LOWER"
                speak("go lower")

            elif final_prediction == 0:
                feedback = "CHECK FORM"
                speak("check form")

            else:
                feedback = "GOOD FORM"

#reps counting
            if arm_angle > 150:
                stage = "up"

            if arm_angle < 110 and stage == "up":
                stage = "down"
                counter += 1

                if final_prediction == 1:
                    good_reps += 1
                    speak("Good rep")
                else:
                    bad_reps += 1
                    speak("Fix your form")

    cv2.rectangle(image, (0, 0), (w, 70), (0, 0, 0), -1)

    
    if feedback == "GOOD FORM":
        text_color = (0, 255, 0) 
    elif feedback == "GET INTO PUSHUP POSITION":
        text_color = (0, 0, 255) 
    else:
        text_color = (0, 165, 255) 

    cv2.putText(image, f"PUSHUPS | {feedback}", 
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.putText(image, f"REPS: {counter}", 
                (w//2 - 80, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("ML Pushup Trainer", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
