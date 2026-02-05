import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import pyttsx3 # for text-to-speech
import time
from collections import deque

FRAME_SKIP = 5    # process 1 in every 5 frames
PREDICT_EVERY = 2  # predict every 2 processed frames
FEATURE_DIM = 318

FACE_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
]
sequence = []
prediction_buffer = []
cooldown = 20
cooldown_counter = 0
last_spoken_label = None

DISPLAY_THRESHOLD = 0.60
SPEECH_MIN_PROB = 0.82 # stricter than display threshold

# Actions (must match training labels order)
actions = ["HELLO_HI", "THANKS", "STOP"]
sequence_length = 30

# --- Load your trained model ---
MODEL_PATH = "sign_lstm_best3.keras"  # adjust to your trained model path
model = load_model(MODEL_PATH)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)  # speech rate
engine.setProperty("volume", 1.0)  # volume (0.0 to 1.0)
# ---------- CONFIDENCE & SMOOTHING ----------
SMOOTHING_WINDOW = 1  # last 10 predictions for smoothing
CONFIDENCE_THRESHOLD = 0.85   # IMPORTANT
confidence_gap_threshold = 0.20  # IMPORTANT
# ---------- COOLDOWN ----------
COOLDOWN_SECONDS = 1.5 


# --- Load MediaPipe Task models ---
hand_model = "D:/PATIENT_SIGN_DATA/hand_landmarker.task"
pose_model = "D:/PATIENT_SIGN_DATA/pose_landmarker_full.task"
face_model = "D:/PATIENT_SIGN_DATA/face_landmarker.task"

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=pose_model),
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model),
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# --- Webcam ---
cap = cv2.VideoCapture(0)

sequence = []   # buffer of frames
current_label = "Detecting..."
avg_prediction = np.zeros(len(actions))  # SAFE INIT
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    hand_result = hand_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    # ---------- KEYPOINT EXTRACTION ----------
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    pose = np.zeros(33 * 4)
    face = np.zeros(len(FACE_IDX) * 3)

    if hand_result.hand_landmarks:
        for idx, hand in enumerate(hand_result.hand_landmarks[:2]):
            kp = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
            if idx == 0:
                left_hand = kp
            else:
                right_hand = kp

    if pose_result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in pose_result.pose_landmarks[0]]
        ).flatten()

    if face_result.face_landmarks:
        face = np.array(
            [[face_result.face_landmarks[0][i].x,
              face_result.face_landmarks[0][i].y,
              face_result.face_landmarks[0][i].z]
             for i in FACE_IDX]
        ).flatten()

    keypoints = np.concatenate([left_hand, right_hand, pose, face])

    if keypoints.shape[0] != FEATURE_DIM:
        continue
    frame_count += 1

    # ---------- FRAME SKIP (performance only) ----------
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Sign Recognition", frame)
        continue

    # ---------- SEQUENCE BUFFER ----------
    sequence.append(keypoints)
    if len(sequence) > sequence_length:
        sequence.pop(0)

     # ---------- PREDICTION ----------
    if len(sequence) == sequence_length and frame_count % (FRAME_SKIP * PREDICT_EVERY) == 0:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]
    # Temporal Smoothing
        #prediction_buffer.append(prediction)
        #if len(prediction_buffer) > 0:
         #  avg_prediction = np.mean(np.array(prediction_buffer), axis=0)
        #else:
         #  avg_prediction = np.zeros(len(actions))
         # NO smoothing for demo safety
        avg_prediction = prediction
    # Get top prediction
        action_idx = np.argmax(avg_prediction)
        action_prob = avg_prediction[action_idx]
        action_label = actions[action_idx]
        # ---------- DOMINANCE FILTER ----------
        sorted_probs = np.sort(avg_prediction)
        confidence_gap = sorted_probs[-1] - sorted_probs[-2]
        
        if action_prob > 0.70 and confidence_gap > 0.18:
            current_label = f"{action_label} ({action_prob:.2f})"
        else:
            current_label = "STOP"
        if (
            action_prob >= DISPLAY_THRESHOLD and
            confidence_gap >= confidence_gap_threshold and
            cooldown_counter == 0
        ):

            if action_label != last_spoken_label:
               pyttsx3.speak(action_label)
               last_spoken_label = action_label
               cooldown_counter = cooldown
        current_time = time.time()

        if (
            action_prob > 0.85 and
            action_label != last_spoken_label and
            current_time - last_spoken_time > 2.0
        ):
            engine.say(action_label)
            engine.runAndWait()
            last_spoken_label = action_label
            last_spoken_time = current_time
    
        # ---------- COOLDOWN ----------
    if cooldown_counter > 0:
       cooldown_counter -= 1
       
    cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        current_label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
   ) 

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

hand_detector.close()
pose_detector.close()
face_detector.close()

cap.release()
cv2.destroyAllWindows()