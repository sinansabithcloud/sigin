from flask import Flask
from flask_socketio import SocketIO
import cv2, time, numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------- CONFIG ----------------
FRAME_SKIP = 5
PREDICT_EVERY = 2
FEATURE_DIM = 318
sequence_length = 30

actions = ["HELLO_HI", "THANKS", "STOP"]

# ---------------- LOAD MODEL ----------------
model = load_model("sign_lstm_best3.keras")

# ---------------- MEDIAPIPE ----------------
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/hand_landmarker.task"
        ),
        num_hands=2
    )
)

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/pose_landmarker_full.task"
        )
    )
)

face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/face_landmarker.task"
        )
    )
)

# ---------------- MAIN LOOP ----------------
def run_camera():
    cap = cv2.VideoCapture(0)
    sequence = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        hand = hand_detector.detect(mp_image)
        pose = pose_detector.detect(mp_image)
        face = face_detector.detect(mp_image)

        left_hand = np.zeros(63)
        right_hand = np.zeros(63)
        pose_kp = np.zeros(132)
        face_kp = np.zeros(60)

        if hand.hand_landmarks:
            for i, h in enumerate(hand.hand_landmarks[:2]):
                kp = np.array([[lm.x, lm.y, lm.z] for lm in h]).flatten()
                if i == 0:
                    left_hand = kp
                else:
                    right_hand = kp

        if pose.pose_landmarks:
            pose_kp = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in pose.pose_landmarks[0]]
            ).flatten()

        if face.face_landmarks:
            face_kp = np.array(
                [[face.face_landmarks[0][i].x,
                  face.face_landmarks[0][i].y,
                  face.face_landmarks[0][i].z]
                 for i in range(20)]
            ).flatten()

        keypoints = np.concatenate([left_hand, right_hand, pose_kp, face_kp])
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        sequence.append(keypoints)
        if len(sequence) > sequence_length:
            sequence.pop(0)

        if len(sequence) == sequence_length:
            prediction = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
            )[0]

            idx = np.argmax(prediction)
            socketio.emit("prediction", {
                "label": actions[idx],
                "confidence": float(prediction[idx])
            })

@socketio.on("connect")
def on_connect():
    print("Client connected")

# ---------------- START ----------------
if __name__ == "__main__":
    socketio.start_background_task(run_camera)
    socketio.run(app, host="0.0.0.0", port=5000)
