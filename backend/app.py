from flask import Flask, request
from flask_socketio import SocketIO
import cv2, time, numpy as np, base64
from PIL import Image
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100 * 1024 * 1024)

# ---------------- CONFIG ----------------
FRAME_SKIP = 5
PREDICT_EVERY = 2
FEATURE_DIM = 318
sequence_length = 30
prediction_threshold = 0.1 # Temporarily lowered for debugging
PREDICT_INTERVAL = 10 # Predict every 10 frames

actions = ["HELLO_HI", "THANKS", "STOP"]

sequence = [] # Global sequence for LSTM input
last_emitted_label = None # Global variable to track the last emitted label
frame_counter = 0 # Global counter for frames to control prediction interval

# ---------------- LOAD MODEL ----------------
model = load_model("sign_lstm_best3.keras")

print("Mediapipe models are loading...")
# ---------------- MEDIAPIPE ----------------
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/hand_landmarker.task"
        ),
        num_hands=2
    )
)
print("Hand detector loaded.")
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/pose_landmarker_full.task"
        )
    )
)
print("Pose detector loaded.")
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="mediapipe_models/face_landmarker.task"
        )
    )
)
print("Face detector loaded.")

# ---------------- MAIN LOOP ----------------


@socketio.on("video_frame")
def handle_video_frame(data):
    global sequence
    # Decode base64 image
    try:
        # data will be like "data:image/webp;base64,..."
        # Extract the base64 part
        if "base64," in data:
            data = data.split("base64,")[1]
        
        # Add padding if necessary
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)

        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
        frame = np.array(img)
        # Convert RGB to BGR for OpenCV compatibility if needed
        if frame.shape[2] == 3: # Check if it's an RGB image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        print(f"Received frame shape: {frame.shape}") # Log frame dimensions

    except Exception as e:
        print(f"Error decoding image: {e}")
        return

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    try:
        hand = hand_detector.detect(mp_image)
    except Exception as e:
        print(f"Error detecting hands: {e}")
        hand = None # Set to None to avoid further errors

    try:
        pose = pose_detector.detect(mp_image)
    except Exception as e:
        print(f"Error detecting pose: {e}")
        pose = None # Set to None to avoid further errors

    try:
        face = face_detector.detect(mp_image)
    except Exception as e:
        print(f"Error detecting face: {e}")
        face = None # Set to None to avoid further errors

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
    
    global frame_counter
    frame_counter += 1

    sequence.append(keypoints)
    if len(sequence) > sequence_length:
        sequence.pop(0)

    if len(sequence) == sequence_length and frame_counter % PREDICT_INTERVAL == 0:
        print("Sequence full and prediction interval met, attempting prediction...")
        # Temporarily bypass model.predict for debugging
        dummy_prediction = np.array([0.9, 0.05, 0.05]) # Simulate a high confidence "HELLO_HI"
        prediction = dummy_prediction
        # prediction = model.predict(
        #     np.expand_dims(sequence, axis=0),
        #     verbose=0
        # )[0]
        print(f"Prediction raw output (dummy): {prediction}")

        idx = np.argmax(prediction)
        current_label = actions[idx]
        current_confidence = float(prediction[idx])

        global last_emitted_label
        if current_confidence >= prediction_threshold and current_label != last_emitted_label:
            socketio.emit("prediction", {
                "label": current_label,
                "confidence": current_confidence
            }, room=request.sid)
            last_emitted_label = current_label
            print(f"Emitted prediction: Label='{current_label}', Confidence={current_confidence:.2f}")
        else:
            print(f"Prediction not emitted: Label='{current_label}', Confidence={current_confidence:.2f}, Last Emitted='{last_emitted_label}' (Threshold: {prediction_threshold})")


@socketio.on("connect")
def on_connect():
    print("Client connected: ", request.sid)

# ---------------- START ----------------
if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000)
