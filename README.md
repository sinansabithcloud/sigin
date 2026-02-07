# Signkit-prototype: Continuous Indian Sign Language Translation

This project is a prototype for real-time Continuous Indian Sign Language (ISL) to English translation. It consists of a Python Flask backend with Socket.IO for sign language recognition and a React frontend for displaying the translations.

## Architecture Overview

The system operates with the following data flow:
1.  **Frontend (React App - `frontend/src/App.js`):** This is the main component. It captures video frames from the user's webcam.
2.  **Frontend to Backend:** Sends these video frames in real-time to the backend via Socket.IO.
3.  **Backend (Flask Server):** Receives the video frames, processes them using MediaPipe to extract key landmarks (hands, pose, face), and then feeds these landmarks into a pre-trained Keras LSTM model for sign language prediction.
4.  **Backend to Frontend:** Emits the predicted English translation (label and confidence score) back to the specific frontend client via Socket.IO.
5.  **Frontend Display:** Displays the received English translation to the user.

## Getting Started

Follow these instructions to set up and run the project.

### Prerequisites

*   **Python 3.8+:** For the backend.
*   **pip:** Python package installer.
*   **Node.js (LTS recommended):** For the frontend.
*   **npm (comes with Node.js):** Node.js package manager.
*   **Webcam:** A physical webcam is required on the machine running the frontend.

### Backend Setup (Python)

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download MediaPipe Model Assets:**
    The backend requires MediaPipe's task models. Create a directory named `mediapipe_models` inside the `backend` directory and download the following `.task` files into it:
    *   `hand_landmarker.task`
    *   `pose_landmarker_full.task`
    *   `face_landmarker.task`
    You can typically find these models in the MediaPipe documentation or examples.

6.  **Place the Keras Model:**
    Ensure your pre-trained Keras LSTM model (`sign_lstm_best3.keras`) is located in the `backend` directory.

7.  **Run the Backend Server:**
    With the virtual environment activated, run:
    ```bash
    python app.py
    ```
    The backend server will start on `http://0.0.0.0:5000`.

### Frontend Setup (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Run the Frontend Development Server:**
    ```bash
    npm start
    ```
    The frontend application will typically open in your browser at `http://localhost:3000`.
## Usage

1.  Ensure both the backend server and the frontend development server are running.
2.  Open your browser to `http://localhost:3000`.
3.  Allow camera access when prompted by your browser.
4.  Perform Indian Sign Language gestures in front of your webcam.
5.  The translated English words and confidence scores will appear on the frontend. If the backend server is not running or the connection is lost, the frontend will display a "Backend Not Reachable" message.

## Important Notes

*   **Model Files:** The `.gitignore` file includes `sign_lstm_best3.keras` to prevent it from being committed to Git due to its size. Make sure you have this file in your `backend` directory to run the application.
*   **MediaPipe Assets:** The `mediapipe_models` directory with its `.task` files is crucial for the backend's landmark detection.
*   **Performance:** Real-time performance may vary depending on your hardware and network conditions.
*   **CORS:** The Flask backend is configured with `cors_allowed_origins="*"` for development purposes. For production, restrict this to your frontend's domain.
