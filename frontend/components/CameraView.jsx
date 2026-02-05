import React from "react";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

export default function CameraView() {
  return (
    <div className="camera-box">
      <Webcam
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
      />
    </div>
  );
}
