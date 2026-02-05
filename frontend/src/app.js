import React, { useEffect, useState } from "react";
import Webcam from "react-webcam";
import io from "socket.io-client";
import "./App.css";

const socket = io("http://localhost:5000");

function App() {
  const [label, setLabel] = useState("Detecting...");
  const [confidence, setConfidence] = useState(0);

  useEffect(() => {
    socket.on("prediction", data => {
      setLabel(data.label);
      setConfidence(data.confidence);
    });
  }, []);

  return (
    <div className="app">
      <h1>🤟 Sign Language Recognition</h1>

      <Webcam className="cam" />

      <div className="badge">
        <h2>{label}</h2>
        <p>{(confidence * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
}

export default App;
