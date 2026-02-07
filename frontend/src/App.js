import React, { useEffect, useState } from "react";
import Webcam from "react-webcam";
import io from "socket.io-client";
import "./App.css";

const socket = io("http://backend:5000", {
  transports: ["websocket"],
});


function App() {
  const webcamRef = React.useRef(null);
  const [label, setLabel] = useState("Detecting..."); // Still used for the badge if needed, or can be removed
  const [confidence, setConfidence] = useState(0); // Still used for the badge if needed, or can be removed
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [displayedSubtitle, setDisplayedSubtitle] = useState("");
  const subtitleTimeoutRef = React.useRef(null);

  useEffect(() => {
    function onConnect() {
      setIsConnected(true);
      setDisplayedSubtitle(""); // Clear any subtitle on connect
      console.log("Connected to backend");
    }

    function onDisconnect() {
      setIsConnected(false);
      // Set the "Backend Not Reachable" message as a transient subtitle
      setDisplayedSubtitle("Backend Not Reachable");
      if (subtitleTimeoutRef.current) {
        clearTimeout(subtitleTimeoutRef.current);
      }
      subtitleTimeoutRef.current = setTimeout(() => {
        setDisplayedSubtitle("");
      }, 5000); // Display for 5 seconds

      setLabel("Backend Not Reachable"); // For the badge, if still used
      setConfidence(0); // For the badge, if still used
      console.log("Disconnected from backend");
    }

    function onPrediction(data) {
      setLabel(data.label);
      setConfidence(data.confidence);

      // Clear any existing timeout for subtitles
      if (subtitleTimeoutRef.current) {
        clearTimeout(subtitleTimeoutRef.current);
      }

      // Display new subtitle
      setDisplayedSubtitle(data.label);

      // Set timeout to clear subtitle after 2 seconds
      subtitleTimeoutRef.current = setTimeout(() => {
        setDisplayedSubtitle("");
      }, 2000);
    }

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("prediction", onPrediction);

    const interval = setInterval(() => {
      if (webcamRef.current && isConnected) { // Only send frames if connected
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          socket.emit("video_frame", imageSrc);
        }
      }
    }, 100); // Send frame every 100ms

    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("prediction", onPrediction);
      clearInterval(interval);
      if (subtitleTimeoutRef.current) {
        clearTimeout(subtitleTimeoutRef.current);
      }
    };
  }, [isConnected]); // Re-run effect if isConnected changes

  return (
    <div className="app">


      {/* Always display the transient subtitle, if it exists */}
      {displayedSubtitle && (
        <div className="subtitle-display">
          <h2>{displayedSubtitle}</h2>
        </div>
      )}

      {/* Only show connection error if not connected AND no temporary subtitle is active */}
      {!isConnected && !displayedSubtitle && (
        <div className="connection-error">
          <p>Backend Not Reachable. Please ensure the backend server is running.</p>
        </div>
      )}

      {isConnected && (
        <>
          <Webcam className="cam" ref={webcamRef} />
          {/* Keep the badge for continuous display, or remove if subtitles are primary */}
          <div className="badge">
            <h2>{label}</h2>
            <p>{(confidence * 100).toFixed(1)}%</p>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
