import React from "react";

export default function PredictionBadge({ label, confidence }) {
  return (
    <div className="prediction-badge">
      <h2>{label}</h2>
      {confidence && <p>Confidence: {(confidence * 100).toFixed(1)}%</p>}
    </div>
  );
}
