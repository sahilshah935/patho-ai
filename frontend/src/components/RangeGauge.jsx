import React from "react";

/**
 * Visual Biological Reference Range Gauge
 * Visualizes where the patient's biomarker value falls relative to reference bounds.
 */
export default function RangeGauge({ value, min, max, status }) {
  const numVal = parseFloat(value);
  const numMin = parseFloat(min);
  const numMax = parseFloat(max);

  if (isNaN(numVal) || isNaN(numMin) || isNaN(numMax) || numMax <= numMin) {
    return (
      <span className="text-xs text-slate-400 font-mono">
        {min && max ? `${min} – ${max}` : "—"}
      </span>
    );
  }

  // Calculate dynamic display scale with 25% margin on both sides
  const rangeSpan = numMax - numMin;
  const scaleMin = Math.max(0, numMin - rangeSpan * 0.35);
  const scaleMax = numMax + rangeSpan * 0.35;
  const totalSpan = scaleMax - scaleMin;

  // Percentage positions
  const normalStartPct = Math.max(0, Math.min(100, ((numMin - scaleMin) / totalSpan) * 100));
  const normalWidthPct = Math.max(5, Math.min(100 - normalStartPct, ((numMax - numMin) / totalSpan) * 100));
  
  // Marker position clamped between 3% and 97% for visibility
  const rawMarkerPct = ((numVal - scaleMin) / totalSpan) * 100;
  const markerPct = Math.max(3, Math.min(97, rawMarkerPct));

  // Determine dot color according to clinical status
  const normStatus = (status || "").toUpperCase();
  let dotColor = "#059669"; // Emerald Normal
  let dotRing = "rgba(5, 150, 105, 0.25)";

  if (normStatus.includes("CRITICAL")) {
    dotColor = "#dc2626"; // Crimson Critical
    dotRing = "rgba(220, 38, 38, 0.3)";
  } else if (normStatus.includes("HIGH") || normStatus.includes("ELEVATED")) {
    dotColor = "#d97706"; // Amber High
    dotRing = "rgba(217, 119, 6, 0.3)";
  } else if (normStatus.includes("LOW")) {
    dotColor = "#0284c7"; // Cerulean Low
    dotRing = "rgba(2, 132, 199, 0.3)";
  }

  return (
    <div className="range-gauge-container" title={`Result: ${numVal} | Reference: ${numMin} – ${numMax}`}>
      <div className="range-gauge-bar">
        {/* Full scale track */}
        <div className="range-gauge-track" />
        
        {/* Normal target zone */}
        <div
          className="range-gauge-normal-zone"
          style={{
            left: `${normalStartPct}%`,
            width: `${normalWidthPct}%`,
          }}
        />

        {/* Patient value marker */}
        <div
          className="range-gauge-marker"
          style={{
            left: `${markerPct}%`,
            backgroundColor: dotColor,
            boxShadow: `0 0 0 3px ${dotRing}, 0 2px 4px rgba(0,0,0,0.15)`,
          }}
        />
      </div>

      <div className="range-gauge-labels">
        <span className="range-label">{numMin}</span>
        <span className="range-label-center">Ref Interval</span>
        <span className="range-label">{numMax}</span>
      </div>
    </div>
  );
}
