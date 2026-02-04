import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

export default function TradeTotalsChart({
  left,
  right,
  leftLabel,
  rightLabel,
  showLegend
}: {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
  leftLabel: string;
  rightLabel: string;
  showLegend: boolean;
}) {
  const leftPalette = ["#60a5fa", "#3b82f6", "#93c5fd", "#1d4ed8", "#bfdbfe"];
  const rightPalette = ["#fca5a5", "#f87171", "#fecaca", "#ef4444", "#fecdd3"];
  const legendEnabled = showLegend && left.length + right.length <= 6;

  const leftTraces = left.map((player, index) => ({
    type: "bar" as const,
    x: [leftLabel],
    y: [player.season_value],
    name: player.name,
    marker: { color: leftPalette[index % leftPalette.length] },
    text: [player.season_value.toFixed(1)],
    textposition: "auto" as const,
    textfont: { color: "#1e3a8a", size: 14, family: "Inter, system-ui, sans-serif" },
    hovertemplate: "%{y:.1f}<extra>%{fullData.name}</extra>"
  }));

  const rightTraces = right.map((player, index) => ({
    type: "bar" as const,
    x: [rightLabel],
    y: [player.season_value],
    name: player.name,
    marker: { color: rightPalette[index % rightPalette.length] },
    text: [player.season_value.toFixed(1)],
    textposition: "auto" as const,
    textfont: { color: "#7f1d1d", size: 14, family: "Inter, system-ui, sans-serif" },
    hovertemplate: "%{y:.1f}<extra>%{fullData.name}</extra>"
  }));

  // Stack player values to show each side's total composition.
  const data = [...leftTraces, ...rightTraces];

  return (
    <Plot
      data={data}
      layout={{
        title: "",
        barmode: "stack",
        margin: { t: 40, l: 40, r: 10, b: 80 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Squadre scambio", automargin: true },
        yaxis: {
          title: "Valore",
          tickmode: "linear",
          dtick: 10,
          tick0: 0
        },
        showlegend: legendEnabled,
        legend: { orientation: "h", y: 1.12, x: 0 }
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
