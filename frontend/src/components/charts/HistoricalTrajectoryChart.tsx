import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

export default function HistoricalTrajectoryChart({
  left,
  right,
  leftLabel,
  rightLabel
}: {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
  leftLabel: string;
  rightLabel: string;
}) {
  const maxSeasons = Math.max(
    ...left.map((p) => p.yoy_avg_fv.length),
    ...right.map((p) => p.yoy_avg_fv.length)
  );

  if (maxSeasons < 2) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        Dati storici non disponibili
      </div>
    );
  }

  const seasonLabels = ["Stagione attuale", "Stagione -1", "Stagione -2"].slice(0, maxSeasons);

  const avg = (players: PlayerMetrics[], idx: number) => {
    const vals = players
      .map((p) => p.yoy_avg_fv[idx])
      .filter((v) => v != null && !isNaN(v));
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const leftAvgs = seasonLabels.map((_, i) => avg(left, i));
  const rightAvgs = seasonLabels.map((_, i) => avg(right, i));

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: seasonLabels,
          y: leftAvgs,
          name: leftLabel,
          marker: { color: "#2563eb" },
          text: leftAvgs.map((v) => (v != null ? v.toFixed(2) : "")),
          textposition: "auto",
          textfont: { size: 12, family: "Inter, system-ui, sans-serif" },
          hovertemplate: "%{y:.2f}<extra>%{fullData.name}</extra>"
        },
        {
          type: "bar",
          x: seasonLabels,
          y: rightAvgs,
          name: rightLabel,
          marker: { color: "#dc2626" },
          text: rightAvgs.map((v) => (v != null ? v.toFixed(2) : "")),
          textposition: "auto",
          textfont: { size: 12, family: "Inter, system-ui, sans-serif" },
          hovertemplate: "%{y:.2f}<extra>%{fullData.name}</extra>"
        }
      ]}
      layout={{
        barmode: "group",
        margin: { t: 10, l: 40, r: 10, b: 60 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        legend: { orientation: "h", y: 1.08, x: 0 },
        yaxis: {
          title: "FV medio",
          automargin: true,
          tickmode: "linear",
          dtick: 0.5
        }
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
