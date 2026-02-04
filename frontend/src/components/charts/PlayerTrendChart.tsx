import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

type TrendProps = {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
  leftLabel: string;
  rightLabel: string;
  metric: "fv" | "voto";
  yTitle: string;
};

const buildAverageMap = (players: PlayerMetrics[], metric: "fv" | "voto") => {
  const map = new Map<number, { sum: number; count: number }>();
  players.forEach((player) => {
    player.series.forEach((point) => {
      const value = point[metric];
      if (typeof value !== "number" || Number.isNaN(value)) return;
      const entry = map.get(point.giornata) ?? { sum: 0, count: 0 };
      entry.sum += value;
      entry.count += 1;
      map.set(point.giornata, entry);
    });
  });
  return map;
};

export default function PlayerTrendChart({
  left,
  right,
  leftLabel,
  rightLabel,
  metric,
  yTitle
}: TrendProps) {
  const leftMap = buildAverageMap(left, metric);
  const rightMap = buildAverageMap(right, metric);
  const allDays = Array.from(
    new Set([...leftMap.keys(), ...rightMap.keys()])
  ).sort((a, b) => a - b);

  const leftY = allDays.map((day) => {
    const entry = leftMap.get(day);
    return entry ? entry.sum / entry.count : null;
  });
  const rightY = allDays.map((day) => {
    const entry = rightMap.get(day);
    return entry ? entry.sum / entry.count : null;
  });

  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "lines+markers",
          x: allDays,
          y: leftY,
          name: leftLabel,
          connectgaps: false,
          line: { color: "#2563eb", width: 3 },
          marker: { color: "#2563eb", size: 6 }
        },
        {
          type: "scatter",
          mode: "lines+markers",
          x: allDays,
          y: rightY,
          name: rightLabel,
          connectgaps: false,
          line: { color: "#dc2626", width: 3 },
          marker: { color: "#dc2626", size: 6 }
        }
      ]}
      layout={{
        title: "",
        margin: { t: 10, l: 40, r: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        legend: { orientation: "h", y: 1.05, x: 0 },
        xaxis: { title: "Giornata", tickmode: "linear", dtick: 1, automargin: true },
        yaxis: {
          title: yTitle,
          tickmode: "linear",
          dtick: 0.5,
          tick0: 0,
          automargin: true
        }
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
