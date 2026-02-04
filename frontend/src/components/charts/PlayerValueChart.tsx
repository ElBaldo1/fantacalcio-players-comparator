import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

export default function PlayerValueChart({
  left,
  right,
  valueKey,
  title
}: {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
  valueKey: "season_value" | "next3_value";
  title: string;
}) {
  const labels = [...left.map((p) => `${p.name} (S)`), ...right.map((p) => `${p.name} (D)`)];
  const values = [...left.map((p) => p[valueKey]), ...right.map((p) => p[valueKey])];

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: labels,
          y: values,
          marker: {
            color: labels.map((label) => (label.includes("(S)") ? "#1d4ed8" : "#dc2626"))
          }
        }
      ]}
      layout={{
        title,
        margin: { t: 20, l: 50, r: 20, b: 120 },
        xaxis: { tickangle: -30, title: "Giocatore" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        yaxis: {
          title: "Valore",
          tickmode: "linear",
          dtick: 10,
          tick0: 0
        }
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
