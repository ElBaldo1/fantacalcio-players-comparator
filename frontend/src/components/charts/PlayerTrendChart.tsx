import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

export default function PlayerTrendChart({ player }: { player: PlayerMetrics }) {
  const x = player.series.map((p) => p.giornata);
  const y = player.series.map((p) => p.fv);

  return (
    <Plot
      data={[
        {
          type: "scatter",
          mode: "lines+markers",
          x,
          y,
          name: player.name
        }
      ]}
      layout={{
        title: `FV Trend - ${player.name}`,
        margin: { t: 40, l: 50, r: 20, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        yaxis: {
          tickmode: "linear",
          dtick: 0.5,
          tick0: 0
        }
      }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
