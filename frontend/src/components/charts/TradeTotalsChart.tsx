import Plot from "react-plotly.js";
import type { Totals } from "../../types";

export default function TradeTotalsChart({ totals }: { totals: Totals }) {
  const data = [
    {
      type: "bar",
      x: ["Sinistra", "Destra"],
      y: [totals.season.left, totals.season.right],
      name: "Stagione"
    },
    {
      type: "bar",
      x: ["Sinistra", "Destra"],
      y: [totals.next3.left, totals.next3.right],
      name: "Prossime 3"
    }
  ];

  return (
    <Plot
      data={data}
      layout={{
        title: "Totali valore scambio",
        barmode: "group",
        margin: { t: 40, l: 50, r: 20, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "Lato" },
        yaxis: {
          title: "Valore",
          tickmode: "linear",
          dtick: 10,
          tick0: 0
        }
      }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
