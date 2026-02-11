import Plot from "react-plotly.js";
import type { PlayerMetrics } from "../../types";

export default function FVMComparisonChart({
  left,
  right
}: {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
}) {
  const allPlayers = [
    ...left.map((p) => ({ ...p, side: "S" as const })),
    ...right.map((p) => ({ ...p, side: "D" as const }))
  ];

  const labels = allPlayers.map((p) => `${p.name} (${p.side})`);
  const fvmValues = allPlayers.map((p) => p.fvm_classic ?? 0);
  const quotValues = allPlayers.map((p) => p.quotazione_classic ?? 0);

  const hasFvm = fvmValues.some((v) => v > 0);
  const hasQuot = quotValues.some((v) => v > 0);

  if (!hasFvm && !hasQuot) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        Dati FVM e Quotazione non disponibili
      </div>
    );
  }

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: labels,
          y: fvmValues,
          name: "FVM Classic",
          marker: { color: "#10b981" },
          text: fvmValues.map((v) => String(v)),
          textposition: "auto",
          textfont: { size: 12, family: "Inter, system-ui, sans-serif" },
          hovertemplate: "%{y}<extra>FVM</extra>"
        },
        {
          type: "bar",
          x: labels,
          y: quotValues,
          name: "Quotazione",
          marker: { color: "#f59e0b" },
          text: quotValues.map((v) => String(v)),
          textposition: "auto",
          textfont: { size: 12, family: "Inter, system-ui, sans-serif" },
          hovertemplate: "%{y}<extra>Quotazione</extra>"
        }
      ]}
      layout={{
        barmode: "group",
        margin: { t: 10, l: 40, r: 10, b: 100 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        legend: { orientation: "h", y: 1.08, x: 0 },
        xaxis: { tickangle: -25, automargin: true },
        yaxis: { title: "Valore", automargin: true }
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
