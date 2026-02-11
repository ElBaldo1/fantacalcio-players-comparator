import { ArrowLeft, ChartLine, Eye, EyeOff } from "lucide-react";
import { useMemo, useState } from "react";
import type { TradeResponse } from "../types";
import TradeTotalsChart from "./charts/TradeTotalsChart";
import PlayerTrendChart from "./charts/PlayerTrendChart";
import HistoricalTrajectoryChart from "./charts/HistoricalTrajectoryChart";
import FVMComparisonChart from "./charts/FVMComparisonChart";

const roleLabels: Record<string, string> = {
  GK: "Portiere",
  DEF: "Difensore",
  MID: "Centrocampista",
  FWD: "Attaccante"
};

function TrajectoryBadge({ direction, modifier }: { direction: string; modifier: number }) {
  const config: Record<string, { label: string; color: string; arrow: string }> = {
    improving: {
      label: "In crescita",
      color: "text-emerald-700 bg-emerald-50 border-emerald-200",
      arrow: "\u2197"
    },
    declining: {
      label: "In calo",
      color: "text-rose-700 bg-rose-50 border-rose-200",
      arrow: "\u2198"
    },
    stable: {
      label: "Stabile",
      color: "text-slate-600 bg-slate-50 border-slate-200",
      arrow: "\u2192"
    }
  };
  const c = config[direction] ?? config.stable;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-semibold ${c.color}`}
    >
      {c.arrow} {c.label} ({modifier > 0 ? "+" : ""}
      {modifier.toFixed(1)})
    </span>
  );
}

function Table({ players, side }: { players: TradeResponse["left"]; side: "left" | "right" }) {
  const headers = [
    { label: "Giocatore", className: "whitespace-nowrap" },
    { label: "Ruolo", className: "whitespace-nowrap" },
    { label: "Punteggio", className: "whitespace-nowrap" },
    { label: "Valore Stagione", className: "whitespace-nowrap" },
    { label: "Voto medio", className: "whitespace-nowrap" },
    { label: "FV medio", className: "whitespace-nowrap" },
    { label: "Quotaz.", className: "whitespace-nowrap" },
    { label: "FVM", className: "whitespace-nowrap" },
    { label: "Effic. Valore", className: "whitespace-nowrap" },
    { label: "Copertura (%)", className: "whitespace-nowrap" },
    { label: "Std FV", className: "whitespace-nowrap" },
    { label: "Traiettoria", className: "whitespace-nowrap" }
  ];
  const badgeClass = side === "left" ? "badge-left" : "badge-right";
  const badgeLabel = side === "left" ? "Sinistra" : "Destra";
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-[1000px] w-full">
          <thead>
            <tr className="bg-slate-100 text-left">
              {headers.map((head) => (
                <th key={head.label} className={`table-head ${head.className}`}>
                  {head.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {players.map((player) => {
              const fmt = (v: number | null | undefined, d = 2) =>
                v != null ? v.toFixed(d) : "N/D";
              return (
                <tr key={player.name} className="border-t border-slate-200">
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <span className={badgeClass}>{badgeLabel}</span>
                      <span>{player.name}</span>
                    </div>
                  </td>
                  <td className="table-cell">{roleLabels[player.role] ?? player.role}</td>
                  <td className="table-cell">{fmt(player.final_score, 1)}</td>
                  <td className="table-cell">{fmt(player.season_value, 1)}</td>
                  <td className="table-cell">{fmt(player.avg_voto)}</td>
                  <td className="table-cell">{fmt(player.avg_fv)}</td>
                  <td className="table-cell">
                    {player.quotazione_classic != null ? player.quotazione_classic : "N/D"}
                  </td>
                  <td className="table-cell font-semibold">
                    {player.fvm_classic != null ? player.fvm_classic : "N/D"}
                  </td>
                  <td className="table-cell">{fmt(player.value_efficiency)}</td>
                  <td className="table-cell">
                    {player.availability != null
                      ? `${(player.availability * 100).toFixed(0)}%`
                      : "N/D"}
                  </td>
                  <td className="table-cell">{fmt(player.std_fv)}</td>
                  <td className="table-cell">
                    <TrajectoryBadge
                      direction={player.trajectory_direction ?? "stable"}
                      modifier={player.trajectory_modifier ?? 0}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function VerdictBlock({ title, data }: { title: string; data: TradeResponse["totals"]["season"] }) {
  return (
    <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 shadow-sm">
      <h3 className="text-lg font-semibold text-emerald-900">{title}</h3>
      <p className="text-sm text-emerald-900">{data.verdict}</p>
      <p className="text-sm text-emerald-700">Differenza: {data.delta.toFixed(1)}</p>
      <div className="mt-2 text-xs text-emerald-700">
        Fattori chiave:{" "}
        {data.drivers.map((d) => `${d.player} (${d.delta.toFixed(1)})`).join(", ") || "N/D"}
      </div>
    </div>
  );
}

export default function ResultsScreen({
  data,
  onBack
}: {
  data: TradeResponse;
  onBack: () => void;
}) {
  const [legendOpen, setLegendOpen] = useState(true);
  const leftLabel = useMemo(() => {
    if (data.left.length === 1) return data.left[0].name;
    if (data.left.length === 2) return `${data.left[0].name} + ${data.left[1].name}`;
    return `${data.left[0].name} + ${data.left[1].name} + ${data.left.length - 2} altri`;
  }, [data.left]);
  const rightLabel = useMemo(() => {
    if (data.right.length === 1) return data.right[0].name;
    if (data.right.length === 2) return `${data.right[0].name} + ${data.right[1].name}`;
    return `${data.right[0].name} + ${data.right[1].name} + ${data.right.length - 2} altri`;
  }, [data.right]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 px-4 py-4 md:px-8">
        <div className="mx-auto flex max-w-screen-2xl items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-600">Risultati</p>
            <h1 className="text-4xl font-semibold">Analisi scambio</h1>
          </div>
          <button type="button" className="button-secondary" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" /> Indietro
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-screen-2xl space-y-5 px-4 py-5 md:px-8">
        <section className="grid gap-6 lg:grid-cols-2">
          <div>
            <h2 className="mb-3 text-xl font-semibold">{leftLabel}</h2>
            <Table players={data.left} side="left" />
          </div>
          <div>
            <h2 className="mb-3 text-xl font-semibold">{rightLabel}</h2>
            <Table players={data.right} side="right" />
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[2fr_1fr]">
          <div className="card flex flex-col gap-3 p-4">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm text-slate-500">Valore stagione</p>
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-100"
                  onClick={() => setLegendOpen((prev) => !prev)}
                >
                  {legendOpen ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  {legendOpen ? "Nascondi legenda" : "Mostra legenda"}
                </button>
              </div>
              <div className="mt-2 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-sm text-slate-500">{leftLabel}</p>
                  <p className="text-2xl font-semibold">{data.totals.season.left.toFixed(1)}</p>
                  <p className="text-sm text-slate-500">
                    Media voto{" "}
                    {(
                      data.left.reduce((acc, p) => acc + p.avg_voto, 0) / data.left.length
                    ).toFixed(2)}{" "}
                    · Media FV{" "}
                    {(
                      data.left.reduce((acc, p) => acc + p.avg_fv, 0) / data.left.length
                    ).toFixed(2)}{" "}
                    · FVM tot.{" "}
                    {data.left.reduce((acc, p) => acc + (p.fvm_classic ?? 0), 0)}{" "}
                    · Quotaz. tot.{" "}
                    {data.left.reduce((acc, p) => acc + (p.quotazione_classic ?? 0), 0)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-500">{rightLabel}</p>
                  <p className="text-2xl font-semibold">{data.totals.season.right.toFixed(1)}</p>
                  <p className="text-sm text-slate-500">
                    Media voto{" "}
                    {(
                      data.right.reduce((acc, p) => acc + p.avg_voto, 0) / data.right.length
                    ).toFixed(2)}{" "}
                    · Media FV{" "}
                    {(
                      data.right.reduce((acc, p) => acc + p.avg_fv, 0) / data.right.length
                    ).toFixed(2)}{" "}
                    · FVM tot.{" "}
                    {data.right.reduce((acc, p) => acc + (p.fvm_classic ?? 0), 0)}{" "}
                    · Quotaz. tot.{" "}
                    {data.right.reduce((acc, p) => acc + (p.quotazione_classic ?? 0), 0)}
                  </p>
                </div>
              </div>
              <p className="mt-2 text-sm text-slate-500">
                Differenza {data.totals.season.delta.toFixed(1)}
              </p>
            </div>
            <div className="flex-1 min-h-[220px]">
              <TradeTotalsChart
                left={data.left}
                right={data.right}
                leftLabel={leftLabel}
                rightLabel={rightLabel}
                showLegend={legendOpen}
              />
            </div>
          </div>
          <VerdictBlock title="Verdetto stagione" data={data.totals.season} />
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="card flex h-[320px] flex-col p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-500">
              <ChartLine className="h-4 w-4 text-emerald-600" />
              Andamento FV (media per lato)
            </div>
            <div className="flex-1 min-h-0">
              <PlayerTrendChart
                left={data.left}
                right={data.right}
                leftLabel={leftLabel}
                rightLabel={rightLabel}
                metric="fv"
                yTitle="Fantavoto (FV)"
              />
            </div>
          </div>
          <div className="card flex h-[320px] flex-col p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-500">
              <ChartLine className="h-4 w-4 text-emerald-600" />
              Andamento Voto (media per lato)
            </div>
            <div className="flex-1 min-h-0">
              <PlayerTrendChart
                left={data.left}
                right={data.right}
                leftLabel={leftLabel}
                rightLabel={rightLabel}
                metric="voto"
                yTitle="Voto"
              />
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="card flex h-[320px] flex-col p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-500">
              <ChartLine className="h-4 w-4 text-emerald-600" />
              Traiettoria storica — FV medio per stagione
            </div>
            <div className="flex-1 min-h-0">
              <HistoricalTrajectoryChart
                left={data.left}
                right={data.right}
                leftLabel={leftLabel}
                rightLabel={rightLabel}
              />
            </div>
          </div>
          <div className="card flex h-[320px] flex-col p-4">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-500">
              <ChartLine className="h-4 w-4 text-emerald-600" />
              Confronto FVM e Quotazione
            </div>
            <div className="flex-1 min-h-0">
              <FVMComparisonChart left={data.left} right={data.right} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
