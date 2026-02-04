import { ArrowLeft, ChartLine } from "lucide-react";
import { useMemo, useState } from "react";
import type { TradeResponse } from "../types";
import TradeTotalsChart from "./charts/TradeTotalsChart";
import PlayerValueChart from "./charts/PlayerValueChart";
import PlayerTrendChart from "./charts/PlayerTrendChart";

const roleLabels: Record<string, string> = {
  GK: "Portiere",
  DEF: "Difensore",
  MID: "Centrocampista",
  FWD: "Attaccante"
};

function Table({ players }: { players: TradeResponse["left"] }) {
  return (
    <div className="card overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-slate-100 text-left">
            {[
              "Giocatore",
              "Ruolo",
              "Punteggio",
              "Valore Stagione",
              "Valore Prossime 3",
              "FV Previsto",
              "FV Previsto (3)",
              "Affidabilità",
              "Disponibilità",
              "FV Medio",
              "Std FV"
            ].map((head) => (
              <th key={head} className="table-head">
                {head}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={player.name} className="border-t border-slate-200">
              <td className="table-cell">{player.name}</td>
              <td className="table-cell">{roleLabels[player.role] ?? player.role}</td>
              <td className="table-cell">{player.final_score.toFixed(1)}</td>
              <td className="table-cell">{player.season_value.toFixed(1)}</td>
              <td className="table-cell">{player.next3_value.toFixed(1)}</td>
              <td className="table-cell">{player.predicted_next_fv.toFixed(2)}</td>
              <td className="table-cell">{player.predicted_next3_fv.toFixed(2)}</td>
              <td className="table-cell">{player.matches_n >= 15 ? "Alta" : player.matches_n >= 10 ? "Media" : "Bassa"}</td>
              <td className="table-cell">{player.availability.toFixed(2)}</td>
              <td className="table-cell">{player.avg_fv.toFixed(2)}</td>
              <td className="table-cell">{player.std_fv.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VerdictBlock({ title, data }: { title: string; data: TradeResponse["totals"]["season"] }) {
  return (
    <div className="card p-4">
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="text-sm text-slate-600">{data.verdict}</p>
      <p className="text-sm text-slate-500">Differenza: {data.delta.toFixed(1)}</p>
      <div className="mt-2 text-xs text-slate-500">
        Fattori chiave: {data.drivers.map((d) => `${d.player} (${d.delta.toFixed(1)})`).join(", ") || "N/D"}
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
  const allPlayers = useMemo(() => [...data.left, ...data.right], [data]);
  const [selectedPlayer, setSelectedPlayer] = useState(allPlayers[0]?.name ?? "");
  const selected = allPlayers.find((p) => p.name === selectedPlayer) ?? allPlayers[0];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 px-4 py-5 md:px-8">
        <div className="mx-auto flex max-w-screen-2xl items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-600">Risultati</p>
            <h1 className="text-3xl font-semibold">Analisi scambio</h1>
          </div>
          <button type="button" className="button-secondary" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" /> Indietro
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-screen-2xl space-y-8 px-4 py-8 md:px-8">
        <section className="grid gap-6 lg:grid-cols-2">
          <div>
            <h2 className="mb-3 text-lg font-semibold">Lato sinistro</h2>
            <Table players={data.left} />
          </div>
          <div>
            <h2 className="mb-3 text-lg font-semibold">Lato destro</h2>
            <Table players={data.right} />
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-3">
          <div className="card p-4">
            <p className="text-sm text-slate-500">Totali stagione</p>
            <p className="text-xl font-semibold">Sinistra {data.totals.season.left.toFixed(1)}</p>
            <p className="text-xl font-semibold">Destra {data.totals.season.right.toFixed(1)}</p>
            <p className="text-sm text-slate-500">Differenza {data.totals.season.delta.toFixed(1)}</p>
          </div>
          <div className="card p-4">
            <p className="text-sm text-slate-500">Totali prossime 3</p>
            <p className="text-xl font-semibold">Sinistra {data.totals.next3.left.toFixed(1)}</p>
            <p className="text-xl font-semibold">Destra {data.totals.next3.right.toFixed(1)}</p>
            <p className="text-sm text-slate-500">Differenza {data.totals.next3.delta.toFixed(1)}</p>
          </div>
          <div className="card p-4">
            <p className="text-sm text-slate-500">Sintesi</p>
            <p className="text-sm">Verdetto stagione: {data.totals.season.verdict}</p>
            <p className="text-sm">Verdetto prossime 3: {data.totals.next3.verdict}</p>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="card p-4 h-[360px]">
            <TradeTotalsChart totals={data.totals} />
          </div>
          <div className="card p-4 h-[360px]">
            <PlayerValueChart
              left={data.left}
              right={data.right}
              valueKey="season_value"
              title="Valore stagione per giocatore"
            />
          </div>
          <div className="card p-4 h-[360px]">
            <PlayerValueChart
              left={data.left}
              right={data.right}
              valueKey="next3_value"
              title="Valore prossime 3 per giocatore"
            />
          </div>
          <div className="card p-4 h-[360px]">
            <div className="mb-2 flex items-center gap-2 text-sm text-slate-500">
              <ChartLine className="h-4 w-4 text-emerald-600" />
              Andamento FV
            </div>
            <select
              className="select mb-3"
              value={selectedPlayer}
              onChange={(event) => setSelectedPlayer(event.target.value)}
            >
              {allPlayers.map((player) => (
                <option key={player.name} value={player.name}>
                  {player.name}
                </option>
              ))}
            </select>
            {selected && <PlayerTrendChart player={selected} />}
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <VerdictBlock title="Verdetto breve periodo (Prossime 3)" data={data.totals.next3} />
          <VerdictBlock title="Verdetto lungo periodo (Stagione)" data={data.totals.season} />
        </section>
      </main>
    </div>
  );
}
