import { ArrowLeft } from "lucide-react";
import type { TradeResponse } from "../types";
import TradeTotalsChart from "./charts/TradeTotalsChart";
import PlayerValueChart from "./charts/PlayerValueChart";
import PlayerTrendChart from "./charts/PlayerTrendChart";

function Table({ players }: { players: TradeResponse["left"] }) {
  return (
    <div className="card overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-slate-900/80 text-left">
            {[
              "Player",
              "Role",
              "Final",
              "Season",
              "Next3",
              "Pred FV",
              "Pred Next3",
              "Conf",
              "Avail",
              "Avg FV",
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
            <tr key={player.name} className="border-t border-slate-800/80">
              <td className="table-cell">{player.name}</td>
              <td className="table-cell">{player.role}</td>
              <td className="table-cell">{player.final_score.toFixed(1)}</td>
              <td className="table-cell">{player.season_value.toFixed(1)}</td>
              <td className="table-cell">{player.next3_value.toFixed(1)}</td>
              <td className="table-cell">{player.predicted_next_fv.toFixed(2)}</td>
              <td className="table-cell">{player.predicted_next3_fv.toFixed(2)}</td>
              <td className="table-cell">{player.matches_n >= 15 ? "High" : player.matches_n >= 10 ? "Medium" : "Low"}</td>
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
      <p className="text-sm text-slate-300">{data.verdict}</p>
      <p className="text-sm text-slate-400">Delta: {data.delta.toFixed(1)}</p>
      <div className="mt-2 text-xs text-slate-400">
        Key drivers: {data.drivers.map((d) => `${d.player} (${d.delta.toFixed(1)})`).join(", ") || "N/A"}
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
  const allPlayers = [...data.left, ...data.right];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-900 px-6 py-6">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-400">Results</p>
            <h1 className="text-3xl font-semibold">Trade Analysis</h1>
          </div>
          <button type="button" className="button-secondary" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" /> Back
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-8 px-6 py-10">
        <section className="grid gap-6 lg:grid-cols-2">
          <div>
            <h2 className="mb-3 text-lg font-semibold">Left Side</h2>
            <Table players={data.left} />
          </div>
          <div>
            <h2 className="mb-3 text-lg font-semibold">Right Side</h2>
            <Table players={data.right} />
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-3">
          <div className="card p-4">
            <p className="text-sm text-slate-400">Season Totals</p>
            <p className="text-xl font-semibold">Left {data.totals.season.left.toFixed(1)}</p>
            <p className="text-xl font-semibold">Right {data.totals.season.right.toFixed(1)}</p>
            <p className="text-sm text-slate-400">Delta {data.totals.season.delta.toFixed(1)}</p>
          </div>
          <div className="card p-4">
            <p className="text-sm text-slate-400">Next3 Totals</p>
            <p className="text-xl font-semibold">Left {data.totals.next3.left.toFixed(1)}</p>
            <p className="text-xl font-semibold">Right {data.totals.next3.right.toFixed(1)}</p>
            <p className="text-sm text-slate-400">Delta {data.totals.next3.delta.toFixed(1)}</p>
          </div>
          <div className="card p-4">
            <p className="text-sm text-slate-400">Snapshot</p>
            <p className="text-sm">Season verdict: {data.totals.season.verdict}</p>
            <p className="text-sm">Next3 verdict: {data.totals.next3.verdict}</p>
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
              title="Season Value by Player"
            />
          </div>
          <div className="card p-4 h-[360px]">
            <PlayerValueChart
              left={data.left}
              right={data.right}
              valueKey="next3_value"
              title="Next3 Value by Player"
            />
          </div>
          <div className="card p-4 h-[360px]">
            <PlayerTrendChart player={allPlayers[0]} />
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <VerdictBlock title="Short-Term Verdict (Next 3)" data={data.totals.next3} />
          <VerdictBlock title="Long-Term Verdict (Season)" data={data.totals.season} />
        </section>
      </main>
    </div>
  );
}
