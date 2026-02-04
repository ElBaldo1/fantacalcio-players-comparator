import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

type Metrics = {
  avg_voto: number | null;
  avg_fv: number | null;
  bonus_impact: number | null;
  consistency: number | null;
  recent_trend: number | null;
  matches_n: number;
};

type Player = {
  name: string;
  metrics: Metrics;
  final_score: number | null;
  components: Record<string, number | null>;
  matches: { giornata: number; fv: number }[];
};

type CompareResponse = {
  player1: Player;
  player2: Player;
};

const formatNumber = (
  value: number | null | undefined,
  digits = 2,
  signed = false
) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
};

const useVerdict = (p1?: Player, p2?: Player) => {
  return useMemo(() => {
    if (!p1 || !p2 || p1.final_score === null || p2.final_score === null) {
      return { title: "Verdict", detail: "Not enough data to decide." };
    }
    const diff = p1.final_score - p2.final_score;
    const winner = diff > 0 ? p1.name : p2.name;
    const status = Math.abs(diff) < 1 ? "Tie" : "Winner";

    const componentDiffs = Object.keys(p1.components).map((key) => {
      const v1 = p1.components[key] ?? 0;
      const v2 = p2.components[key] ?? 0;
      return { key, value: v1 - v2 };
    });
    componentDiffs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const top = componentDiffs.slice(0, 3);
    const explanation = top
      .map((item) => `${item.key} (${item.value >= 0 ? "+" : ""}${item.value.toFixed(1)})`)
      .join(", ");

    const title =
      status === "Tie"
        ? `Verdict: Tie (${diff.toFixed(1)})`
        : `Verdict: ${winner} by ${Math.abs(diff).toFixed(1)}`;

    return { title, detail: `Key differences: ${explanation}` };
  }, [p1, p2]);
};

export default function App() {
  const [url1, setUrl1] = useState("");
  const [url2, setUrl2] = useState("");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const verdict = useVerdict(data?.player1, data?.player2);

  const onCompare = async () => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url1, url2 })
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed");
      }
      const payload = (await response.json()) as CompareResponse;
      setData(payload);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const lineChartData = useMemo(() => {
    if (!data) {
      return [];
    }
    const p1 = data.player1;
    const p2 = data.player2;
    return [
      {
        x: p1.matches.map((m) => m.giornata),
        y: p1.matches.map((m) => m.fv),
        type: "scatter",
        mode: "lines+markers",
        name: p1.name
      },
      {
        x: p2.matches.map((m) => m.giornata),
        y: p2.matches.map((m) => m.fv),
        type: "scatter",
        mode: "lines+markers",
        name: p2.name
      }
    ];
  }, [data]);

  const radarChartData = useMemo(() => {
    if (!data) {
      return [];
    }
    const categories = [
      "Avg Voto",
      "Bonus Impact",
      "Consistency",
      "Recent Trend",
      "Goal Involvement"
    ];
    const valuesFor = (player: Player) =>
      categories.map((key) => player.components[key] ?? 50);

    return [
      {
        type: "scatterpolar",
        r: [...valuesFor(data.player1), valuesFor(data.player1)[0]],
        theta: [...categories, categories[0]],
        fill: "toself",
        name: data.player1.name
      },
      {
        type: "scatterpolar",
        r: [...valuesFor(data.player2), valuesFor(data.player2)[0]],
        theta: [...categories, categories[0]],
        fill: "toself",
        name: data.player2.name
      }
    ];
  }, [data]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Fanta-Analyst Pro</h1>
          <p>Compare two Fantacalcio players with advanced scoring and trends.</p>
        </div>
        <div className="inputs">
          <label>
            Player 1 URL
            <input
              value={url1}
              onChange={(event) => setUrl1(event.target.value)}
              placeholder="https://www.fantacalcio.it/..."
            />
          </label>
          <label>
            Player 2 URL
            <input
              value={url2}
              onChange={(event) => setUrl2(event.target.value)}
              placeholder="https://www.fantacalcio.it/..."
            />
          </label>
          <button onClick={onCompare} disabled={loading || !url1 || !url2}>
            {loading ? "Analyzing..." : "Compare"}
          </button>
        </div>
      </header>

      {error && <div className="alert error">{error}</div>}

      {data && (
        <>
          <section className="grid">
            {[data.player1, data.player2].map((player) => (
              <div key={player.name} className="card">
                <div className="card-header">
                  <h2>{player.name}</h2>
                  <div className="score">
                    <span>Final Score</span>
                    <strong>{formatNumber(player.final_score, 1)}</strong>
                  </div>
                </div>
                <div className="metrics">
                  <div>
                    <span>Avg Voto</span>
                    <strong>{formatNumber(player.metrics.avg_voto, 2)}</strong>
                  </div>
                  <div>
                    <span>Avg FV</span>
                    <strong>{formatNumber(player.metrics.avg_fv, 2)}</strong>
                  </div>
                  <div>
                    <span>Bonus Impact</span>
                    <strong>{formatNumber(player.metrics.bonus_impact, 2, true)}</strong>
                  </div>
                  <div>
                    <span>Consistency</span>
                    <strong>{formatNumber(player.metrics.consistency, 2)}</strong>
                  </div>
                  <div>
                    <span>Recent Trend</span>
                    <strong>{formatNumber(player.metrics.recent_trend, 2, true)}</strong>
                  </div>
                  <div>
                    <span>Matches</span>
                    <strong>{player.metrics.matches_n}</strong>
                  </div>
                </div>
              </div>
            ))}
          </section>

          <section className="charts">
            <div className="chart-card">
              <h3>FV Trend by Matchday</h3>
              <Plot
                data={lineChartData}
                layout={{
                  autosize: true,
                  margin: { t: 30, l: 40, r: 20, b: 40 },
                  xaxis: { title: "Giornata" },
                  yaxis: { title: "FV" }
                }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
            <div className="chart-card">
              <h3>Radar Comparison</h3>
              <Plot
                data={radarChartData}
                layout={{
                  autosize: true,
                  margin: { t: 30, l: 40, r: 40, b: 30 },
                  polar: { radialaxis: { visible: true, range: [0, 100] } },
                  showlegend: true
                }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
          </section>

          <section className="verdict">
            <h3>{verdict.title}</h3>
            <p>{verdict.detail}</p>
          </section>
        </>
      )}

      {!data && !loading && !error && (
        <div className="empty">
          Provide two Fantacalcio player URLs to start the comparison.
        </div>
      )}
    </div>
  );
}
