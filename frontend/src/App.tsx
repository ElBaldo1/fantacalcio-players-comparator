import { useMemo, useState } from "react";
import SelectionScreen from "./components/SelectionScreen";
import ResultsScreen from "./components/ResultsScreen";
import type { PlayerInput, TradeResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const emptyPlayer = (): PlayerInput => ({ url: "", role: "MID" });

const countRoles = (players: PlayerInput[]) => {
  const counts = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  players.forEach((p) => {
    if (counts[p.role] !== undefined) {
      counts[p.role] += 1;
    }
  });
  return counts;
};

const roleLabels: Record<PlayerInput["role"], string> = {
  GK: "Portiere",
  DEF: "Difensore",
  MID: "Centrocampista",
  FWD: "Attaccante"
};

const formatRoleCounts = (counts: Record<PlayerInput["role"], number>) =>
  `Portiere ${counts.GK}, Difensore ${counts.DEF}, Centrocampista ${counts.MID}, Attaccante ${counts.FWD}`;

export default function App() {
  const [left, setLeft] = useState<PlayerInput[]>([emptyPlayer()]);
  const [right, setRight] = useState<PlayerInput[]>([emptyPlayer()]);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [results, setResults] = useState<TradeResponse | null>(null);

  const validations = useMemo(() => {
    const errs: string[] = [];
    if (left.length !== right.length) {
      errs.push(
        `Serve lo stesso numero di giocatori per lato. Ora: sinistra ${left.length}, destra ${right.length}.`
      );
    }
    const leftCounts = countRoles(left);
    const rightCounts = countRoles(right);
    if (JSON.stringify(leftCounts) !== JSON.stringify(rightCounts)) {
      errs.push(
        `I ruoli devono essere identici tra i due lati. ` +
          `Sinistra: ${formatRoleCounts(leftCounts)}. ` +
          `Destra: ${formatRoleCounts(rightCounts)}.`
      );
    }
    const missingUrls = [...left, ...right].some((p) => !p.url.trim());
    if (missingUrls) {
      errs.push("Manca almeno un URL: inserisci il link per ogni giocatore.");
    }
    return errs;
  }, [left, right]);

  const onAdd = (side: "left" | "right") => {
    if (side === "left") {
      setLeft((prev) => [...prev, emptyPlayer()]);
    } else {
      setRight((prev) => [...prev, emptyPlayer()]);
    }
  };

  const onRemove = (side: "left" | "right", index: number) => {
    if (side === "left") {
      setLeft((prev) => prev.filter((_, i) => i !== index));
    } else {
      setRight((prev) => prev.filter((_, i) => i !== index));
    }
  };

  const onChange = (side: "left" | "right", index: number, next: PlayerInput) => {
    if (side === "left") {
      setLeft((prev) => prev.map((p, i) => (i === index ? next : p)));
    } else {
      setRight((prev) => prev.map((p, i) => (i === index ? next : p)));
    }
  };

  const onEvaluate = async () => {
    setErrors([]);
    if (validations.length > 0) {
      setErrors(validations);
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/trade/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ left, right })
      });
      if (!response.ok) {
        let message = "Impossibile valutare lo scambio.";
        const contentType = response.headers.get("content-type") ?? "";
        if (contentType.includes("application/json")) {
          const payload = await response.json();
          if (payload?.detail) {
            message = payload.detail;
          }
        } else {
          const text = await response.text();
          if (text) message = text;
        }
        throw new Error(message);
      }
      const payload = (await response.json()) as TradeResponse;
      setResults(payload);
    } catch (error) {
      setErrors([error instanceof Error ? error.message : "Errore imprevisto"]);
    } finally {
      setLoading(false);
    }
  };

  if (results) {
    return (
      <ResultsScreen
        data={results}
        onBack={() => setResults(null)}
      />
    );
  }

  return (
    <SelectionScreen
      left={left}
      right={right}
      errors={errors}
      loading={loading}
      onAdd={onAdd}
      onRemove={onRemove}
      onChange={onChange}
      onEvaluate={onEvaluate}
    />
  );
}
