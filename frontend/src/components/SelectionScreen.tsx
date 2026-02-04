import { ArrowRightLeft, Plus, Shield, Trash2 } from "lucide-react";
import type { PlayerInput } from "../types";

const roles = [
  { value: "GK", label: "Portiere" },
  { value: "DEF", label: "Difensore" },
  { value: "MID", label: "Centrocampista" },
  { value: "FWD", label: "Attaccante" }
] as const;

type SideProps = {
  label: string;
  side: "left" | "right";
  players: PlayerInput[];
  onAdd: (side: "left" | "right") => void;
  onRemove: (side: "left" | "right", index: number) => void;
  onChange: (side: "left" | "right", index: number, next: PlayerInput) => void;
};

function SidePanel({ label, side, players, onAdd, onRemove, onChange }: SideProps) {
  return (
    <div className="card p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">{label}</h2>
        <span className={side === "left" ? "badge-left" : "badge-right"}>{label}</span>
      </div>
      <div className="space-y-4">
        {players.map((player, index) => (
          <div key={`${side}-${index}`} className="rounded-xl border border-slate-800 p-3">
            <div className="grid gap-3 md:grid-cols-5">
              <div className="md:col-span-3">
                <label className="text-xs uppercase text-slate-500">URL giocatore</label>
                <input
                  className="input mt-1"
                  value={player.url}
                  onChange={(event) =>
                    onChange(side, index, { ...player, url: event.target.value })
                  }
                  placeholder="https://www.fantacalcio.it/..."
                />
              </div>
              <div>
                <label className="text-xs uppercase text-slate-500">Ruolo</label>
                <select
                  className="select mt-1"
                  value={player.role}
                  onChange={(event) =>
                    onChange(side, index, {
                      ...player,
                      role: event.target.value as PlayerInput["role"]
                    })
                  }
                >
                  {roles.map((role) => (
                    <option key={role.value} value={role.value}>
                      {role.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-end">
                <button
                  type="button"
                  className="button-secondary w-full"
                  onClick={() => onRemove(side, index)}
                >
                  <Trash2 className="h-4 w-4" /> Rimuovi
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      <button
        type="button"
        className="button-secondary mt-4 w-full"
        onClick={() => onAdd(side)}
      >
        <Plus className="h-4 w-4" /> Aggiungi giocatore
      </button>
    </div>
  );
}

type SelectionProps = {
  left: PlayerInput[];
  right: PlayerInput[];
  errors: string[];
  loading: boolean;
  onAdd: (side: "left" | "right") => void;
  onRemove: (side: "left" | "right", index: number) => void;
  onChange: (side: "left" | "right", index: number, next: PlayerInput) => void;
  onEvaluate: () => void;
};

export default function SelectionScreen({
  left,
  right,
  errors,
  loading,
  onAdd,
  onRemove,
  onChange,
  onEvaluate
}: SelectionProps) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="pitch-bg border-b border-slate-200 px-4 py-5 md:px-8">
        <div className="mx-auto flex max-w-screen-2xl flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 text-emerald-600">
              <Shield className="h-6 w-6" />
              <p className="text-xs uppercase tracking-[0.3em]">Analisi scambi</p>
            </div>
            <h1 className="text-3xl font-semibold text-slate-900">Fanta-Analyst Pro</h1>
            <p className="text-sm text-slate-600">
              Costruisci i due lati dello scambio e confronta valore stagione e prossime 3.
            </p>
          </div>
          <button
            type="button"
            className="button-primary"
            disabled={loading}
            onClick={onEvaluate}
          >
            <ArrowRightLeft className="h-5 w-5" />
            {loading ? "Valutazione in corso..." : "Valuta scambio"}
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-screen-2xl gap-6 px-4 py-8 md:px-8 lg:grid-cols-2">
        <SidePanel
          label="Lato sinistro"
          side="left"
          players={left}
          onAdd={onAdd}
          onRemove={onRemove}
          onChange={onChange}
        />
        <SidePanel
          label="Lato destro"
          side="right"
          players={right}
          onAdd={onAdd}
          onRemove={onRemove}
          onChange={onChange}
        />
      </main>

      {errors.length > 0 && (
        <div className="mx-auto max-w-screen-2xl px-4 pb-6 md:px-8">
          <div className="alert space-y-1">
            {errors.map((error, index) => (
              <p key={`err-${index}`}>{error}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
