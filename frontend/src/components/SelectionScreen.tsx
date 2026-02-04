import { ArrowRightLeft, Plus, Trash2 } from "lucide-react";
import type { PlayerInput } from "../types";

const roles = ["GK", "DEF", "MID", "FWD"] as const;

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
                <label className="text-xs uppercase text-slate-400">Player URL</label>
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
                <label className="text-xs uppercase text-slate-400">Role</label>
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
                    <option key={role} value={role}>
                      {role}
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
                  <Trash2 className="h-4 w-4" /> Remove
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
        <Plus className="h-4 w-4" /> Add player
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
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="pitch-bg border-b border-slate-900 px-8 py-6">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-400">
              Trade Analyzer
            </p>
            <h1 className="text-3xl font-semibold">Fanta-Analyst Pro</h1>
            <p className="text-sm text-slate-300">
              Build both sides of a trade and evaluate season & next-3 value.
            </p>
          </div>
          <button
            type="button"
            className="button-primary"
            disabled={loading}
            onClick={onEvaluate}
          >
            <ArrowRightLeft className="h-5 w-5" />
            {loading ? "Evaluating..." : "Evaluate trade"}
          </button>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 px-6 py-10 lg:grid-cols-2">
        <SidePanel
          label="Left Side"
          side="left"
          players={left}
          onAdd={onAdd}
          onRemove={onRemove}
          onChange={onChange}
        />
        <SidePanel
          label="Right Side"
          side="right"
          players={right}
          onAdd={onAdd}
          onRemove={onRemove}
          onChange={onChange}
        />
      </main>

      {errors.length > 0 && (
        <div className="mx-auto max-w-6xl px-6 pb-6">
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
