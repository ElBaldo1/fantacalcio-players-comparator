export type PlayerInput = {
  url: string;
  role: "GK" | "DEF" | "MID" | "FWD";
};

export type PlayerSeriesPoint = {
  giornata: number;
  fv: number | null;
  voto: number | null;
};

export type PlayerMetrics = {
  name: string;
  role: PlayerInput["role"];
  avg_voto: number;
  avg_fv: number;
  bonus_impact: number;
  std_fv: number;
  recent_trend: number;
  matches_n: number;
  availability: number;
  fv_unreliable: boolean;
  fv_equals_voto_ratio: number;
  final_score: number;
  predicted_next_fv: number;
  predicted_next3_fv: number;
  predicted_range60: [number, number];
  predicted_next3_range60: [number, number];
  p_big_game: number;
  p_bad_game: number;
  season_value: number;
  next3_value: number;
  components: Record<string, number>;
  series: PlayerSeriesPoint[];
};

export type VerdictBlock = {
  left: number;
  right: number;
  delta: number;
  verdict: string;
  drivers: { player: string; delta: number }[];
};

export type Totals = {
  season: VerdictBlock;
  next3: VerdictBlock;
};

export type TradeResponse = {
  left: PlayerMetrics[];
  right: PlayerMetrics[];
  totals: Totals;
};
