/** Response types mirroring the FastAPI Pydantic schemas. */

export interface InstrumentSummary {
  symbol: string;
  name: string;
  sector: string | null;
  industry: string | null;
  exchange: string | null;
  market_cap: number | null;
  tier: string;
  first_date: string | null;
  last_date: string | null;
  has_data: boolean;
}

export interface SparklineSeries {
  points: number[];
  change_pct: number;
  last: number;
  n: number;
}

export interface SparklineResponse {
  series: Record<string, SparklineSeries>;
  days: number;
}

export interface Page<T> {
  items: T[];
  total: number | null;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface Bar {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adj_close: number;
  volume: number;
}

export interface OHLCVResponse {
  symbol: string;
  adjusted: boolean;
  source_tier: "hot" | "cold";
  bars: Bar[];
  count: number;
}

export interface InstrumentSummaryStats {
  symbol: string;
  source_tier: string;
  as_of: string;
  last_close: number;
  change_1d: number;
  change_1w: number | null;
  change_1m: number | null;
  change_1y: number | null;
  volume: number;
  avg_volume_20d: number;
  range_52w_high: number;
  range_52w_low: number;
  n_bars: number;
  first_date: string;
}

export interface RiskResponse {
  symbol: string;
  annual_volatility: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  var_95: number | null;
  cvar_95: number | null;
  n_observations: number;
}

export interface LeaderboardRow {
  model: string;
  display_name: string | null;
  family: string;
  is_baseline: boolean;
  rmse: number | null;
  mase: number | null;
  r2: number | null;
  hit_rate: number | null;
  base_rate: number | null;
  rmse_skill_pct: number | null;
  directional_skill_pct: number | null;
  dm_pvalue: number | null;
  interval_coverage: number | null;
  sharpe: number | null;
}

export interface EvaluationResponse {
  symbol: string;
  horizon: number;
  target_type: string;
  n_folds: number;
  n_samples: number;
  n_features: number;
  leaderboard: LeaderboardRow[];
  selected_model: string;
  duration_seconds: number;
  run_ids: Record<string, string>;
}

export interface ForecastResponse {
  symbol: string;
  model: string;
  horizon: number;
  target_type: string;
  as_of_date: string;
  prediction: number;
  lower: number | null;
  upper: number | null;
  skill_pct: number | null;
  dm_pvalue: number | null;
  narrative: string;
  is_informative: boolean;
}

export interface GlossaryEntry {
  key: string;
  label: string;
  short: string;
  plain: string;
  good_direction: "higher" | "lower" | "context";
  caveat: string | null;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  family: string;
  is_classifier: boolean;
  requires_scaling: boolean;
  is_sequence_model: boolean;
  is_baseline: boolean;
  requires_extra?: string | null;
}

export interface EquityPoint {
  ts: string;
  equity: number;
  benchmark_equity: number | null;
  drawdown: number | null;
  position: number | null;
}

export interface BacktestResponse {
  symbol: string;
  model: string;
  stats: Record<string, number | string | boolean | null>;
  equity_curve: EquityPoint[];
  trades: Record<string, unknown>[];
  cost_sensitivity: Record<string, number | string | null>[];
}

export interface Methodology {
  validation: Record<string, string>;
  baselines: string[];
  reference_baseline: string;
  significance_tests: string[];
  prediction_intervals: { method: string; nominal_coverage: number };
  leakage_guards: string[];
  known_findings: Record<string, string>;
  limitations: string[];
  disclaimer: string;
}

export interface ProblemDetail {
  type: string;
  title: string;
  status: number;
  detail: string;
  request_id?: string;
  details?: Record<string, unknown>;
}
