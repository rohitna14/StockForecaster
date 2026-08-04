/**
 * Typed API client.
 *
 * All network access goes through `request()`, which understands the backend's
 * RFC 7807 `problem+json` errors and rethrows them as `ApiError` carrying the
 * request id — so a failure in the UI can be traced to a line in the server log.
 */

import type {
  BacktestResponse,
  EvaluationResponse,
  ForecastResponse,
  GlossaryEntry,
  InstrumentSummary,
  InstrumentSummaryStats,
  Methodology,
  ModelInfo,
  OHLCVResponse,
  Page,
  ProblemDetail,
  RiskResponse,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly problem?: ProblemDetail,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True when the resource simply has not been ingested yet. */
  get isMissingData(): boolean {
    return this.status === 404 || this.problem?.type.includes("not_found") === true;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { params?: Record<string, string | number | boolean | undefined> },
): Promise<T> {
  const { params, ...rest } = init ?? {};

  const url = new URL(`${API_URL}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }

  let response: Response;
  try {
    response = await fetch(url.toString(), {
      ...rest,
      headers: { "Content-Type": "application/json", ...rest.headers },
    });
  } catch (cause) {
    throw new ApiError(
      `Cannot reach the API at ${API_URL}. Is the backend running?`,
      0,
    );
  }

  if (!response.ok) {
    let problem: ProblemDetail | undefined;
    try {
      problem = (await response.json()) as ProblemDetail;
    } catch {
      // Non-JSON error body (proxy timeout, etc.) — fall through.
    }
    throw new ApiError(
      problem?.detail ?? `${response.status} ${response.statusText}`,
      response.status,
      problem,
    );
  }

  return (await response.json()) as T;
}

export const api = {
  // ── instruments ────────────────────────────────────────────────────────
  searchInstruments: (q?: string, limit = 20) =>
    request<Page<InstrumentSummary>>("/instruments", { params: { q, limit } }),

  getInstrument: (symbol: string) =>
    request<InstrumentSummary>(`/instruments/${symbol}`),

  getSummary: (symbol: string) =>
    request<InstrumentSummaryStats>(`/instruments/${symbol}/summary`),

  getOHLCV: (symbol: string, limit = 750) =>
    request<OHLCVResponse>(`/instruments/${symbol}/ohlcv`, { params: { limit } }),

  getRisk: (symbol: string) =>
    request<RiskResponse>(`/instruments/${symbol}/risk`),

  getIndicators: (symbol: string, features: string, limit = 500) =>
    request<{
      symbol: string;
      features: string[];
      index: string[];
      values: Record<string, (number | null)[]>;
    }>(`/instruments/${symbol}/indicators`, { params: { features, limit } }),

  // ── catalog ────────────────────────────────────────────────────────────
  getModels: () => request<ModelInfo[]>("/models"),
  getGlossary: () => request<GlossaryEntry[]>("/glossary"),
  getMethodology: () => request<Methodology>("/methodology"),
  getBaselines: () =>
    request<{ reference: string; note: string; baselines: ModelInfo[] }>("/baselines"),

  // ── evaluation ─────────────────────────────────────────────────────────
  evaluate: (body: {
    symbol: string;
    models: string[];
    horizon: number;
    target_type: string;
    train_size?: number;
    test_size?: number;
  }) =>
    request<EvaluationResponse>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getForecast: (symbol: string, model = "lightgbm", horizon = 5, target = "vol_ratio") =>
    request<ForecastResponse>(`/forecast/${symbol}`, {
      params: { model, horizon, target_type: target },
    }),

  // ── backtests ──────────────────────────────────────────────────────────
  backtest: (body: {
    symbol: string;
    model: string;
    horizon: number;
    cost_preset: string;
    sizing?: string;
  }) =>
    request<BacktestResponse>("/backtests", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  health: () => request<{ status: string; version: string }>("/health"),
};
