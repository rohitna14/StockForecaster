"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Bar } from "@/lib/types";

/**
 * Candlestick + volume chart.
 *
 * Uses TradingView's lightweight-charts — the same rendering engine behind
 * their real product. It is what makes this read as a trading terminal rather
 * than a matplotlib screenshot, and it handles crosshair, pan and zoom for free.
 */
export function PriceChart({
  bars,
  height = 420,
  showVolume = true,
}: {
  bars: Bar[];
  height?: number;
  showVolume?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || bars.length === 0) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9AA3AF",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1A1D23" },
        horzLines: { color: "#1A1D23" },
      },
      rightPriceScale: { borderColor: "#22262D", scaleMargins: { top: 0.08, bottom: showVolume ? 0.28 : 0.08 } },
      timeScale: { borderColor: "#22262D", rightOffset: 4 },
      crosshair: {
        mode: 1,
        vertLine: { color: "#6366F1", width: 1, style: 2, labelBackgroundColor: "#6366F1" },
        horzLine: { color: "#6366F1", width: 1, style: 2, labelBackgroundColor: "#6366F1" },
      },
      handleScale: { axisPressedMouseMove: { time: true, price: false } },
    });
    chartRef.current = chart;

    const candles: ISeriesApi<"Candlestick"> = chart.addCandlestickSeries({
      upColor: "#22C55E",
      downColor: "#EF4444",
      borderUpColor: "#22C55E",
      borderDownColor: "#EF4444",
      wickUpColor: "#22C55E80",
      wickDownColor: "#EF444480",
    });

    candles.setData(
      bars.map((bar) => ({
        time: (Date.parse(bar.ts) / 1000) as UTCTimestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      })),
    );

    if (showVolume) {
      const volume = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.78, bottom: 0 },
      });
      volume.setData(
        bars.map((bar) => ({
          time: (Date.parse(bar.ts) / 1000) as UTCTimestamp,
          value: bar.volume,
          color: bar.close >= bar.open ? "#22C55E30" : "#EF444430",
        })),
      );
    }

    chart.timeScale().fitContent();

    const observer = new ResizeObserver(() => {
      chart.applyOptions({ width: container.clientWidth });
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, height, showVolume]);

  if (bars.length === 0) {
    return (
      <div
        className="grid place-items-center rounded-card border border-line text-sm text-ink-faint"
        style={{ height }}
      >
        No price data
      </div>
    );
  }

  return <div ref={containerRef} className="w-full" />;
}
