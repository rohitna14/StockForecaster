import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { ExplainProvider, ExplainToggle } from "@/components/ExplainMode";
import { CommandPalette } from "@/components/CommandPalette";

export const metadata: Metadata = {
  title: {
    default: "StockForecaster — leakage-free volatility forecasting",
    template: "%s · StockForecaster",
  },
  description:
    "Walk-forward validated forecasting with purged splits, honest baselines, " +
    "conformal prediction intervals, and published negative results.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <ExplainProvider>
          <div className="flex min-h-screen flex-col">
            <header className="sticky top-0 z-50 border-b border-line bg-canvas/85 backdrop-blur-xl">
              <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-6 px-5">
                <div className="flex items-center gap-7">
                  <Link href="/" className="flex items-center gap-2.5">
                    <span className="grid h-6 w-6 place-items-center rounded-md bg-accent text-[11px] font-bold text-white">
                      SF
                    </span>
                    <span className="text-sm font-semibold tracking-tight">
                      StockForecaster
                    </span>
                  </Link>

                  <nav className="hidden items-center gap-5 text-sm text-ink-muted md:flex">
                    <Link href="/s/AAPL" className="transition-colors hover:text-ink">
                      Explore
                    </Link>
                    <Link href="/methodology" className="transition-colors hover:text-ink">
                      Methodology
                    </Link>
                    <Link href="/results" className="transition-colors hover:text-ink">
                      Results
                    </Link>
                  </nav>
                </div>

                <div className="flex items-center gap-3">
                  <ExplainToggle />
                  <CommandPalette />
                </div>
              </div>
            </header>

            <main className="mx-auto w-full max-w-7xl flex-1 px-5 py-8">{children}</main>

            <footer className="border-t border-line px-5 py-6">
              <div className="mx-auto flex max-w-7xl flex-col gap-2 text-xs text-ink-faint sm:flex-row sm:items-center sm:justify-between">
                <p>
                  Research and educational use. Not financial advice. Past performance
                  does not predict future results.
                </p>
                <p className="tnum font-mono">
                  Walk-forward · purged · baseline-benchmarked
                </p>
              </div>
            </footer>
          </div>
        </ExplainProvider>
      </body>
    </html>
  );
}
