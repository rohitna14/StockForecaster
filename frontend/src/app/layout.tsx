import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { ExplainProvider, ExplainToggle } from "@/components/ExplainMode";
import { SearchBar } from "@/components/SearchBar";

export const metadata: Metadata = {
  title: {
    default: "StockForecaster — volatility forecasting you can actually trust",
    template: "%s · StockForecaster",
  },
  description:
    "Walk-forward validated forecasting with purged splits, honest baselines, " +
    "conformal prediction intervals, and published negative results.",
};

const NAV = [
  { href: "/", label: "Home" },
  { href: "/explore", label: "Explore" },
  { href: "/methodology", label: "How it works" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <ExplainProvider>
          <div className="flex min-h-screen flex-col">
            <header className="sticky top-0 z-50 border-b border-line/80 bg-canvas/70 backdrop-blur-2xl">
              <div className="mx-auto flex h-16 max-w-[1400px] items-center gap-4 px-4 sm:px-6">
                <Link href="/" className="group flex shrink-0 items-center gap-2.5">
                  <span className="grid h-9 w-9 place-items-center rounded-xl bg-grad-violet text-sm font-black text-white shadow-glow transition-transform duration-300 ease-spring group-hover:scale-110 group-hover:rotate-6">
                    SF
                  </span>
                  <span className="hidden text-sm font-bold tracking-tight sm:block">
                    Stock<span className="grad-text">Forecaster</span>
                  </span>
                </Link>

                <nav className="hidden items-center gap-1 lg:flex">
                  {NAV.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="rounded-pill px-3 py-1.5 text-sm text-ink-muted transition-all duration-200 hover:bg-violet/10 hover:text-ink"
                    >
                      {item.label}
                    </Link>
                  ))}
                </nav>

                {/* Search is the primary action, so it gets the primary space. */}
                <div className="mx-auto w-full max-w-md">
                  <SearchBar />
                </div>

                <div className="hidden shrink-0 sm:block">
                  <ExplainToggle />
                </div>
              </div>
            </header>

            <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-8 sm:px-6">
              {children}
            </main>

            <footer className="border-t border-line px-4 py-8 sm:px-6">
              <div className="mx-auto flex max-w-[1400px] flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2.5">
                  <span className="grid h-7 w-7 place-items-center rounded-lg bg-grad-violet text-[10px] font-black text-white">
                    SF
                  </span>
                  <p className="text-xs text-ink-faint">
                    Research and educational use. Not financial advice.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-faint">
                  <span className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-gain" />
                    walk-forward validated
                  </span>
                  <span>purged splits</span>
                  <span>7 baselines</span>
                  <span>145 tests</span>
                </div>
              </div>
            </footer>
          </div>
        </ExplainProvider>
      </body>
    </html>
  );
}
