import type { Config } from "tailwindcss";

/**
 * Design tokens.
 *
 * Two rules the whole UI follows:
 *
 * 1. **One accent colour.** Indigo. Everything else is neutral. The previous
 *    Streamlit app used six competing gradients plus an animated hue-rotate on
 *    the page background; restraint is what reads as expensive.
 * 2. **Green/red are reserved for P&L semantics only** and are always paired
 *    with a directional glyph, so the meaning survives colour-blindness and
 *    greyscale printing.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#0A0B0D",
          raised: "#121417",
          overlay: "#171A1F",
        },
        line: {
          DEFAULT: "#22262D",
          strong: "#31363F",
        },
        ink: {
          DEFAULT: "#E8EAED",
          muted: "#9AA3AF",
          faint: "#646C78",
        },
        accent: {
          DEFAULT: "#6366F1",
          hover: "#7C7FF5",
          muted: "#3730A3",
          faint: "#1E1B4B",
        },
        gain: { DEFAULT: "#22C55E", faint: "#052E16" },
        loss: { DEFAULT: "#EF4444", faint: "#450A0A" },
        warn: { DEFAULT: "#F59E0B", faint: "#451A03" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.02em" }],
      },
      borderRadius: {
        card: "0.75rem",
      },
      animation: {
        "fade-in": "fadeIn 0.25s ease-out",
        shimmer: "shimmer 1.6s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%, 100%": { opacity: "0.35" },
          "50%": { opacity: "0.7" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
