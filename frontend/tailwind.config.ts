import type { Config } from "tailwindcss";

/**
 * Design tokens.
 *
 * Vivid but coherent. The palette is built from a violet→fuchsia→cyan spine
 * with per-sector accent hues, so colour carries *meaning* (which sector, which
 * direction, which confidence level) rather than just decorating. Green/red
 * stay reserved for P&L and are always paired with a directional glyph so the
 * meaning survives colour-blindness.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#07070C",
          raised: "#101018",
          overlay: "#16161F",
          hover: "#1C1C27",
        },
        line: {
          DEFAULT: "#22222E",
          strong: "#31313F",
          glow: "#3D3D52",
        },
        ink: {
          DEFAULT: "#F2F3F7",
          muted: "#A0A3B1",
          faint: "#6B6E7E",
        },
        // Primary spine
        violet: {
          DEFAULT: "#7C5CFF",
          bright: "#9B7FFF",
          deep: "#5B3FD9",
          glow: "#7C5CFF33",
        },
        fuchsia: { DEFAULT: "#E056C1", bright: "#F472D9", glow: "#E056C133" },
        cyan: { DEFAULT: "#22D3EE", bright: "#67E8F9", glow: "#22D3EE33" },
        amber: { DEFAULT: "#FBBF24", bright: "#FCD34D", glow: "#FBBF2433" },
        lime: { DEFAULT: "#A3E635", glow: "#A3E63533" },

        gain: { DEFAULT: "#2DD97B", bright: "#4ADE80", glow: "#2DD97B33", faint: "#052E16" },
        loss: { DEFAULT: "#FF5A6E", bright: "#FB7185", glow: "#FF5A6E33", faint: "#450A0A" },
        warn: { DEFAULT: "#FBBF24", glow: "#FBBF2433" },

        // Sector hues — colour as data, not decoration
        sector: {
          tech: "#7C5CFF",
          health: "#22D3EE",
          finance: "#2DD97B",
          energy: "#FB923C",
          consumer: "#E056C1",
          industrial: "#FBBF24",
          utilities: "#A3E635",
          realestate: "#F472B6",
          materials: "#38BDF8",
          telecom: "#C084FC",
          other: "#94A3B8",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.03em" }],
      },
      borderRadius: { card: "1rem", pill: "999px" },
      backgroundImage: {
        "grad-primary": "linear-gradient(135deg, #7C5CFF 0%, #E056C1 50%, #22D3EE 100%)",
        "grad-violet": "linear-gradient(135deg, #7C5CFF 0%, #E056C1 100%)",
        "grad-cyan": "linear-gradient(135deg, #22D3EE 0%, #7C5CFF 100%)",
        "grad-gain": "linear-gradient(135deg, #2DD97B 0%, #A3E635 100%)",
        "grad-loss": "linear-gradient(135deg, #FF5A6E 0%, #E056C1 100%)",
        "grad-surface": "linear-gradient(160deg, #16161F 0%, #101018 100%)",
        "mesh":
          "radial-gradient(at 12% 8%, #7C5CFF22 0px, transparent 55%)," +
          "radial-gradient(at 88% 4%, #E056C11f 0px, transparent 50%)," +
          "radial-gradient(at 62% 88%, #22D3EE1a 0px, transparent 55%)," +
          "radial-gradient(at 8% 92%, #A3E6350f 0px, transparent 45%)",
      },
      boxShadow: {
        glow: "0 0 32px -6px rgba(124,92,255,0.55)",
        "glow-lg": "0 0 60px -10px rgba(124,92,255,0.65)",
        "glow-gain": "0 0 32px -8px rgba(45,217,123,0.6)",
        "glow-loss": "0 0 32px -8px rgba(255,90,110,0.6)",
        lift: "0 12px 40px -12px rgba(0,0,0,0.8)",
      },
      animation: {
        "fade-up": "fadeUp 0.45s cubic-bezier(0.22,1,0.36,1) both",
        "fade-in": "fadeIn 0.3s ease-out both",
        "scale-in": "scaleIn 0.2s cubic-bezier(0.22,1,0.36,1) both",
        shimmer: "shimmer 1.8s ease-in-out infinite",
        float: "float 7s ease-in-out infinite",
        "pulse-glow": "pulseGlow 2.6s ease-in-out infinite",
        "slide-down": "slideDown 0.22s cubic-bezier(0.22,1,0.36,1) both",
        marquee: "marquee 42s linear infinite",
        "draw-line": "drawLine 1.4s ease-out both",
        "spin-slow": "spin 2.4s linear infinite",
      },
      keyframes: {
        fadeUp: { "0%": { opacity: "0", transform: "translateY(14px)" }, "100%": { opacity: "1", transform: "none" } },
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        scaleIn: { "0%": { opacity: "0", transform: "scale(0.96)" }, "100%": { opacity: "1", transform: "scale(1)" } },
        slideDown: { "0%": { opacity: "0", transform: "translateY(-8px)" }, "100%": { opacity: "1", transform: "none" } },
        shimmer: { "0%,100%": { opacity: "0.3" }, "50%": { opacity: "0.65" } },
        float: { "0%,100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-14px)" } },
        pulseGlow: {
          "0%,100%": { boxShadow: "0 0 24px -8px rgba(124,92,255,0.45)" },
          "50%": { boxShadow: "0 0 44px -6px rgba(124,92,255,0.8)" },
        },
        marquee: { "0%": { transform: "translateX(0)" }, "100%": { transform: "translateX(-50%)" } },
        drawLine: { "0%": { strokeDashoffset: "1000" }, "100%": { strokeDashoffset: "0" } },
      },
      transitionTimingFunction: { spring: "cubic-bezier(0.22,1,0.36,1)" },
    },
  },
  plugins: [],
};

export default config;
