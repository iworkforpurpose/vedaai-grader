import type { Metadata } from "next";
import { Bricolage_Grotesque, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

/**
 * Both faces come from the Figma file rather than being chosen here: Bricolage
 * Grotesque carries the screen, and Inter appears on exactly one label — the
 * toolkit button — which is worth honouring rather than tidying away, because a
 * substitution there would be a silent design change.
 *
 * Loaded through `next/font`, which self-hosts them at build time. That matters
 * for the deployed container: a Google Fonts link would be a runtime dependency
 * on a third party for the page to render as designed, and a network hiccup would
 * show up as the wrong typeface rather than as an error.
 */
const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
  weight: ["400", "500", "600", "700"],
});

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
  weight: ["400", "500", "600"],
});

/**
 * The numerals.
 *
 * `--font-mono` used to resolve to a system stack, so the counts a teacher reads
 * -- how many answered, the zoom level, which page of how many -- rendered in a
 * different typeface on every machine, and on the ones without SF Mono in a
 * proportional fallback whose digits changed width as they changed value.
 *
 * Loaded the same way as the other two, so it is still self-hosted at build time
 * and still not a runtime dependency on anyone else's CDN.
 */
const mono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono-face",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Exams · VedaAI",
  description:
    "Upload a question paper and a handwritten answer sheet, then see which question was answered, where the answer is, and which questions were left unanswered.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>): React.JSX.Element {
  return (
    <html lang="en" className={`${bricolage.variable} ${inter.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
