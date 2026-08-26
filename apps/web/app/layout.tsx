import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Answer Sheet Review",
  description:
    "Upload a question paper and a handwritten answer sheet, then see which question was answered, where the answer is, and which questions were left unanswered.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>): React.JSX.Element {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
