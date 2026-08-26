import { ApiUnavailableError, EXPECTED_RENDER_DPI, fetchHealth, type Health } from "@/lib/api";

/**
 * Phase 0 landing page.
 *
 * Its only job is to prove the seams hold: Next boots, the generated contract
 * types compile, the API is reachable, and both sides agree on the render DPI.
 * The upload flow lands in Phase 1 and the review surface in Phase 7.
 */

export const dynamic = "force-dynamic";

export default async function Home(): Promise<React.JSX.Element> {
  let health: Health | null = null;
  let error: string | null = null;

  try {
    health = await fetchHealth();
  } catch (caught) {
    error =
      caught instanceof ApiUnavailableError
        ? caught.message
        : caught instanceof Error
          ? caught.message
          : "Unknown error";
  }

  const dpiAgrees = health !== null && health.render_dpi === EXPECTED_RENDER_DPI;

  return (
    <main
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "var(--sp-7) var(--sp-5)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--sp-5)",
      }}
    >
      <header>
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "var(--fs-xs)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--text-muted)",
            margin: "0 0 var(--sp-2)",
          }}
        >
          Phase 0 · scaffold
        </p>
        <h1 style={{ fontSize: "var(--fs-xl)", margin: 0, letterSpacing: "-0.02em" }}>
          Answer Sheet Review
        </h1>
        <p style={{ color: "var(--text-2)", margin: "var(--sp-2) 0 0" }}>
          Upload a question paper and a handwritten answer sheet to see which question was
          answered, where the answer is, and which questions were left unanswered.
        </p>
      </header>

      <section
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderLeft: `3px solid ${
            error ? "var(--status-missing)" : dpiAgrees ? "var(--status-answered)" : "var(--status-review)"
          }`,
          borderRadius: "var(--radius)",
          padding: "var(--sp-4) var(--sp-5)",
          boxShadow: "var(--shadow)",
        }}
      >
        <h2 style={{ fontSize: "var(--fs-base)", margin: "0 0 var(--sp-3)" }}>Service check</h2>

        {error !== null ? (
          <p style={{ margin: 0, color: "var(--status-missing)" }}>{error}</p>
        ) : health === null ? null : (
          <dl
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              gap: "var(--sp-2) var(--sp-5)",
              margin: 0,
              fontSize: "var(--fs-sm)",
            }}
          >
            <dt style={{ color: "var(--text-muted)" }}>API</dt>
            <dd style={{ margin: 0 }}>
              {health.status} · v{health.version}
            </dd>

            <dt style={{ color: "var(--text-muted)" }}>Contract models</dt>
            <dd style={{ margin: 0 }}>{health.contract_model_count}</dd>

            <dt style={{ color: "var(--text-muted)" }}>Render DPI</dt>
            <dd style={{ margin: 0 }}>
              {health.render_dpi}
              {dpiAgrees ? (
                " · agrees with the frontend"
              ) : (
                <strong style={{ color: "var(--status-review)" }}>
                  {" "}
                  · disagrees with the frontend ({EXPECTED_RENDER_DPI}). Highlights would be
                  offset.
                </strong>
              )}
            </dd>
          </dl>
        )}
      </section>

      <p style={{ color: "var(--text-muted)", fontSize: "var(--fs-sm)", margin: 0 }}>
        Next: page rendering, transcription, and a debug overlay that draws every recognized
        line box over the page — so highlight geometry is verifiable by eye before anything is
        built on top of it.
      </p>
    </main>
  );
}
