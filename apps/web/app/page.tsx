import { UploadForm } from "@/components/UploadForm";
import { ApiUnavailableError, EXPECTED_RENDER_DPI, fetchHealth, type Health } from "@/lib/api";

/**
 * Upload page.
 *
 * Also cross-checks the render DPI against the pipeline service. A mismatch
 * would not throw — normalized coordinates mean the browser never divides by
 * DPI — it would silently offset every highlight, so it is worth surfacing.
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
          Teacher review
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

      <section
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          padding: "var(--sp-5)",
          boxShadow: "var(--shadow)",
        }}
      >
        <h2 style={{ fontSize: "var(--fs-lg)", margin: "0 0 var(--sp-4)" }}>Upload</h2>
        <UploadForm />
      </section>
    </main>
  );
}
