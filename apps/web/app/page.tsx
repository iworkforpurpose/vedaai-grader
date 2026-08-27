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
    <main className="upload-page">
      <header className="upload-hero">
        <p
          className="eyebrow"
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
        <h1>Grade handwritten answers with confidence.</h1>
        <p>Upload an exam paper and answer sheet to map answers, surface gaps, and review marks in one focused workspace.</p>
      </header>

      <div className="upload-grid">
        <section className="upload-card service-card">
          <div className="card-heading"><span className="step-number">01</span><h2>Service check</h2></div>

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

        <section className="upload-card upload-card-main">
          <div className="card-heading"><span className="step-number">02</span><h2>Upload documents</h2></div>
        <UploadForm />
        </section>
      </div>
    </main>
  );
}
