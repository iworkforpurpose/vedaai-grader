import { DotShape, SparkleShape } from "./icons";

/**
 * The waiting screen, from the loading frame.
 *
 * It exists because the job is genuinely slow — rendering pages and reading
 * handwriting takes roughly fifteen seconds a page — and a teacher staring at a
 * frozen button assumes a failure long before the work finishes.
 *
 * `detail` is not in the frame and is optional. The frame says "This may take a
 * while", which is honest but says nothing about progress; where the pipeline can
 * report which page it is on, saying so costs nothing and turns an indefinite wait
 * into a finite one.
 */
export function LoadingStage({
  title = "Extracting...",
  note = "This may take a while",
  detail,
}: {
  title?: string;
  note?: string;
  detail?: string;
}): React.JSX.Element {
  return (
    <div className="stage" role="status" aria-live="polite">
      <div className="stage-inner">
        {/* Four sparkles at the frame's sizes: 96, 72, 29, and a 12px dot. */}
        <div className="sparkles" aria-hidden>
          <span data-s="lg">
            <SparkleShape size={96} />
          </span>
          <span data-s="md">
            <SparkleShape size={72} />
          </span>
          <span data-s="sm">
            <SparkleShape size={29} />
          </span>
          <span data-s="dot">
            <DotShape size={12} />
          </span>
        </div>

        <div>
          <p className="stage-title">{title}</p>
          <p className="stage-note">{note}</p>
        </div>

        {detail && <p className="stage-detail">{detail}</p>}
      </div>
    </div>
  );
}
