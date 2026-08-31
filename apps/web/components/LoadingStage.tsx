import { LoaderMark } from "./icons";
import { LoadingPhases } from "./LoadingPhases";

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
  phases,
}: {
  title?: string;
  note?: string;
  detail?: string;
  /**
   * Stages to cycle through in the detail line, instead of a fixed sentence.
   *
   * Takes the place of `detail` rather than sitting beside it: two lines of
   * status, one moving and one not, read as the screen disagreeing with itself.
   */
  phases?: readonly string[];
}): React.JSX.Element {
  return (
    <div className="stage" role="status" aria-live="polite">
      <div className="stage-inner">
        <LoaderMark />

        <div>
          <p className="stage-title">{title}</p>
          <p className="stage-note">{note}</p>
        </div>

        {phases ? (
          <p className="stage-detail">
            <LoadingPhases phases={phases} />
          </p>
        ) : (
          detail && <p className="stage-detail">{detail}</p>
        )}
      </div>
    </div>
  );
}
