/**
 * The shapes a screen wears while it is still arriving.
 *
 * These are the *contents* of a screen, not a screen. The rail and the top bar
 * are rendered for real by `AppShell` around them, because neither depends on
 * anything being fetched — they are the same markup either side of the load, so
 * drawing placeholder versions of them would be inventing a second layout that
 * has to be kept in step with the first. It would not be, and the seam is exactly
 * the misalignment a skeleton is supposed to avoid.
 *
 * For the same reason everything here reuses the real layout classes — `.map`,
 * `.q-pane`, `.q-card`, `.sheet-bar`, `.upload`, `.dropzones` — and only replaces
 * the text and images inside them with blocks. Alignment is then a property of
 * the stylesheet rather than a set of numbers copied into a second one.
 *
 * Deliberately not a spinner. A spinner says "something is happening" and nothing
 * else; a skeleton says where things will be, so the eye has already settled on
 * the layout by the time the content lands.
 */

/** One shimmering block. `w` and `h` are any CSS length. */
function Block({
  w = "100%",
  h = "1rem",
  radius = "var(--r-sm)",
}: {
  w?: string;
  h?: string;
  radius?: string;
}): React.JSX.Element {
  return <span className="sk" style={{ width: w, height: h, borderRadius: radius }} />;
}

/**
 * The upload screen's contents.
 *
 * Wears `.upload` itself, so the column width, the gaps and the centring are the
 * ones the real form uses rather than a reconstruction of them.
 */
export function UploadSkeleton(): React.JSX.Element {
  return (
    <div className="upload sk-screen" role="status" aria-busy="true" aria-label="Loading">
      <div className="upload-heading">
        <Block w="min(560px, 90%)" h="var(--fs-title)" radius="var(--r-md)" />
        <Block w="min(240px, 55%)" h="var(--fs-lg)" />
      </div>

      {/* The hero is a circle at the frame's size, so the drop zones below it land
          where they land on the real screen rather than a few pixels off. */}
      <Block w="clamp(110px, 26vw, 138px)" h="clamp(110px, 26vw, 138px)" radius="50%" />

      <div className="dropzones">
        <div className="dropzone-slot">
          <span className="dropzone sk-quiet" />
        </div>
        <div className="dropzone-slot">
          <span className="dropzone sk-quiet" />
        </div>
      </div>

      <div className="actions">
        <Block w="161px" h="44px" radius="var(--r-pill)" />
        <Block w="min(320px, 80%)" h="var(--fs-sm)" />
      </div>
    </div>
  );
}

/**
 * The review screen's contents: the question list beside the answer sheet.
 *
 * Six cards because that is roughly what fits before the fold at the desktop
 * frame — enough that the list reads as a list, not so many that the skeleton
 * claims to know how long the paper is.
 */
export function ReviewSkeleton(): React.JSX.Element {
  return (
    <div className="map sk-screen" role="status" aria-busy="true" aria-label="Loading">
      <div className="map-panes">
        <section className="q-pane">
          <div className="q-head">
            <Block w="min(300px, 70%)" h="var(--fs-base)" />
          </div>

          <div className="q-head-actions">
            <Block w="112px" h="38px" radius="var(--r-circle)" />
            <Block w="104px" h="38px" radius="var(--r-circle)" />
          </div>

          <Block w="min(380px, 85%)" h="var(--fs-sm)" />

          <div className="q-list">
            {Array.from({ length: 6 }, (_, i) => (
              <div className="q-card" key={i}>
                <div className="q-row">
                  <Block w="28px" h="28px" radius="50%" />
                  <span className="q-text sk-lines">
                    <Block w="92%" h="0.9rem" />
                    <Block w={`${68 - (i % 3) * 12}%`} h="0.9rem" />
                  </span>
                  <span className="q-right">
                    <Block w="76px" h="30px" radius="var(--r-circle)" />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="sheet-pane">
          <div className="sheet-bar">
            <Block w="112px" h="var(--fs-base)" />
            <div className="sheet-tools">
              <Block w="104px" h="30px" radius="var(--r-circle)" />
              <Block w="132px" h="30px" radius="var(--r-circle)" />
            </div>
          </div>

          {/* Ruled lines at the answer sheet's own rhythm, so the page reads as a
              page rather than as a stack of bars. */}
          <div className="sheet-scroll sk-sheet-body">
            {Array.from({ length: 10 }, (_, i) => (
              <Block key={i} w={`${94 - (i % 4) * 13}%`} h="0.85rem" />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
