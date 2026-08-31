/**
 * The shapes a screen wears while it is still arriving.
 *
 * Deliberately not a spinner. A spinner says "something is happening" and nothing
 * else; a skeleton says where things will be, so the eye has already settled on
 * the layout by the time the content lands and the arrival reads as filling in
 * rather than as replacing. It also cannot spin forever in a way that looks fine,
 * which is the failure mode of a spinner over a request that never returns.
 *
 * The blocks mirror the real geometry rather than approximating it -- the rail is
 * the rail's width, the cards are the card's height -- because a skeleton whose
 * boxes move when the content arrives is worse than none. That is the whole
 * mechanism, so the sizes here are load-bearing.
 */

/** One shimmering block. `w` and `h` are any CSS length. */
function Block({
  w = "100%",
  h = "1rem",
  radius = "var(--r-sm)",
  className,
}: {
  w?: string;
  h?: string;
  radius?: string;
  className?: string;
}): React.JSX.Element {
  return (
    <span
      className={className ? `sk ${className}` : "sk"}
      style={{ width: w, height: h, borderRadius: radius }}
    />
  );
}

/** The navigation rail and top bar, which are present on every screen. */
function Chrome(): React.JSX.Element {
  return (
    <>
      <div className="sk-rail" aria-hidden="true">
        <div className="sk-rail-head">
          <Block w="40px" h="40px" radius="var(--r-md)" />
          <Block w="96px" h="20px" />
        </div>
        <Block w="100%" h="42px" radius="var(--r-circle)" />
        <div className="sk-rail-menu">
          {Array.from({ length: 7 }, (_, i) => (
            <Block key={i} w="100%" h="38px" radius="var(--r-sm)" />
          ))}
        </div>
        <div className="sk-rail-foot">
          <Block w="100%" h="64px" radius="var(--r-md)" />
        </div>
      </div>

      <div className="sk-topbar" aria-hidden="true">
        <Block w="36px" h="36px" radius="50%" />
        <Block w="88px" h="18px" />
        <span className="sk-spacer" />
        <Block w="36px" h="36px" radius="50%" />
        <Block w="36px" h="36px" radius="50%" />
        <Block w="132px" h="32px" radius="var(--r-circle)" />
      </div>
    </>
  );
}

/**
 * The upload screen: title, hero, two drop zones, the button.
 *
 * Announced as a busy region rather than silently drawn. A screen reader gets one
 * "Loading" and then the finished page, which is the same information a sighted
 * reader gets from the shapes.
 */
export function UploadSkeleton(): React.JSX.Element {
  return (
    <div className="sk-shell" role="status" aria-busy="true" aria-label="Loading">
      <Chrome />
      <div className="sk-main">
        <div className="sk-upload">
          <Block w="min(520px, 80%)" h="44px" radius="var(--r-md)" />
          <Block w="min(280px, 60%)" h="22px" />
          <Block w="138px" h="138px" radius="50%" className="sk-hero" />
          <div className="sk-dropzones">
            <Block w="100%" h="133px" radius="var(--r-lg)" />
            <Block w="100%" h="133px" radius="var(--r-lg)" />
          </div>
          <Block w="161px" h="44px" radius="var(--r-pill)" />
        </div>
      </div>
    </div>
  );
}

/**
 * The review screen: the question list beside the answer sheet.
 *
 * The rail is drawn collapsed, because the review route collapses it — a skeleton
 * that shows a full rail and then snaps to icons has animated the wrong thing at
 * exactly the moment the reader is deciding where to look.
 */
export function ReviewSkeleton(): React.JSX.Element {
  return (
    <div className="sk-shell" data-collapsed="true" role="status" aria-busy="true" aria-label="Loading">
      <div className="sk-rail sk-rail-narrow" aria-hidden="true">
        <Block w="40px" h="40px" radius="var(--r-md)" />
        <Block w="40px" h="40px" radius="50%" />
        <div className="sk-rail-menu">
          {Array.from({ length: 7 }, (_, i) => (
            <Block key={i} w="32px" h="32px" radius="var(--r-sm)" />
          ))}
        </div>
      </div>

      <div className="sk-topbar" aria-hidden="true">
        <Block w="36px" h="36px" radius="50%" />
        <Block w="88px" h="18px" />
        <span className="sk-spacer" />
        <Block w="132px" h="32px" radius="var(--r-circle)" />
      </div>

      <div className="sk-main">
        <div className="sk-panes">
          <div className="sk-pane">
            <Block w="min(320px, 70%)" h="20px" />
            <div className="sk-actions">
              <Block w="112px" h="38px" radius="var(--r-circle)" />
              <Block w="104px" h="38px" radius="var(--r-circle)" />
            </div>
            <Block w="min(380px, 85%)" h="16px" />
            <div className="sk-cards">
              {Array.from({ length: 6 }, (_, i) => (
                <div key={i} className="sk-card">
                  <Block w="28px" h="28px" radius="50%" />
                  <span className="sk-card-text">
                    <Block w="90%" h="14px" />
                    <Block w="62%" h="14px" />
                  </span>
                  <Block w="76px" h="30px" radius="var(--r-circle)" />
                </div>
              ))}
            </div>
          </div>

          <div className="sk-sheet">
            <div className="sk-sheet-bar">
              <Block w="112px" h="18px" />
              <span className="sk-spacer" />
              <Block w="104px" h="30px" radius="var(--r-circle)" />
              <Block w="132px" h="30px" radius="var(--r-circle)" />
            </div>
            <div className="sk-page">
              {Array.from({ length: 9 }, (_, i) => (
                <Block key={i} w={`${92 - (i % 3) * 14}%`} h="14px" />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
