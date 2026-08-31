/**
 * The shapes a screen wears while it is still arriving.
 *
 * The whole page is blocks — rail, top bar and content alike — so the loading
 * state reads as the application waiting rather than as a page with a hole in it.
 *
 * Every one of them is built from the application's own layout classes: `.shell`,
 * `.rail`, `.nav-row`, `.topbar`, `.map-panes`, `.q-card`, `.dropzones`. An
 * earlier version drew its own shell with the measurements transcribed into it,
 * and two layouts describing one screen only agree until somebody edits either.
 * Reusing the rules makes alignment a property of the stylesheet instead of a set
 * of numbers kept in step by hand.
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
 * The whole page, as blocks.
 *
 * Built out of the application's own layout classes -- `.shell`, `.rail`,
 * `.rail-menu`, `.nav-row`, `.topbar`, `.topbar-actions` -- with a block wherever
 * a logo, a label or an icon would be. That is the part that matters: an earlier
 * version drew its own shell with the measurements transcribed into it, and two
 * layouts describing one screen only agree until somebody edits either. Reusing
 * the rules means the skeleton cannot land anywhere the content will not.
 *
 * The rail carries `data-collapsed` for the same reason the real one does. Show a
 * full-width rail here and the review route snaps it to icons the instant the
 * content arrives -- an animation on the one part of the screen that did not
 * change, at the moment the reader is deciding where to look.
 */
export function ShellSkeleton({
  collapsed = false,
  children,
}: {
  collapsed?: boolean;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className="shell" role="status" aria-busy="true" aria-label="Loading">
      <nav className="rail" data-collapsed={collapsed} aria-hidden="true">
        <div className="rail-head">
          <span className="brand-mark">
            <Block w="40px" h="40px" radius="var(--r-md)" />
          </span>
          <span className="brand-word">
            <Block w="96px" h="20px" />
          </span>
        </div>

        {/*
          * The toolkit button is a block itself, not a block inside one.
          *
          * Every other class reused here paints a container -- a pane, a card, a
          * chip -- and those persist through the load, so keeping their fill is
          * right. This one paints a filled control at #272727, and reusing it
          * gave a finished black button with a shimmer strip inside: the loudest
          * thing on a screen that is supposed to be waiting.
          *
          * `sk` follows `toolkit` in the stylesheet, so at equal specificity the
          * shimmer wins the fill while the geometry -- full width, 42px, pill --
          * still comes from the real rule rather than from numbers copied here.
          */}
        <span className="toolkit sk" />

        <ul className="rail-menu">
          {Array.from({ length: 7 }, (_, i) => (
            <li key={i}>
              <span className="nav-row">
                <span className="nav-icon">
                  <Block w="20px" h="20px" radius="var(--r-sm)" />
                </span>
                <span className="nav-label">
                  <Block w={`${94 - (i % 4) * 16}px`} h="14px" />
                </span>
              </span>
            </li>
          ))}
        </ul>

        <div className="rail-foot">
          <span className="nav-row">
            <span className="nav-icon">
              <Block w="20px" h="20px" radius="var(--r-sm)" />
            </span>
            <span className="nav-label">
              <Block w="66px" h="14px" />
            </span>
          </span>

          <div className="school">
            <span className="school-crest">
              <Block w="59px" h="60px" radius="50%" />
            </span>
            <span className="school-text sk-lines">
              <Block w="128px" h="14px" />
              <Block w="92px" h="12px" />
            </span>
          </div>
        </div>
      </nav>

      <div className="shell-main">
        <header className="topbar" aria-hidden="true">
          <span className="round-button">
            <Block w="22px" h="22px" radius="var(--r-sm)" />
          </span>

          <span className="crumb">
            <span className="nav-icon only-wide">
              <Block w="20px" h="20px" radius="var(--r-sm)" />
            </span>
            <Block w="62px" h="16px" />
          </span>

          <div className="topbar-actions">
            <span className="round-button only-wide">
              <Block w="20px" h="20px" radius="50%" />
            </span>
            <span className="round-button">
              <Block w="20px" h="20px" radius="50%" />
            </span>
            <span className="round-button only-wide">
              <Block w="20px" h="20px" radius="50%" />
            </span>
            <span className="user-button">
              <span className="avatar">
                <Block w="32px" h="32px" radius="50%" />
              </span>
              <span className="user-name">
                <Block w="112px" h="16px" />
              </span>
            </span>
          </div>
        </header>

        {children}
      </div>
    </div>
  );
}

/**
 * The upload screen's contents.
 *
 * Wears `.upload` itself, so the column width, the gaps and the centring are the
 * ones the real form uses rather than a reconstruction of them.
 */
export function UploadSkeleton(): React.JSX.Element {
  return (
    <div className="upload sk-screen">
      {/*
        * Viewport units, not percentages.
        *
        * `.upload` centres its children, so `.upload-heading` and `.actions` are
        * shrink-to-fit: their width comes from their contents. A block asking for
        * 90% of a parent that is measuring itself from the block is circular, and
        * it resolves to zero -- the two lines above the hero were rendering at no
        * width at all, while the hero itself, sized in vw, showed up fine.
        */}
      <div className="upload-heading">
        <Block w="min(560px, 74vw)" h="var(--fs-title)" radius="var(--r-md)" />
        <Block w="min(240px, 44vw)" h="var(--fs-lg)" />
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

      {/* Same trap: `.actions` is shrink-to-fit too. The button was fine at a
          fixed 161px; the caption beneath it was collapsing. */}
      <div className="actions">
        <Block w="161px" h="44px" radius="var(--r-pill)" />
        <Block w="min(320px, 62vw)" h="var(--fs-sm)" />
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
    <div className="map sk-screen">
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
