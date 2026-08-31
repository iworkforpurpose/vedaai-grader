import { Fragment } from "react";

/** Milliseconds between one character appearing and the next. */
const STEP_MS = 42;

/**
 * A phrase that arrives a character at a time.
 *
 * Every character is in the DOM from the first frame and only its opacity
 * changes, which is the part that matters on this screen. A typewriter that grows
 * its box would reflow the heading on every keystroke, and the heading sits above
 * the hero and the drop zones — so the whole page would shuffle downward for a
 * second and a quarter while it typed. Revealing in place costs nothing and moves
 * nothing.
 *
 * Opacity rather than a transform for the same reason: `transform` does not apply
 * to inline elements, so animating it would mean `display: inline-block` on every
 * character, and that changes how the line breaks. The title wraps at phone width
 * and carries `text-wrap: balance`; neither survives thirty inline-blocks intact.
 *
 * The reveal is a step rather than a fade. A character that fades up over 200ms
 * reads as an entrance; one that simply appears reads as having been typed, which
 * is what was asked for.
 */
export function TypedPhrase({ text }: { text: string }): React.JSX.Element {
  const words = text.split(" ");

  // Character index across the whole phrase, so the delay keeps climbing through
  // the spaces rather than restarting at each word.
  let index = 0;

  return (
    <>
      {/*
       * The real string, for anything that reads rather than looks.
       *
       * Thirty separately-marked-up characters are a hostile thing to hand a
       * screen reader — some announce them one at a time — so the animated copy
       * is hidden from the accessibility tree and this carries the text.
       */}
      <span className="sr-only">{text}</span>

      <span className="typed" aria-hidden="true">
        {words.map((word, wordIndex) => (
          <Fragment key={wordIndex}>
            <span className="typed-word">
              {[...word].map((character, characterIndex) => {
                const delay = index * STEP_MS;
                index += 1;
                return (
                  <span
                    key={characterIndex}
                    className="typed-char"
                    style={{ animationDelay: `${delay}ms` }}
                  >
                    {character}
                  </span>
                );
              })}
            </span>
            {/* A real space between word spans, so the line still breaks here. */}
            {wordIndex < words.length - 1 ? " " : null}
          </Fragment>
        ))}

        {/*
         * The caret. It blinks while the phrase types and then leaves, because a
         * caret that stays is a text field, and this is a heading.
         */}
        <span
          className="typed-caret"
          style={{ animationDelay: `0ms, ${index * STEP_MS + 220}ms` }}
        />
      </span>
    </>
  );
}
