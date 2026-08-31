"use client";

import { useEffect } from "react";

/**
 * A ring that expands from wherever the pointer went down, and fades.
 *
 * The whole point is that it costs nothing and is felt rather than noticed, so
 * everything here is arranged around not getting in the way.
 *
 * It draws into its own fixed layer with `pointer-events: none`, so it can never
 * intercept a click it is supposed to be decorating — which matters more here
 * than on most pages, because the answer sheet turns a click into a position and
 * would be broken by a stray overlay.
 *
 * The nodes are created and removed directly rather than held in React state.
 * A click is not information the application needs; routing it through a render
 * would re-render the whole tree several times a second while somebody works
 * through a list of questions, which is exactly the frame budget the highlight
 * animations are trying to keep.
 *
 * Reduced motion turns it off entirely rather than shortening it. An expanding
 * ring has no useful still frame — it is decoration or it is nothing, and someone
 * who has asked for less of it should get none.
 */
export function ClickPulse(): null {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const layer = document.createElement("div");
    layer.className = "pulse-layer";
    layer.setAttribute("aria-hidden", "true");
    document.body.appendChild(layer);

    const onPointerDown = (event: PointerEvent): void => {
      // Primary button only. A right-click opens a menu and a middle-click opens
      // a tab; neither is the gesture this is acknowledging.
      if (event.button !== 0) return;

      const ring = document.createElement("span");
      ring.className = "pulse";
      ring.style.left = `${event.clientX}px`;
      ring.style.top = `${event.clientY}px`;
      layer.appendChild(ring);

      // Removed by the animation it was created for, so nothing has to track it.
      ring.addEventListener("animationend", () => ring.remove(), { once: true });
    };

    // `pointerdown` rather than `click`: it fires the instant the finger or button
    // goes down, which is when a person expects the acknowledgement — waiting for
    // the release makes the interface feel like it is lagging behind them. Passive
    // because this never calls preventDefault and must not delay scrolling.
    window.addEventListener("pointerdown", onPointerDown, { passive: true });

    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      layer.remove();
    };
  }, []);

  return null;
}
