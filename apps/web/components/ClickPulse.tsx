"use client";

import { useEffect } from "react";

/**
 * Four short strokes that flick outward from the pointer and vanish.
 *
 * Measured off myvedaai.com rather than guessed at. Clicking the same point twice
 * and two different points produced the same four bearings every time -- 211, 244,
 * 278 and 312 degrees -- so this is a fixed fan rather than a random burst, and
 * the fixedness is most of its character: it reads as one mark being stamped, not
 * as particles being thrown.
 *
 * Expressed here as CSS rotations, which measure from straight up rather than
 * from the positive x-axis, so each is the measured bearing plus ninety.
 *
 * The rest of the shape came from the same frames: the strokes travel from about
 * sixteen pixels out to thirty-one, and *shorten* as they go -- eight pixels, then
 * five, then three -- which is what stops a fan of straight lines reading as a
 * cartoon starburst. The whole thing is over in about a quarter of a second.
 *
 * Everything else here is about staying out of the way. Its own fixed layer with
 * `pointer-events: none`, so it can decorate a click but never intercept one --
 * which matters on the answer sheet, where a click is a coordinate rather than a
 * command. Nodes created and removed directly rather than held in React state,
 * because a click is not something the application needs to know and routing it
 * through a render would re-render the tree several times a second while somebody
 * works down a list of questions.
 */

/** The four bearings, converted from measured screen angles to CSS rotations. */
const RAYS = [-59, -26, 8, 42] as const;

export function ClickPulse(): null {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const layer = document.createElement("div");
    layer.className = "spark-layer";
    layer.setAttribute("aria-hidden", "true");
    document.body.appendChild(layer);

    const onPointerDown = (event: PointerEvent): void => {
      // Primary button only. A right-click opens a menu and a middle-click opens a
      // tab; neither is the gesture this is acknowledging.
      if (event.button !== 0) return;

      const burst = document.createElement("span");
      burst.className = "spark";
      burst.style.left = `${event.clientX}px`;
      burst.style.top = `${event.clientY}px`;

      for (const angle of RAYS) {
        const ray = document.createElement("i");
        ray.style.setProperty("--a", `${angle}deg`);
        burst.appendChild(ray);
      }

      layer.appendChild(burst);

      // Removed by the animation it was created for, so nothing has to track it.
      // Listening on the burst catches the last ray to finish, since they all run
      // the same duration and the event bubbles.
      burst.addEventListener("animationend", () => burst.remove(), { once: true });
    };

    // `pointerdown` rather than `click`: the acknowledgement is expected when the
    // finger goes down, and waiting for the release reads as the interface lagging
    // behind the person using it. Passive because this never calls preventDefault
    // and must not delay scrolling.
    window.addEventListener("pointerdown", onPointerDown, { passive: true });

    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      layer.remove();
    };
  }, []);

  return null;
}
