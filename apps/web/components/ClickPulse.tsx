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

/** How far a touch may wander and still count as a tap rather than a scroll. */
const TAP_SLOP_PX = 10;

export function ClickPulse(): null {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const layer = document.createElement("div");
    layer.className = "spark-layer";
    layer.setAttribute("aria-hidden", "true");
    document.body.appendChild(layer);

    const spark = (x: number, y: number): void => {
      const burst = document.createElement("span");
      burst.className = "spark";
      burst.style.left = `${x}px`;
      burst.style.top = `${y}px`;

      for (const angle of RAYS) {
        const ray = document.createElement("i");
        ray.style.setProperty("--a", `${angle}deg`);
        burst.appendChild(ray);
      }

      layer.appendChild(burst);
      // Removed by the animation it was created for, so nothing has to track it.
      burst.addEventListener("animationend", () => burst.remove(), { once: true });
    };

    /*
     * A touch that has gone down but has not yet earned a spark.
     *
     * A mouse press cannot be a scroll -- scrolling is a wheel -- so a mouse gets
     * its spark immediately, which is where the responsiveness comes from. A
     * finger press usually *is* a scroll, and firing on contact meant every swipe
     * down the question list left a spark behind it. So touch waits for the lift
     * and only sparks if the finger stayed put.
     */
    let pending: { id: number; x: number; y: number } | null = null;

    const onPointerDown = (event: PointerEvent): void => {
      // Primary button only. A right-click opens a menu and a middle-click opens
      // a tab; neither is the gesture this is acknowledging.
      if (event.button !== 0) return;

      if (event.pointerType === "mouse") {
        spark(event.clientX, event.clientY);
        return;
      }
      pending = { id: event.pointerId, x: event.clientX, y: event.clientY };
    };

    const onPointerMove = (event: PointerEvent): void => {
      if (!pending || event.pointerId !== pending.id) return;
      const travelled = Math.hypot(event.clientX - pending.x, event.clientY - pending.y);
      // Far enough to be a scroll or a drag, so it was never a tap.
      if (travelled > TAP_SLOP_PX) pending = null;
    };

    const onPointerUp = (event: PointerEvent): void => {
      if (!pending || event.pointerId !== pending.id) return;
      const travelled = Math.hypot(event.clientX - pending.x, event.clientY - pending.y);
      if (travelled <= TAP_SLOP_PX) spark(event.clientX, event.clientY);
      pending = null;
    };

    const forget = (event: PointerEvent): void => {
      if (pending && event.pointerId === pending.id) pending = null;
    };

    // Passive throughout: none of these call preventDefault, and a non-passive
    // listener on pointermove would put this in the way of every scroll.
    window.addEventListener("pointerdown", onPointerDown, { passive: true });
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerup", onPointerUp, { passive: true });
    window.addEventListener("pointercancel", forget, { passive: true });

    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", forget);
      layer.remove();
    };
  }, []);

  return null;
}
