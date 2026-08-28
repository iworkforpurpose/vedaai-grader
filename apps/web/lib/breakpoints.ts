"use client";

import { useEffect, useState } from "react";

/** Where the navigation rail stops being a drawer and becomes a column. */
export const RAIL_BREAKPOINT = 1024;

/**
 * Whether the layout is in its narrow form, where the rail is a drawer over the
 * content rather than a column beside it.
 *
 * The CSS already knows this from a media query. This exists for the things a
 * media query cannot reach: the `inert` attribute on a faded-out pane, and the
 * label on a control whose meaning differs between the two layouts.
 *
 * A subscription rather than a one-off read, so resizing a window does not leave
 * either of those describing the layout it used to be.
 */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(false);

  useEffect(() => {
    const query = window.matchMedia(`(max-width: ${RAIL_BREAKPOINT - 1}px)`);
    const sync = () => setNarrow(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  return narrow;
}
