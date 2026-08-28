() => {
  const F = [];
  const add = (kind, detail) => F.push({ kind, detail });
  const name = (el) => {
    const c = (el.className && el.className.toString ? el.className.toString() : "").trim();
    return (el.tagName.toLowerCase() + (c ? "." + c.split(/\s+/).slice(0,2).join(".") : "")).slice(0, 44);
  };
  const scrollers = [...document.querySelectorAll("*")].filter((el) => {
    const s = getComputedStyle(el);
    const oy = s.overflowY, ox = s.overflowX;
    return ((oy === "auto" || oy === "scroll") && el.scrollHeight > el.clientHeight + 1)
        || ((ox === "auto" || ox === "scroll") && el.scrollWidth  > el.clientWidth  + 1);
  });

  // 1. Dead scroll space: room to scroll past the last thing in the box.
  for (const el of scrollers) {
    // Deepest painted bottom anywhere inside, not just the direct children's
    // boxes: a child that is shorter than its own content would otherwise make
    // real content below the fold look like empty space.
    const all = [...el.querySelectorAll("*")].filter((k) => {
      const r = k.getBoundingClientRect();
      const cs = getComputedStyle(k);
      return r.height > 0 && cs.visibility !== "hidden" && cs.display !== "none";
    });
    if (!all.length) continue;
    const host = el.getBoundingClientRect();
    const lastBottom = Math.max(...all.map((k) => k.getBoundingClientRect().bottom));
    const remaining = el.scrollHeight - el.clientHeight - el.scrollTop;   // still scrollable
    const belowFold = lastBottom - host.bottom;                          // content still to come
    const cs = getComputedStyle(el);
    // A scroller's own bottom padding is scrollable and is meant to be — reaching
    // the end of a document should show its bottom margin. Counting it made the
    // navigation rail report 12-24px of "dead" space on every short viewport,
    // which is the kind of standing false positive that gets a check ignored.
    const padBottom = parseFloat(cs.paddingBottom) || 0;
    const dead = Math.round(remaining - Math.max(0, belowFold) - padBottom);
    if (dead > 2) add("dead-scroll", `${name(el)} can scroll ${dead}px past its content`);
    if (cs.paddingTop !== "0px" || cs.paddingBottom !== "0px")
      add("scroller-padding", `${name(el)} pt=${cs.paddingTop} pb=${cs.paddingBottom} (counts as scrollable space)`);
  }

  // 2. Page-level overflow.
  const d = document.documentElement;
  if (d.scrollWidth  > d.clientWidth  + 1) add("page-overflow-x", `${d.scrollWidth - d.clientWidth}px`);
  if (d.scrollHeight > d.clientHeight + 1) add("page-overflow-y", `${d.scrollHeight - d.clientHeight}px`);

  // 3. Interactive elements clipped horizontally by their scroll parent.
  const xParent = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const o = getComputedStyle(p).overflowX;
      if (o === "auto" || o === "scroll" || o === "hidden") return p;
    }
    return null;
  };
  // Skip the drawer's contents only while it is genuinely off screen. The check is
  // narrow on purpose: written any looser it excused everything inside the rail,
  // and the phone drawer was never examined at all.
  const inClosedRail = (el) => {
    for (let p = el; p; p = p.parentElement) {
      if (!p.classList?.contains("rail")) continue;
      if (p.dataset.open === "true") return false;
      const cs = getComputedStyle(p);
      if (cs.position !== "fixed") return false;
      const r = p.getBoundingClientRect();
      return r.right <= 0 || r.left >= document.documentElement.clientWidth;
    }
    return false;
  };
  for (const el of document.querySelectorAll("button, a[href], input, [role='tab']")) {
    const b = el.getBoundingClientRect();
    if (b.width === 0 && b.height === 0) continue;
    if (inClosedRail(el)) continue;
    if (el.closest("[inert]")) continue;
    const host = xParent(el);
    const lim = host ? host.getBoundingClientRect() : { left: 0, right: d.clientWidth };
    if (b.right > lim.right + 1 || b.left < lim.left - 1)
      add("clipped-control", `${name(el)} x[${Math.round(b.left)},${Math.round(b.right)}] vs [${Math.round(lim.left)},${Math.round(lim.right)}]`);
  }

  // 4. Tap targets.
  for (const el of document.querySelectorAll("button, a[href], [role='tab'], input[type=checkbox]")) {
    const b = el.getBoundingClientRect();
    if (b.width === 0 || b.height === 0) continue;
    if (inClosedRail(el) || el.closest("[inert]")) continue;
    if (b.height < 24 || b.width < 24)
      add("tiny-target", `${name(el)} ${Math.round(b.width)}x${Math.round(b.height)}`);
  }

  // 5. Distorted images.
  for (const im of document.querySelectorAll("img")) {
    const b = im.getBoundingClientRect();
    if (!im.naturalWidth || !b.width || !b.height) continue;
    const fit = getComputedStyle(im).objectFit;
    if (fit !== "fill") continue;
    const want = im.naturalWidth / im.naturalHeight, got = b.width / b.height;
    if (Math.abs(want - got) / want > 0.02)
      add("stretched-image", `${im.getAttribute("src")?.slice(-28)} ${Math.round(b.width)}x${Math.round(b.height)} vs natural ${im.naturalWidth}x${im.naturalHeight}`);
  }

  // 6. A wide rail with no labels — the fault a clean run missed entirely.
  const rail = document.querySelector(".rail");
  if (rail && !inClosedRail(rail)) {
    const rw = rail.getBoundingClientRect().width;
    const rows = [...rail.querySelectorAll(".nav-row")];
    const hidden = rows.filter((row) => {
      const lab = row.querySelector(".nav-label");
      return lab && getComputedStyle(lab).display === "none";
    });
    if (rw > 120 && hidden.length === rows.length && rows.length > 0)
      add("rail-labels-hidden", `rail is ${Math.round(rw)}px wide with all ${rows.length} labels hidden`);
    for (const row of rows) {
      const b = row.getBoundingClientRect();
      const icon = row.querySelector(".nav-icon");
      if (!icon) continue;
      const ib = icon.getBoundingClientRect();
      // An icon parked at the far left of a wide empty row is the visual symptom.
      if (rw > 120 && b.width - (ib.right - b.left) > b.width * 0.6 && hidden.length)
        add("rail-row-empty", `${row.textContent.trim().slice(0,14) || "row"} ${Math.round(b.width)}px wide, icon only`);
    }
  }

  // 7. Hygiene.
  const ids = {};
  for (const el of document.querySelectorAll("[id]")) ids[el.id] = (ids[el.id] || 0) + 1;
  for (const [id, n] of Object.entries(ids)) if (n > 1) add("duplicate-id", `#${id} x${n}`);
  for (const im of document.querySelectorAll("img")) if (im.getAttribute("alt") === null) add("img-no-alt", name(im));
  for (const el of document.querySelectorAll("button")) {
    if (!el.textContent.trim() && !el.getAttribute("aria-label") && !el.getAttribute("title"))
      add("unlabelled-button", name(el));
  }

  // 8. Text clipped by its own box.
  for (const el of document.querySelectorAll("h1,h2,h3,p,span,button,a")) {
    if (el.children.length) continue;
    const s = getComputedStyle(el);
    if (s.overflow === "visible" && s.textOverflow !== "ellipsis") continue;
    if (el.scrollWidth > el.clientWidth + 1 && s.textOverflow !== "ellipsis")
      add("clipped-text", `${name(el)} "${el.textContent.trim().slice(0,24)}" ${el.scrollWidth}>${el.clientWidth}`);
  }

  return F;
}
