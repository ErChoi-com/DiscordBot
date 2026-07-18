// Tier -1: honeypot / hidden-field gate. Runs before any other tier ever
// looks at a field. Mirrors the actionability check Playwright already
// uses before it will click or type into anything (attached, visible,
// receives pointer events, enabled) - see
// https://playwright.dev/docs/actionability - plus the off-screen-position
// check anti-spam honeypot fields specifically rely on
// (position:absolute;left:-9999px is the most common hiding convention,
// more common than plain display:none because naive bots only check that).
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  const OFFSCREEN_THRESHOLD_PX = 1000;

  function isAriaHiddenByAncestry(el) {
    let node = el;
    while (node && node.nodeType === 1) {
      if (node.getAttribute && node.getAttribute("aria-hidden") === "true") {
        return true;
      }
      node = node.parentElement || (node.getRootNode && node.getRootNode().host) || null;
    }
    return false;
  }

  function isPositionedOffscreen(rect) {
    const docLeft = rect.left + global.scrollX;
    const docTop = rect.top + global.scrollY;
    return docLeft < -OFFSCREEN_THRESHOLD_PX || docTop < -OFFSCREEN_THRESHOLD_PX;
  }

  // Sampled at several points, not just dead-center: real pages carry ads,
  // cookie banners, and sticky headers that can transiently overlap
  // exactly one pixel of a perfectly normal field - a real user in that
  // situation just clicks a slightly different spot on the same field, so
  // a single unlucky sample shouldn't fail an otherwise-legitimate
  // element. Found live against demoqa.com: a transient element covered
  // this exact center point on one load and not the next.
  const HIT_TEST_FRACTIONS = [
    [0.5, 0.5],
    [0.25, 0.5],
    [0.75, 0.5],
    [0.5, 0.25],
    [0.5, 0.75],
  ];

  function failsPointerEventsCheck(el, rect) {
    // Hit-test against the element's OWN root, not always `document` -
    // document.elementFromPoint() does not pierce shadow boundaries, so
    // for an element inside a shadow root it would return the shadow
    // HOST (a false "something is covering this" positive). ShadowRoot
    // also implements elementFromPoint, scoped correctly to its own tree.
    const root = el.getRootNode();
    const elementFromPoint = (x, y) =>
      root.elementFromPoint ? root.elementFromPoint(x, y) : el.ownerDocument.elementFromPoint(x, y);

    let sampledAnyPointInViewport = false;
    for (const [fx, fy] of HIT_TEST_FRACTIONS) {
      const x = rect.left + rect.width * fx;
      const y = rect.top + rect.height * fy;
      const withinViewport = x >= 0 && x <= global.innerWidth && y >= 0 && y <= global.innerHeight;
      if (!withinViewport) continue;
      sampledAnyPointInViewport = true;
      const hit = elementFromPoint(x, y);
      if (hit && (hit === el || el.contains(hit) || hit.contains(el))) return false;
    }
    // Can't reliably sample elementFromPoint outside the current viewport
    // (e.g. a perfectly normal field further down a long form) - don't
    // penalize it for needing a scroll.
    return sampledAnyPointInViewport;
  }

  // Returns { fillable: boolean, reason?: string }. opts.allowReadonly:
  // combobox inputs are legitimately readonly in several widget libraries
  // (the value is picked, not typed), so the scanner exempts them.
  function checkVisibility(el, opts) {
    if (!el.isConnected) return { fillable: false, reason: "detached" };
    if (el.disabled) return { fillable: false, reason: "disabled" };
    if (el.readOnly && !(opts && opts.allowReadonly)) {
      return { fillable: false, reason: "readonly" };
    }

    const style = global.getComputedStyle(el);
    if (style.display === "none") return { fillable: false, reason: "display:none" };
    if (style.visibility === "hidden" || style.visibility === "collapse") {
      return { fillable: false, reason: "visibility:hidden" };
    }
    if (parseFloat(style.opacity) < 0.05) return { fillable: false, reason: "opacity~0" };

    const rect = el.getBoundingClientRect();
    if (rect.width <= 1 && rect.height <= 1) return { fillable: false, reason: "zero-size" };
    if (isPositionedOffscreen(rect)) return { fillable: false, reason: "positioned-offscreen" };
    if (isAriaHiddenByAncestry(el)) return { fillable: false, reason: "aria-hidden" };
    if (failsPointerEventsCheck(el, rect)) return { fillable: false, reason: "not-hit-testable" };

    return { fillable: true };
  }

  AF.visibility = { checkVisibility };
})(typeof window !== "undefined" ? window : self);
