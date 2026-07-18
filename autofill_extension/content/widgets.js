// Drives custom (non-native) widgets: ARIA combobox/listbox pickers
// (react-select, MUI, Ant Design, Ashby's location autocomplete, ...)
// and ARIA radio/checkbox divs. Direct value writes do nothing to these -
// they must be operated the way a user operates them: open, filter,
// pick a rendered option, verify it stuck.
//
// Event recipes verified against real library source (react-select
// Select.tsx, job_app_filler's Greenhouse/Workday adapters, MUI's own
// test conventions):
//   - open on MOUSEDOWN (react-select/MUI/antd listen there, not click)
//   - filter by native-setter + InputEvent('input') (React re-renders)
//   - select with the full pointerdown->mousedown->pointerup->mouseup->
//     click sequence on the option node (covers select-on-mousedown,
//     select-on-mouseup, and select-on-click libraries in one path)
//   - none of the mainstream form-widget libraries check event.isTrusted
//   - match against the options the OPEN reveals first; type-to-filter
//     only when that found nothing (typing "3" would filter out the
//     "2-4 years" bucket, while async Places-style lookups and
//     virtualized long lists render nothing UNTIL text is typed)
//   - never keyboard-Enter before async options arrive (Places-style
//     lookups) or the raw text gets committed instead of a selection
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  // Rolling trace of the last combobox interaction - reading this after
  // a silent skip is how live-page failures get diagnosed without
  // reproducing the whole fill.
  let lastTrace = [];
  function trace(msg) {
    lastTrace.push(msg);
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  const MOUSE_SEQUENCE = ["pointerdown", "mousedown", "pointerup", "mouseup", "click"];

  function fireMouse(el, type) {
    const Ctor = type.startsWith("pointer") && global.PointerEvent ? global.PointerEvent : global.MouseEvent;
    el.dispatchEvent(
      new Ctor(type, { bubbles: true, cancelable: true, composed: true, view: global, button: 0 })
    );
  }

  function syntheticClick(el) {
    for (const type of MOUSE_SEQUENCE) fireMouse(el, type);
  }

  function fireKey(el, type, key, keyCode) {
    el.dispatchEvent(
      new KeyboardEvent(type, { bubbles: true, cancelable: true, composed: true, key, keyCode, which: keyCode })
    );
  }

  function setFilterText(input, text) {
    const setter = Object.getOwnPropertyDescriptor(global.HTMLInputElement.prototype, "value").set;
    setter.call(input, text);
    input.dispatchEvent(
      new InputEvent("input", { bubbles: true, composed: true, data: text, inputType: "insertText" })
    );
  }

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 1 && rect.height <= 1) return false;
    const style = global.getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden";
  }

  // The listbox usually lives in a body-level portal, NOT near the input.
  // Association chain, strongest first: aria-controls / aria-owns id ->
  // any visible [role="listbox"] in the input's root -> in document.
  function findListbox(el) {
    const root = el.getRootNode();
    const byId = (id) =>
      (root.getElementById ? root.getElementById(id) : null) || global.document.getElementById(id);
    for (const attr of ["aria-controls", "aria-owns"]) {
      const id = el.getAttribute(attr);
      if (id) {
        const target = byId(id.split(/\s+/)[0]);
        if (target && isVisible(target)) return target;
      }
    }
    const scopes = root === global.document ? [global.document] : [root, global.document];
    for (const scope of scopes) {
      const candidates = Array.from(scope.querySelectorAll('[role="listbox"]')).filter(isVisible);
      if (candidates.length > 0) return candidates[candidates.length - 1]; // newest portal wins
    }
    return null;
  }

  function collectOptions(el) {
    // While the widget itself says it's closed, there are no options FOR
    // IT - the document-wide listbox fallback must not run, or a
    // just-used neighboring combobox's still-mounted portal menu gets
    // mistaken for this one's (found via a flaky fixture run: the gender
    // picker collected the years picker's bucket options).
    if (el.getAttribute("aria-expanded") === "false") return [];
    const listbox = findListbox(el);
    const scope = listbox || el.getRootNode();
    const nodes = Array.from(scope.querySelectorAll('[role="option"]')).filter(
      (o) => isVisible(o) && o.getAttribute("aria-disabled") !== "true"
    );
    return nodes.map((node) => ({
      element: node,
      value: node.getAttribute("data-value") || "",
      text: (node.textContent || "").replace(/\s+/g, " ").trim(),
    }));
  }

  async function waitForOptions(el, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    let last = [];
    while (Date.now() < deadline) {
      const options = collectOptions(el).filter(
        // Loading placeholders ("Searching...", "Loading...") show up as
        // option rows in several libraries and must not be clicked.
        (o) => o.text && !/^(searching|loading|no (options|results))/i.test(o.text)
      );
      if (options.length > 0 && options.length === last.length) return options; // settled
      last = options;
      await sleep(120);
    }
    return last;
  }

  // Does the widget currently DISPLAY this text? Checked as plain
  // normalized containment against the input's own value first, then
  // small ancestor scopes (div comboboxes render the pick as trigger
  // text). Deliberately NOT fuzzyMatchOption: that function ranks
  // candidate options and skips decline-class text - fed a whole-form
  // text blob it once skipped a correct verification because the form's
  // pronouns section contained "Prefer..." (found live on Ashby, where
  // the driver then dismissed - and thereby CLEARED - its own successful
  // pick).
  function displayShows(el, text) {
    const norm = AF.derive.normalize(text);
    if (!norm) return false;
    if (el.tagName === "INPUT" && AF.derive.normalize(el.value).includes(norm)) return true;
    let scope = el;
    for (let i = 0; i < 3 && scope; i += 1) {
      if (AF.derive.normalize(scope.textContent || "").includes(norm)) return true;
      scope = scope.parentElement;
    }
    return false;
  }

  function dismiss(el) {
    if (el.tagName === "INPUT" && !el.readOnly) setFilterText(el, "");
    fireKey(el, "keydown", "Escape", 27);
    fireKey(el, "keyup", "Escape", 27);
    el.blur && el.blur();
  }

  // descriptor: a "combobox" descriptor from scanner.js. value: resolved
  // profile answer (string). Returns true only when the selection
  // verifiably stuck.
  async function fillCombobox(descriptor, value) {
    const el = descriptor.element;
    const target = String(value);
    lastTrace = [`target=${target}`];

    // Already showing the right value (e.g. second fill click)? Done.
    if (displayShows(el, target)) {
      trace("already-set");
      return true;
    }

    el.focus();
    syntheticClick(el); // mousedown in the sequence is what opens most libraries

    // Ladder: match against whatever options the open reveals FIRST -
    // typing first would break semantic matches (typing "3" filters out
    // the "2-4 years" bucket; typing "decline" is the only case it
    // helps). Only when nothing matched, type-to-filter and retry: that
    // is what async lookups (Ashby's Places-style location field shows
    // nothing until text arrives) and virtualized long lists need.
    let options = await waitForOptions(el, 800);
    trace(`open-wait: expanded=${el.getAttribute("aria-expanded")} options=${options.length}`);
    if (options.length === 0 && el.getAttribute("aria-expanded") === "false") {
      syntheticClick(el); // first open click can lose a race with the widget mounting
      options = await waitForOptions(el, 800);
      trace(`re-click: expanded=${el.getAttribute("aria-expanded")} options=${options.length}`);
    }
    let mapped = options.map((o) => ({ value: o.value || o.text, text: o.text }));
    let match = AF.derive.fuzzyMatchOption(mapped, target);

    const canType = el.tagName === "INPUT" && !el.readOnly;
    if (!match && canType) {
      // The most selective short piece (first comma segment): Places
      // lookups match on the city, filters keep the list small.
      setFilterText(el, target.split(",")[0].trim());
      options = await waitForOptions(el, 2500);
      mapped = options.map((o) => ({ value: o.value || o.text, text: o.text }));
      match = AF.derive.fuzzyMatchOption(mapped, target);
      trace(`typed-filter: options=${options.length} texts=${JSON.stringify(mapped.slice(0, 5).map((m) => m.text))}`);
    }

    if (!match) {
      trace("no-match: dismissing");
      dismiss(el);
      return false;
    }
    trace(`match: ${match.text}`);
    const optionNode = options[mapped.indexOf(match)].element;

    syntheticClick(optionNode);

    // Verify the pick stuck; fall back to the keyboard path if not
    // (aria-activedescendant navigation + Enter), then re-verify.
    if (await verifySelection(el, match.text)) {
      trace("verified");
      return true;
    }
    trace(`click-not-verified: value=${el.tagName === "INPUT" ? el.value : "n/a"}`);

    if (await keyboardSelect(el, match.text)) {
      trace("keyboard-verified");
      return true;
    }

    trace("keyboard-failed: dismissing");
    dismiss(el);
    return false;
  }

  async function verifySelection(el, optionText) {
    const deadline = Date.now() + 900;
    while (Date.now() < deadline) {
      await sleep(120);
      if (displayShows(el, optionText)) return true;
    }
    trace(`verify-miss: value=${el.tagName === "INPUT" ? JSON.stringify(el.value) : "n/a"}`);
    return false;
  }

  async function keyboardSelect(el, optionText) {
    const targetNorm = AF.derive.normalize(optionText);
    for (let i = 0; i < 30; i += 1) {
      fireKey(el, "keydown", "ArrowDown", 40);
      fireKey(el, "keyup", "ArrowDown", 40);
      await sleep(40);
      const activeId = el.getAttribute("aria-activedescendant");
      if (!activeId) break;
      const root = el.getRootNode();
      const active =
        (root.getElementById ? root.getElementById(activeId) : null) ||
        global.document.getElementById(activeId);
      if (active && AF.derive.normalize(active.textContent) === targetNorm) {
        fireKey(el, "keydown", "Enter", 13);
        fireKey(el, "keyup", "Enter", 13);
        return verifySelection(el, optionText);
      }
    }
    return false;
  }

  // ARIA radio/checkbox divs: click first (every WAI-ARIA implementation
  // binds it), keyboard Space as fallback, verify via aria-checked.
  async function setAriaChecked(el, desired) {
    const read = () => el.getAttribute("aria-checked") === "true";
    if (read() === desired) return true;
    el.focus();
    syntheticClick(el);
    await sleep(80);
    if (read() === desired) return true;
    fireKey(el, "keydown", " ", 32);
    fireKey(el, "keyup", " ", 32);
    await sleep(80);
    return read() === desired;
  }

  AF.widgets = {
    fillCombobox,
    setAriaChecked,
    findListbox,
    get lastTrace() {
      return lastTrace.slice();
    },
  };
})(typeof window !== "undefined" ? window : self);
