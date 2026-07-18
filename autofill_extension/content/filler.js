// Writes resolved values into fields. Uses the native prototype value
// setter (not plain `.value =`) because React overrides the plain setter
// to track state internally and silently reverts direct writes - the
// native setter bypasses that override and is a harmless no-op-equivalent
// for Vue/Angular/vanilla HTML, which just listen for native input/change
// events. One code path, works everywhere. Falls back to simulated
// keystrokes when a masked-input library rejects a bulk value write.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function nativeValueSetter(el) {
    const proto = el.tagName === "TEXTAREA" ? global.HTMLTextAreaElement.prototype : global.HTMLInputElement.prototype;
    return Object.getOwnPropertyDescriptor(proto, "value").set;
  }

  function fireEvent(el, type, options) {
    const EventCtor = type === "input" || type === "change" ? global.Event : global.Event;
    el.dispatchEvent(new EventCtor(type, Object.assign({ bubbles: true, cancelable: false }, options)));
  }

  function writeNative(el, value) {
    nativeValueSetter(el).call(el, value);
    fireEvent(el, "input");
    fireEvent(el, "change");
  }

  async function keystrokeFallback(el, value) {
    nativeValueSetter(el).call(el, "");
    fireEvent(el, "input");
    let current = "";
    for (const ch of String(value)) {
      current += ch;
      fireEvent(el, "keydown");
      nativeValueSetter(el).call(el, current);
      fireEvent(el, "input");
      fireEvent(el, "keyup");
    }
    fireEvent(el, "change");
  }

  // Masked/formatted inputs (phone, date) reformat the raw value (e.g.
  // "5551234567" -> "(555) 123-4567") - that's a successful write, not a
  // failed one, so "did it stick" compares the significant characters
  // only, not a literal string match.
  function normalizedForCompare(text) {
    return String(text).replace(/[^a-z0-9]/gi, "").toLowerCase();
  }

  async function fillTextlike(el, value) {
    el.focus();
    writeNative(el, value);
    await sleep(0);
    if (normalizedForCompare(el.value) !== normalizedForCompare(value)) {
      await keystrokeFallback(el, value);
    }
    fireEvent(el, "blur");
    el.blur();
    return normalizedForCompare(el.value) === normalizedForCompare(value);
  }

  function normalizeText(text) {
    return String(text || "").toLowerCase().trim();
  }

  function matchOption(options, value) {
    if (value === undefined || value === null || value === "") return null;
    const normValue = normalizeText(value);
    const boolWords = typeof value === "boolean" ? (value ? ["yes", "true", "y"] : ["no", "false", "n"]) : null;

    for (const opt of options) {
      if (normalizeText(opt.value) === normValue) return opt;
    }
    for (const opt of options) {
      if (normalizeText(opt.text) === normValue) return opt;
    }
    if (boolWords) {
      for (const opt of options) {
        if (boolWords.includes(normalizeText(opt.text))) return opt;
      }
    }
    // Tier 2: token/synonym-class matching (derivation.js) - "decline"
    // finds "Prefer not to say", "Toronto, Ontario, Canada" finds the
    // "Canada" option, "3" finds the "2-4 years" bucket. Returns null on
    // ambiguity - which is also why the old bidirectional-substring pass
    // that used to sit here is gone: given the value "Georgia, Canada"
    // and options containing both "Georgia" and "Canada", substring
    // returned whichever came first in DOM order (a coin flip); the
    // fuzzy tier scores them equal and correctly refuses to pick.
    if (AF.derive) {
      const fuzzy = AF.derive.fuzzyMatchOption(options, value);
      if (fuzzy) return fuzzy;
    }
    return null;
  }

  async function fillSelect(el, options, value) {
    const match = matchOption(options, value);
    if (!match) return false;
    el.focus();
    const setter = Object.getOwnPropertyDescriptor(global.HTMLSelectElement.prototype, "value").set;
    setter.call(el, match.value);
    fireEvent(el, "input");
    fireEvent(el, "change");
    fireEvent(el, "blur");
    el.blur();
    return true;
  }

  async function fillCheckbox(el, value) {
    const desired = Boolean(value);
    if (el.checked !== desired) el.click();
    return el.checked === desired;
  }

  async function fillRadioGroup(options, value) {
    const match = matchOption(options, value);
    if (!match) return false;
    if (!match.element.checked) match.element.click();
    return match.element.checked;
  }

  const TEXTLIKE_TYPES = new Set([
    "text", "email", "tel", "number", "url", "search", "date", "range", "textarea", "password",
  ]);

  // descriptor: as built by scanner.js. value: resolved profile value.
  // Returns true if the field ended up holding the intended value.
  async function fillField(descriptor, value) {
    if (value === undefined || value === null || value === "") return false;
    if (TEXTLIKE_TYPES.has(descriptor.fieldType)) {
      return fillTextlike(descriptor.element, value);
    }
    if (descriptor.fieldType === "select") {
      return fillSelect(descriptor.element, descriptor.options, value);
    }
    if (descriptor.fieldType === "combobox") {
      return AF.widgets.fillCombobox(descriptor, value);
    }
    if (descriptor.fieldType === "checkbox") {
      return fillCheckbox(descriptor.element, value);
    }
    if (descriptor.fieldType === "radio") {
      return fillRadioGroup(descriptor.options, value);
    }
    return false;
  }

  AF.filler = { fillField, matchOption, sleep };
})(typeof window !== "undefined" ? window : self);
