// Walks the DOM (recursing into open shadow roots) to find fillable
// native controls and builds a field descriptor for each: resolved label,
// field type, and (for select/radio groups) the option list. Re-scans on
// MutationObserver so fields a page injects later (an expanding section,
// a wizard step) still get picked up without the user re-clicking Fill.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  const TEXTLIKE_TYPES = new Set([
    "text", "email", "tel", "number", "url", "search", "date", "range", "password",
  ]);

  function isCandidateInput(el) {
    if (el.tagName === "SELECT" || el.tagName === "TEXTAREA") return true;
    if (el.tagName !== "INPUT") return false;
    const type = (el.getAttribute("type") || "text").toLowerCase();
    if (type === "hidden") return false;
    return TEXTLIKE_TYPES.has(type) || type === "checkbox" || type === "radio";
  }

  // Custom pickers (react-select, MUI, Ashby's location autocomplete...)
  // present as an input or div with combobox semantics. They need the
  // widget driver in widgets.js - a plain value write does nothing.
  function isComboboxElement(el) {
    if (el.getAttribute("role") === "combobox") return true;
    if (el.tagName !== "INPUT") return false;
    return (
      el.getAttribute("aria-haspopup") === "listbox" || el.getAttribute("aria-autocomplete") === "list"
    );
  }

  // Recurses into open shadow roots. Closed shadow roots are genuinely
  // inaccessible to any script (documented limitation, not fixable here).
  function collectRoots(root, out) {
    out = out || [];
    out.push(root);
    const all = root.querySelectorAll ? root.querySelectorAll("*") : [];
    for (const el of all) {
      if (el.shadowRoot) collectRoots(el.shadowRoot, out);
    }
    return out;
  }

  function textOf(node) {
    return node && node.textContent ? node.textContent.replace(/\s+/g, " ").trim() : "";
  }

  function resolveLabelledBy(el) {
    const ids = (el.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean);
    if (ids.length === 0) return "";
    const root = el.getRootNode();
    const parts = ids
      .map((id) => (root.getElementById ? root.getElementById(id) : document.getElementById(id)))
      .map(textOf)
      .filter(Boolean);
    return parts.join(" ");
  }

  // A <label> often wraps the control it names, so its textContent
  // includes the control's OWN text - for a <select> that is the entire
  // option list. Found live on a Lever apply form: a label-wrapped
  // country <select> resolved to "What is your location?Select...
  // Afghanistan...United States..." and /state/i then matched the
  // "United States" inside the option text, misclassifying the field as
  // address.region. Strip every embedded control before reading.
  function textOfLabelElement(labelEl) {
    const clone = labelEl.cloneNode(true);
    for (const control of clone.querySelectorAll("input,select,textarea")) {
      control.remove();
    }
    return textOf(clone);
  }

  function resolveLabel(el) {
    if (el.labels && el.labels.length > 0) {
      const text = textOfLabelElement(el.labels[0]);
      if (text) return text;
    }
    const ariaLabel = el.getAttribute("aria-label");
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();

    const labelledBy = resolveLabelledBy(el);
    if (labelledBy) return labelledBy;

    // BEFORE the placeholder: many widget libraries render a real <label>
    // next to (not associated with) the control - Ashby's location
    // autocomplete has <label for="uuid"> pointing at nothing while the
    // input's only attribute-level signal is placeholder "Start
    // typing...". Only trusted when the ancestor container holds exactly
    // this one control, so the label can't belong to a different field.
    const containerLabel = resolveContainerLabel(el);
    if (containerLabel) return containerLabel;

    const placeholder = el.getAttribute("placeholder");
    if (placeholder && placeholder.trim()) return placeholder.trim();

    const wrappingLabel = el.closest("label");
    if (wrappingLabel) {
      const text = textOfLabelElement(wrappingLabel);
      if (text) return text;
    }

    let node = el.previousSibling;
    while (node) {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.trim()) {
        return node.textContent.trim();
      }
      if (node.nodeType === Node.ELEMENT_NODE) {
        const text = textOf(node);
        if (text) return text;
      }
      node = node.previousSibling;
    }

    return "";
  }

  // A radio group's question text usually lives OUTSIDE the radios' own
  // <label>s (which just say "Yes"/"No"). Real Lever markup:
  //   <div class="application-label">QUESTION</div>
  //   <div class="application-field"><ul> ...radio labels... </ul></div>
  // So: fieldset legend first, then walk up from the group's common
  // container reading preceding-sibling text, stopping as soon as a level
  // contains controls that aren't part of this group (that text would
  // belong to someone else).
  function resolveGroupLabel(radios) {
    const fieldset = radios[0].closest("fieldset");
    if (fieldset) {
      const legend = fieldset.querySelector("legend");
      if (legend && textOf(legend)) return textOf(legend);
    }

    const radioSet = new Set(radios);
    let container = radios[0].parentElement;
    while (container && !radios.every((r) => container.contains(r))) {
      container = container.parentElement;
    }
    let node = container;
    for (let depth = 0; node && depth < 4; depth += 1) {
      const controls = node.querySelectorAll('input:not([type="hidden"]), select, textarea');
      const hasForeignControl = Array.from(controls).some((c) => !radioSet.has(c));
      if (hasForeignControl) break;
      let sib = node.previousSibling;
      while (sib) {
        const text = sib.nodeType === Node.TEXT_NODE ? sib.textContent.trim() : textOf(sib);
        if (text) return text;
        sib = sib.previousSibling;
      }
      node = node.parentElement;
    }
    return resolveLabel(radios[0]);
  }

  function resolveContainerLabel(el) {
    let node = el.parentElement;
    for (let depth = 0; node && depth < 4; depth += 1, node = node.parentElement) {
      const raw = Array.from(
        node.querySelectorAll('input:not([type="hidden"]), select, textarea, [role="combobox"]')
      );
      // A [role=combobox] wrapper and its inner input are ONE control.
      const controls = raw.filter((c) => !raw.some((other) => other !== c && other.contains(c)));
      if (controls.length > 1) return ""; // ambiguous container - stop entirely
      const label = node.querySelector("label");
      if (label) {
        const forId = label.getAttribute("for");
        if (forId) {
          const root = el.getRootNode();
          const target = root.getElementById
            ? root.getElementById(forId)
            : document.getElementById(forId);
          // A `for` that resolves to a DIFFERENT element means this label
          // belongs to someone else; a dangling `for` (Ashby points it at
          // a uuid that isn't any element's id) is fine.
          if (target && target !== el && !el.contains(target) && !target.contains(el)) {
            continue;
          }
        }
        const text = textOfLabelElement(label);
        if (text) return text;
      }
    }
    return "";
  }

  function buildTextlikeDescriptor(el) {
    const type = el.tagName === "TEXTAREA" ? "textarea" : (el.getAttribute("type") || "text").toLowerCase();
    return {
      element: el,
      fieldType: type,
      label: resolveLabel(el),
      name: el.getAttribute("name") || "",
      id: el.id || "",
      placeholder: el.getAttribute("placeholder") || "",
      autocomplete: el.getAttribute("autocomplete") || "",
    };
  }

  function buildSelectDescriptor(el) {
    const options = Array.from(el.options || []).map((opt) => ({
      value: opt.value,
      text: textOf(opt) || opt.value,
    }));
    return {
      element: el,
      fieldType: "select",
      label: resolveLabel(el),
      name: el.getAttribute("name") || "",
      id: el.id || "",
      placeholder: "",
      autocomplete: el.getAttribute("autocomplete") || "",
      options,
    };
  }

  function buildComboboxDescriptor(el) {
    return {
      element: el,
      fieldType: "combobox",
      label: resolveLabel(el),
      name: el.getAttribute("name") || "",
      id: el.id || "",
      placeholder: el.getAttribute("placeholder") || "",
      autocomplete: el.getAttribute("autocomplete") || "",
    };
  }

  function buildCheckboxDescriptor(el) {
    return {
      element: el,
      fieldType: "checkbox",
      label: resolveLabel(el),
      name: el.getAttribute("name") || "",
      id: el.id || "",
      placeholder: "",
      autocomplete: el.getAttribute("autocomplete") || "",
    };
  }

  function buildRadioGroupDescriptors(radios) {
    const byName = new Map();
    for (const el of radios) {
      const name = el.getAttribute("name") || `__unnamed_${el.id || Math.random()}`;
      if (!byName.has(name)) byName.set(name, []);
      byName.get(name).push(el);
    }
    const descriptors = [];
    for (const group of byName.values()) {
      descriptors.push({
        element: null,
        fieldType: "radio",
        label: resolveGroupLabel(group),
        name: group[0].getAttribute("name") || "",
        id: "",
        placeholder: "",
        autocomplete: "",
        options: group.map((el) => ({
          value: el.value,
          text: resolveLabel(el) || el.value,
          element: el,
        })),
      });
    }
    return descriptors;
  }

  // Returns an array of field descriptors that passed the Tier -1
  // visibility gate. `seen` (optional WeakSet) lets callers skip elements
  // already processed in an earlier scan pass.
  function scan(root, seen) {
    root = root || document;
    seen = seen || new WeakSet();
    const descriptors = [];
    const radios = [];

    for (const frameRoot of collectRoots(root)) {
      const candidates = frameRoot.querySelectorAll
        ? frameRoot.querySelectorAll("input, select, textarea")
        : [];
      for (const el of candidates) {
        if (!isCandidateInput(el) || seen.has(el)) continue;
        const type = (el.getAttribute("type") || "text").toLowerCase();
        if (el.tagName === "INPUT" && type === "radio") {
          radios.push(el);
          continue;
        }
        // Combobox inputs are legitimately readonly in several libraries
        // (react-select isSearchable=false renders a readonly input), so
        // the honeypot gate's readonly rejection doesn't apply to them.
        const isCombobox = el.tagName === "INPUT" && isComboboxElement(el);
        const visibility = AF.visibility.checkVisibility(el, { allowReadonly: isCombobox });
        if (!visibility.fillable) continue;

        seen.add(el);
        if (isCombobox) {
          descriptors.push(buildComboboxDescriptor(el));
        } else if (el.tagName === "SELECT") {
          descriptors.push(buildSelectDescriptor(el));
        } else if (el.tagName === "INPUT" && type === "checkbox") {
          descriptors.push(buildCheckboxDescriptor(el));
        } else {
          descriptors.push(buildTextlikeDescriptor(el));
        }
      }

      // Div-based comboboxes (MUI Select renders a <div role="combobox">
      // with no inner input at rest) aren't in the input/select/textarea
      // query at all.
      const divComboboxes = frameRoot.querySelectorAll
        ? frameRoot.querySelectorAll('[role="combobox"]')
        : [];
      for (const el of divComboboxes) {
        if (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA") continue;
        if (seen.has(el)) continue;
        const visibility = AF.visibility.checkVisibility(el, { allowReadonly: true });
        if (!visibility.fillable) continue;
        seen.add(el);
        descriptors.push(buildComboboxDescriptor(el));
      }
    }

    const visibleRadios = radios.filter((el) => {
      const visibility = AF.visibility.checkVisibility(el);
      if (visibility.fillable) seen.add(el);
      return visibility.fillable;
    });
    if (visibleRadios.length > 0) {
      descriptors.push(...buildRadioGroupDescriptors(visibleRadios));
    }

    return descriptors;
  }

  // Invokes `onNewFields(descriptors)` whenever DOM mutations reveal new
  // fillable fields not already in `seen`. Attaches a MutationObserver per
  // shadow root too, since observers don't cross shadow boundaries.
  function observe(root, seen, onNewFields) {
    root = root || document;
    const observers = [];
    let pending = false;

    function scheduleRescan() {
      if (pending) return;
      pending = true;
      setTimeout(() => {
        pending = false;
        const fresh = scan(root, seen);
        if (fresh.length > 0) onNewFields(fresh);
        attachToNewShadowRoots();
      }, 150);
    }

    function attachObserver(target) {
      const observer = new MutationObserver(scheduleRescan);
      observer.observe(target, { childList: true, subtree: true, attributes: false });
      observers.push(observer);
    }

    function attachToNewShadowRoots() {
      for (const r of collectRoots(root)) {
        if (r.__afObserved) continue;
        r.__afObserved = true;
        attachObserver(r === document ? document.documentElement : r);
      }
    }

    attachToNewShadowRoots();
    return () => observers.forEach((o) => o.disconnect());
  }

  AF.scanner = { scan, observe, resolveLabel };
})(typeof window !== "undefined" ? window : self);
