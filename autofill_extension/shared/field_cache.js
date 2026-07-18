// Tier 0: signature -> confirmed-answer cache. The single highest-leverage
// tier: EEO/work-authorization/"how did you hear about us" boilerplate
// repeats near-verbatim across thousands of postings on the same ATS, so
// after an answer is confirmed once, every future occurrence resolves in
// O(1) with zero LLM cost. Populated by every tier that successfully
// fills a field, including the (Phase 2) LLM tier.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});
  const STORAGE_KEY = "af_field_cache";

  function normalize(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function signature(descriptor) {
    return `${normalize(descriptor.label)}|${descriptor.fieldType}`;
  }

  async function readCacheMap() {
    const stored = await global.chrome.storage.local.get(STORAGE_KEY);
    return stored[STORAGE_KEY] || {};
  }

  async function get(sig) {
    const map = await readCacheMap();
    return Object.prototype.hasOwnProperty.call(map, sig) ? map[sig] : undefined;
  }

  async function remember(sig, value) {
    const map = await readCacheMap();
    map[sig] = value;
    await global.chrome.storage.local.set({ [STORAGE_KEY]: map });
  }

  async function clear() {
    await global.chrome.storage.local.remove(STORAGE_KEY);
  }

  AF.cache = { signature, get, remember, clear };
})(typeof window !== "undefined" ? window : self);
