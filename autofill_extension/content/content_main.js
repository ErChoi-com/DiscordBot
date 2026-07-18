// Entry point injected into each frame. Wires scan -> resolve (cache,
// heuristics, then optionally LLM) -> fill, staggers fills to look like a
// human tabbing through the form (reliability against bot-detection, not
// evasion), and keeps a MutationObserver running so fields that appear
// later in this frame's lifetime still get resolved without a second
// click - the observer path only runs Tiers -1/0/1 (never the LLM tier),
// so a page that keeps mutating its DOM can't keep triggering paid LLM
// calls in the background; the LLM tier only ever runs once, for the
// fields still unresolved right after an explicit "Fill this page" click.
(function (global) {
  "use strict";

  if (global.__afContentMainLoaded) return;
  global.__afContentMainLoaded = true;

  const AF = global.AF || {};

  function randomDelay(minMs, maxMs) {
    return minMs + Math.random() * (maxMs - minMs);
  }

  // Tiers -1/0/1 only (visibility gate already applied by the scanner).
  // Returns { summary, unresolved } - unresolved descriptors are exactly
  // what a caller may want to hand off to the LLM tier.
  async function resolveTiers01(descriptors, profile) {
    const summary = { total: descriptors.length, filled: 0, skipped: 0, fields: [] };

    const withCacheResult = await Promise.all(
      descriptors.map(async (d) => ({ descriptor: d, cached: await AF.cache.get(AF.cache.signature(d)) }))
    );

    const needsHeuristics = [];
    const resolved = [];
    for (const { descriptor, cached } of withCacheResult) {
      if (cached !== undefined) {
        resolved.push({ descriptor, value: cached, source: "cache" });
      } else {
        needsHeuristics.push(descriptor);
      }
    }

    const assignments = [];
    for (const descriptor of needsHeuristics) {
      const classification = AF.heuristics.classify(descriptor);
      if (classification) assignments.push({ descriptor, ...classification });
    }
    const rationalized = AF.heuristics.rationalize(assignments);
    const resolvedDescriptors = new Set();
    for (const assignment of rationalized) {
      const value = AF.heuristics.resolveValue(profile, assignment.path);
      if (value !== undefined && value !== null && value !== "") {
        resolved.push({ descriptor: assignment.descriptor, value, source: assignment.path });
        resolvedDescriptors.add(assignment.descriptor);
      }
    }

    for (const { descriptor, value, source } of resolved) {
      await AF.filler.sleep(randomDelay(50, 150));
      const ok = await AF.filler.fillField(descriptor, value);
      if (ok) {
        await AF.cache.remember(AF.cache.signature(descriptor), value);
        summary.filled += 1;
        summary.fields.push({ label: descriptor.label, fieldType: descriptor.fieldType, source, ok: true });
      } else {
        summary.skipped += 1;
        summary.fields.push({ label: descriptor.label, fieldType: descriptor.fieldType, source, ok: false });
      }
    }

    const resolvedSet = new Set(resolved.map((r) => r.descriptor));
    const unresolved = descriptors.filter((d) => !resolvedSet.has(d));
    summary.skipped += unresolved.length;
    return { summary, unresolved };
  }

  // Tier 3: batches every still-unresolved field into one request to the
  // background service worker (which owns the actual provider fetch
  // calls). Fail-open by construction - resolveWithLlm never throws, an
  // empty/failed response just means every field here stays unfilled.
  async function resolveWithLlm(descriptors, profile) {
    const result = { filled: 0, skipped: 0, fields: [] };
    if (descriptors.length === 0) return result;

    const idMap = new Map();
    const serializable = descriptors.map((d, i) => {
      const id = `f${i}`;
      idMap.set(id, d);
      return {
        id,
        label: d.label,
        fieldType: d.fieldType,
        options: d.options ? d.options.map((o) => ({ value: o.value, text: o.text })) : undefined,
      };
    });

    const jobDescription = document.body ? document.body.innerText.slice(0, 6000) : "";

    let response = null;
    try {
      response = await chrome.runtime.sendMessage({
        type: "RESOLVE_WITH_LLM",
        fields: serializable,
        profile,
        jobDescription,
      });
    } catch (err) {
      response = null;
    }

    const answers = (response && response.answers) || {};
    const providerUsed = response && response.providerUsed;

    for (const [id, descriptor] of idMap) {
      const value = answers[id];
      if (value === undefined || value === null || value === "") {
        result.skipped += 1;
        result.fields.push({ label: descriptor.label, fieldType: descriptor.fieldType, source: null, ok: false });
        continue;
      }
      await AF.filler.sleep(randomDelay(50, 150));
      const ok = await AF.filler.fillField(descriptor, value);
      if (ok) await AF.cache.remember(AF.cache.signature(descriptor), value);
      result[ok ? "filled" : "skipped"] += 1;
      result.fields.push({
        label: descriptor.label,
        fieldType: descriptor.fieldType,
        source: ok ? `llm:${providerUsed}` : null,
        ok,
      });
    }
    return result;
  }

  function mergeInto(summary, addition) {
    summary.filled += addition.filled;
    summary.skipped -= addition.fields.length; // these were double-counted as unresolved above
    summary.skipped += addition.skipped;
    summary.fields.push(...addition.fields);
    return summary;
  }

  async function loadActiveProfile() {
    const stored = await chrome.storage.local.get(["af_profiles", "af_active_profile_id"]);
    const profiles = stored.af_profiles || {};
    const activeId = stored.af_active_profile_id;
    if (activeId && profiles[activeId]) return profiles[activeId];
    const firstKey = Object.keys(profiles)[0];
    return firstKey ? profiles[firstKey] : AF.schema.emptyProfile();
  }

  let observerStarted = false;
  function ensureObserverRunning(profile) {
    if (observerStarted) return;
    observerStarted = true;
    const seen = new WeakSet();
    AF.scanner.scan(document, seen); // mark the initial-pass elements as already seen
    AF.scanner.observe(document, seen, (freshDescriptors) => {
      resolveTiers01(freshDescriptors, profile).catch(() => {});
    });
  }

  chrome.runtime.onMessage.addListener((message) => {
    if (!message || message.type !== "PERFORM_FILL") return undefined;

    // ACK before doing any work: the background aggregates one
    // FILL_RESULT per frame, and without a started-count it can only
    // guess when all frames have reported. Found live on jobs.lever.co /
    // jobs.ashbyhq.com: an empty subframe reported total:0 within
    // milliseconds and the aggregation settle window closed before this
    // frame's staggered fills (deliberately 50-150ms apart) finished, so
    // the user-visible summary said 0 while the form filled anyway.
    chrome.runtime.sendMessage({ type: "FILL_STARTED", requestId: message.requestId }).catch(() => {});

    (async () => {
      const profile = await loadActiveProfile();
      const seen = new WeakSet();
      const descriptors = AF.scanner.scan(document, seen);
      const { summary, unresolved } = await resolveTiers01(descriptors, profile);
      if (unresolved.length > 0) {
        const llmResult = await resolveWithLlm(unresolved, profile);
        mergeInto(summary, llmResult);
      }
      ensureObserverRunning(profile);
      chrome.runtime.sendMessage({ type: "FILL_RESULT", requestId: message.requestId, summary }).catch(() => {});
    })();

    return undefined;
  });
})(typeof window !== "undefined" ? window : self);
