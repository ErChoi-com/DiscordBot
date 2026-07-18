// Orchestrates a fill: injects the content-script bundle into the active
// tab (all frames, so embedded ATS iframes get covered too), broadcasts a
// PERFORM_FILL trigger, and aggregates the FILL_RESULT each frame reports
// back independently (sidesteps any ambiguity in how tabs.sendMessage
// response callbacks behave across multiple frames - every frame just
// fires its own runtime.sendMessage back to us instead). Also owns the
// Tier 3 LLM fallback calls themselves - fetches belong here, not in a
// content script, since a page's CSP can interfere with cross-origin
// fetch from an injected content script but never from the background's
// own context.

import { resolveFields } from "../shared/llm_fallback.js";

const CONTENT_FILES = [
  "shared/schema.js",
  "shared/field_visibility.js",
  "shared/field_cache.js",
  "shared/derivation.js",
  "shared/field_heuristics.js",
  "content/scanner.js",
  "content/widgets.js",
  "content/filler.js",
  "content/content_main.js",
];

// Generous relative to Phase 1: a frame's FILL_RESULT may now arrive only
// after it has awaited a full LLM round trip (each provider bounded at
// PROVIDER_TIMEOUT_MS in llm_fallback.js, several tried in sequence on
// failure) - too short a cap here would make the popup report a summary
// that's missing fields the content script was still about to fill.
const AGGREGATION_HARD_CAP_MS = 40000;
// Settle windows are only for the two cases counting can't cover: a
// straggler FILL_STARTED ack arriving after every acked frame already
// reported, and a page where no frame ever acks at all (restricted
// injection). They are NOT how completion is normally detected - that's
// the started/reported frame count. A pure settle-window design was a
// real, live-found bug: an empty subframe's instant total:0 reply closed
// the window while the main frame was still mid-fill (fills are
// deliberately staggered 50-150ms apart), so jobs.lever.co reported
// "0 filled" for a form it was actively filling.
const AGGREGATION_SETTLE_MS = 300;
const NO_FRAMES_ACKED_TIMEOUT_MS = 3000;

function emptySummary() {
  return { total: 0, filled: 0, skipped: 0, fields: [] };
}

function mergeSummary(target, addition) {
  target.total += addition.total;
  target.filled += addition.filled;
  target.skipped += addition.skipped;
  target.fields.push(...addition.fields);
  return target;
}

async function fillActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    return { ok: false, error: "No active tab." };
  }

  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      files: CONTENT_FILES,
    });
  } catch (err) {
    return { ok: false, error: `Can't run on this page: ${err.message}` };
  }

  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const aggregate = emptySummary();

  const resultPromise = new Promise((resolve) => {
    let framesStarted = 0;
    let framesReported = 0;
    let settleTimer = null;
    const hardCap = setTimeout(finish, AGGREGATION_HARD_CAP_MS);
    // If nothing acks at all (injection restricted everywhere), don't sit
    // on the hard cap for 40s just to return an empty summary.
    let noAckTimer = setTimeout(finish, NO_FRAMES_ACKED_TIMEOUT_MS);

    function finish() {
      clearTimeout(hardCap);
      if (noAckTimer) clearTimeout(noAckTimer);
      if (settleTimer) clearTimeout(settleTimer);
      chrome.runtime.onMessage.removeListener(onMessage);
      resolve(aggregate);
    }

    function scheduleSettledFinish() {
      // Every frame that acked has reported. Hold the short settle window
      // open only for a straggler ack racing the last result.
      if (settleTimer) clearTimeout(settleTimer);
      settleTimer = setTimeout(finish, AGGREGATION_SETTLE_MS);
    }

    function onMessage(message) {
      if (!message || message.requestId !== requestId) return;
      if (message.type === "FILL_STARTED") {
        framesStarted += 1;
        if (noAckTimer) {
          clearTimeout(noAckTimer);
          noAckTimer = null;
        }
        // A frame is now working; completion is counted, not timed.
        if (settleTimer) {
          clearTimeout(settleTimer);
          settleTimer = null;
        }
      } else if (message.type === "FILL_RESULT") {
        framesReported += 1;
        mergeSummary(aggregate, message.summary);
        if (framesReported >= framesStarted) scheduleSettledFinish();
      }
    }

    chrome.runtime.onMessage.addListener(onMessage);
  });

  try {
    await chrome.tabs.sendMessage(tab.id, { type: "PERFORM_FILL", requestId });
  } catch (err) {
    // Top frame may have no listener yet on a page with restricted
    // injection; other frames can still respond independently.
  }

  const summary = await resultPromise;
  return { ok: true, summary };
}

async function resolveWithLlm(fields, profile, jobDescription) {
  const stored = await chrome.storage.local.get("af_llm_config");
  const config = stored.af_llm_config || {};
  return resolveFields(fields, profile, jobDescription, config);
}

// Exposed so the Playwright e2e harness can trigger a fill directly on the
// service worker without needing to simulate a toolbar-icon click (the
// activeTab permission's gesture requirement isn't otherwise automatable).
// Not reachable from web content - a MV3 service worker's global scope is
// not exposed to page JS.
self.__afFillActiveTab = fillActiveTab;
self.__afResolveWithLlm = resolveWithLlm;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.type === "FILL_ACTIVE_TAB") {
    fillActiveTab().then(sendResponse);
    return true; // keep the message channel open for the async response
  }
  if (message && message.type === "RESOLVE_WITH_LLM") {
    resolveWithLlm(message.fields, message.profile, message.jobDescription).then(sendResponse);
    return true;
  }
  return undefined;
});
