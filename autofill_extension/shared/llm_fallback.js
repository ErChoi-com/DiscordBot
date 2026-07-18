// Tier 3: client-side LLM fallback. ES module (imported directly by the
// background service worker, which is declared "type":"module" in the
// manifest) - fetch calls belong in the background, not a content script,
// since a page's CSP can interfere with cross-origin fetch from an
// injected content script but not from the background's own context.
//
// Mirrors resume.py's LLM_PROVIDER_SWITCH_ORDER = ("gemini", "gemini-flash",
// "openrouter", "groq") and the real REST shapes that SDK wraps
// (src/services/resumes/listing.py: _generate_with_gemini,
// _generate_with_openai_compatible_provider) - without that file's
// heavier validation/retry sophistication, appropriate here since a wrong
// autofill guess is trivially user-correctable, unlike a fabricated resume
// bullet.

const DEFAULT_MODELS = {
  gemini: "gemini-3.5-flash",
  "gemini-flash": "gemini-3.1-flash-lite",
  openrouter: "nvidia/nemotron-3-super-120b-a12b:free",
  groq: "openai/gpt-oss-120b",
};

function buildPrompt(fields, profile, jobDescription) {
  const fieldLines = fields
    .map((f) => {
      const opts = f.options && f.options.length ? ` Options: ${f.options.map((o) => o.text).join(" | ")}.` : "";
      return `${f.id}: "${f.label}" (${f.fieldType}).${opts}`;
    })
    .join("\n");

  return [
    "You are helping fill out a job application form. Given the candidate",
    "profile and job description below, provide a value for each listed",
    "form field. Respond with ONLY a JSON object mapping each field id to",
    "its answer - no markdown fences, no other text.",
    "",
    "Fields:",
    fieldLines,
    "",
    "Candidate profile:",
    JSON.stringify(profile),
    "",
    "Job description (may be partial):",
    String(jobDescription || "").slice(0, 6000),
    "",
    'Return JSON like {"f0": "answer", "f1": "answer"}. For a field with',
    "Options listed, answer with the exact option text. If you cannot",
    "answer a field confidently, omit its key entirely rather than guessing.",
  ].join("\n");
}

function extractJsonObject(text) {
  if (!text) return null;
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced ? fenced[1] : text;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return null;
  try {
    return JSON.parse(candidate.slice(start, end + 1));
  } catch (err) {
    return null;
  }
}

function extractGeminiText(payload) {
  const parts = payload && payload.candidates && payload.candidates[0] && payload.candidates[0].content
    ? payload.candidates[0].content.parts
    : null;
  if (!Array.isArray(parts)) return "";
  return parts.map((p) => p.text || "").join("\n").trim();
}

function extractOpenAiCompatibleText(payload) {
  const message = payload && Array.isArray(payload.choices) && payload.choices[0] ? payload.choices[0].message : null;
  if (!message) return "";
  const content = message.content;
  if (typeof content === "string") return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((item) => (typeof item === "string" ? item : item && item.text) || "")
      .join("\n")
      .trim();
  }
  return "";
}

// Per-provider ceiling so one hanging/slow provider can't stall the whole
// fallback chain (and, transitively, the background's fill-result
// aggregation window) - a real risk once a network round trip is in the
// loop, unlike the purely-local Tiers -1/0/1/2.
const PROVIDER_TIMEOUT_MS = 8000;

async function fetchWithTimeout(url, init) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROVIDER_TIMEOUT_MS);
  try {
    return await fetch(url, Object.assign({}, init, { signal: controller.signal }));
  } finally {
    clearTimeout(timer);
  }
}

async function callGemini(apiKey, model, prompt) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { responseMimeType: "application/json" },
    }),
  });
  if (!response.ok) throw new Error(`gemini http ${response.status}`);
  const payload = await response.json();
  return extractGeminiText(payload);
}

async function callOpenAiCompatible(endpoint, apiKey, model, prompt, extraHeaders) {
  const response = await fetchWithTimeout(endpoint, {
    method: "POST",
    headers: Object.assign(
      { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      extraHeaders || {}
    ),
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2,
      response_format: { type: "json_object" },
    }),
  });
  if (!response.ok) throw new Error(`http ${response.status}`);
  const payload = await response.json();
  return extractOpenAiCompatibleText(payload);
}

// config: { gemini_api_key, gemini_model, openrouter_api_key,
// openrouter_model, groq_api_key, groq_model }. Providers with no key
// configured are skipped (not tried, not counted as a failure).
function buildProviderList(config) {
  config = config || {};
  const providers = [];
  if (config.gemini_api_key) {
    providers.push({
      name: "gemini",
      run: (prompt) => callGemini(config.gemini_api_key, config.gemini_model || DEFAULT_MODELS.gemini, prompt),
    });
    providers.push({
      name: "gemini-flash",
      run: (prompt) => callGemini(config.gemini_api_key, DEFAULT_MODELS["gemini-flash"], prompt),
    });
  }
  if (config.openrouter_api_key) {
    providers.push({
      name: "openrouter",
      run: (prompt) =>
        callOpenAiCompatible(
          "https://openrouter.ai/api/v1/chat/completions",
          config.openrouter_api_key,
          config.openrouter_model || DEFAULT_MODELS.openrouter,
          prompt,
          { "HTTP-Referer": "https://github.com/", "X-Title": "Application Autofill" }
        ),
    });
  }
  if (config.groq_api_key) {
    providers.push({
      name: "groq",
      run: (prompt) =>
        callOpenAiCompatible(
          "https://api.groq.com/openai/v1/chat/completions",
          config.groq_api_key,
          config.groq_model || DEFAULT_MODELS.groq,
          prompt
        ),
    });
  }
  return providers;
}

// Returns { answers: {fieldId: value}, providerUsed: string|null }. Never
// throws - if every configured provider fails (or none are configured),
// resolves to an empty answers map so the caller can fail open.
export async function resolveFields(fields, profile, jobDescription, config) {
  if (!fields || fields.length === 0) return { answers: {}, providerUsed: null };

  const providers = buildProviderList(config);
  if (providers.length === 0) return { answers: {}, providerUsed: null };

  const prompt = buildPrompt(fields, profile, jobDescription);

  for (const provider of providers) {
    try {
      const text = await provider.run(prompt);
      const parsed = extractJsonObject(text);
      if (parsed && typeof parsed === "object") {
        return { answers: parsed, providerUsed: provider.name };
      }
    } catch (err) {
      // Try the next provider in the fallback order.
    }
  }
  return { answers: {}, providerUsed: null };
}

export const __testing = { buildPrompt, extractJsonObject, buildProviderList, DEFAULT_MODELS };
