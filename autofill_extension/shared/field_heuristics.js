// Tier 1: HTML `autocomplete` attribute first (standardized, language- and
// framework-agnostic - https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#autofill),
// then an ordered regex/keyword table against the resolved label, followed
// by a rationalization pass so two fields on one page can't both claim
// "email" - the lower-confidence one is left unfilled rather than wrong.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  const AUTOCOMPLETE_MAP = {
    "given-name": "contact.first_name",
    "additional-name": "contact.first_name",
    "family-name": "contact.last_name",
    name: "contact.full_name",
    email: "contact.email",
    tel: "contact.phone",
    "tel-national": "contact.phone",
    url: "contact.website",
    "street-address": "contact.address.street",
    "address-line1": "contact.address.street",
    "address-level2": "contact.address.city",
    "address-level1": "contact.address.region",
    "postal-code": "contact.address.postal_code",
    country: "contact.address.country",
    "country-name": "contact.address.country",
  };

  // Ordered most-specific-first: a field is classified by the first
  // pattern that matches across all its text sources (label, name, id,
  // placeholder). Confidence is CONFIDENCE_REGEX for every entry here -
  // ordering handles specificity (e.g. "first name" is checked before the
  // generic "name" pattern), rationalization handles cross-field conflicts.
  //
  // Rules with `fieldTypes` only apply to those field types - EEO
  // demographics, for example, are only ever answered into option-bearing
  // widgets (select/radio/combobox), never typed into a free-text input.
  //
  // Work-authorization ordering is deliberate and polarity-critical:
  // the compound "authorized ... without sponsorship" phrasing is checked
  // BEFORE either single-keyword rule, and the authorization rule before
  // the bare sponsorship rule - a live Lever posting asks "Are you
  // authorized to work in the United States without sponsorship?" and a
  // naive /sponsor/-first table (which shipped job-apply bots actually
  // have) answers it backwards.
  const REGEX_TABLE = [
    { path: "contact.first_name", pattern: /first\s*name|given\s*name|\bfname\b/i },
    { path: "contact.last_name", pattern: /last\s*name|family\s*name|surname|\blname\b/i },
    {
      path: "derived.authorized_without_sponsorship",
      pattern: /(authori[sz]\w*|legally\s+(entitled|permitted|eligible)).{0,60}without\s+\w*\s*sponsor/i,
    },
    {
      path: "work_authorization.authorized",
      pattern: /authoriz\w*\s+to\s+work|legally\s+(entitled|authorized|permitted|eligible)\s+to\s+work|work\s+authorization/i,
    },
    { path: "work_authorization.requires_sponsorship", pattern: /sponsor(ship)?/i },
    { path: "contact.email", pattern: /e-?mail/i },
    { path: "contact.phone", pattern: /phone|mobile|cell/i },
    { path: "contact.linkedin", pattern: /linked\s*in/i },
    { path: "contact.github", pattern: /github/i },
    { path: "contact.website", pattern: /website|portfolio|personal\s*site/i },
    {
      path: "derived.current_company",
      pattern: /current\s+(company|employer)|most\s+recent\s+(company|employer)|present\s+employer|^\s*(current\s+)?employer\s*$/i,
    },
    {
      path: "derived.current_title",
      pattern: /current\s+(job\s+)?title|(most\s+recent|current)\s+(role|position)|^\s*(job\s+)?title\s*$/i,
    },
    {
      path: "derived.years_experience",
      // Only GENERIC experience questions: "Years of Python experience"
      // must NOT match - per-skill years need dated per-skill evidence
      // the profile doesn't carry, and answering with the TOTAL would
      // overclaim. Skill-specific phrasings fall through to the LLM tier.
      pattern: /(how\s+many\s+)?years?\s+of\s+(relevant\s+|professional\s+|work\s+|total\s+|full.?time\s+)*experience|experience\s*\(\s*(in\s+)?years/i,
    },
    {
      path: "derived.highest_degree",
      pattern: /highest\s+(level\s+of\s+)?education|education\s+level|highest\s+degree|level\s+of\s+education|\bdegree\b/i,
    },
    { path: "derived.school", pattern: /school|university|college|alma\s*mater/i },
    { path: "derived.graduation_year", pattern: /graduat\w*\s+(year|date)|year\s+of\s+graduation/i },
    {
      path: "derived.eeo.gender",
      pattern: /gender/i,
      fieldTypes: ["select", "radio", "combobox"],
    },
    {
      path: "derived.eeo.race",
      pattern: /\brace\b|ethnicit/i,
      fieldTypes: ["select", "radio", "combobox"],
    },
    {
      path: "derived.eeo.veteran_status",
      pattern: /veteran/i,
      fieldTypes: ["select", "radio", "combobox"],
    },
    {
      path: "derived.eeo.disability",
      pattern: /disabilit/i,
      fieldTypes: ["select", "radio", "combobox"],
    },
    // preferences.* are user-STATED facts (typed into the profile or
    // extracted from the user's own description text by the importer) -
    // an empty/null preference means the question simply isn't answered.
    // This is not the derivation layer guessing: salary et al. stay on
    // the never-derive list; these rules only ROUTE a stated answer.
    {
      path: "preferences.salary_expectation",
      // Deliberately NOT matching "hourly rate": a stated expectation
      // like "$85-95k" is annual, and nothing here can convert units -
      // answering an hourly question with an annual range is a wrong
      // fill, so hourly questions stay blank (or go to the LLM tier).
      pattern: /salary|compensation|desired\s+pay|pay\s+(expectation|range)/i,
    },
    { path: "preferences.notice_period", pattern: /notice\s+period/i },
    {
      path: "preferences.earliest_start",
      pattern: /earliest\s+start|available\s+to\s+start|when\s+can\s+you\s+start|start\s+date|availability\s+date/i,
    },
    {
      path: "preferences.willing_to_relocate",
      pattern: /reloc/i,
      fieldTypes: ["select", "radio", "checkbox", "combobox"],
    },
    {
      path: "preferences.work_mode",
      pattern: /work\s+(model|mode|arrangement|setting|preference)|remote\s+or\s+(hybrid|on-?site)|(remote|hybrid|onsite).{0,15}preference/i,
      fieldTypes: ["select", "radio", "combobox"],
    },
    {
      path: "preferences.referral_source",
      pattern: /how\s+did\s+you\s+(hear|find|learn)|where\s+did\s+you\s+hear|referral\s+source/i,
    },
    {
      path: "preferences.drivers_license",
      pattern: /driver.?s?\s+licen[cs]e/i,
      fieldTypes: ["select", "radio", "checkbox", "combobox"],
    },
    { path: "contact.address.postal_code", pattern: /zip|postal/i },
    { path: "contact.address.country", pattern: /country/i },
    { path: "contact.address.region", pattern: /\bstate\b|province|region/i },
    { path: "contact.address.city", pattern: /city|town/i },
    // AFTER the specific address parts: a bare/leading "location" or a
    // "What is your location?" phrasing gets the combined
    // "City, Region, Country" string (typeaheads and country dropdowns
    // both resolve it via fuzzy option matching).
    { path: "derived.location", pattern: /^\s*(current\s+)?location\b|\byour\s+location\b/i },
    {
      path: "contact.address.street",
      // The last alternative is fully anchored (whole label only) so a
      // bare "Address"/"Current Address"/"Mailing Address" matches but
      // "Email Address" doesn't accidentally pick up the street path -
      // email's own pattern is checked earlier in this table anyway, but
      // this stays correct even if table order ever changes.
      pattern: /street|address\s*line\s*1|\baddress1\b|^\s*(current|home|mailing|residential)?\s*address\s*$/i,
    },
    { path: "contact.full_name", pattern: /^\s*(your\s+)?(preferred\s+)?(full\s*)?name\s*$/i },
  ];

  const CONFIDENCE_AUTOCOMPLETE = 100;
  const CONFIDENCE_REGEX = 50;

  function textSources(descriptor) {
    return [descriptor.label, descriptor.name, descriptor.id, descriptor.placeholder].filter(Boolean);
  }

  function classifyByAutocomplete(descriptor) {
    const token = (descriptor.autocomplete || "").toLowerCase().trim();
    // autocomplete can be a space-separated list ("section-x shipping tel");
    // the field-type token is the last word.
    const lastToken = token.split(/\s+/).pop();
    const path = AUTOCOMPLETE_MAP[lastToken];
    return path ? { path, confidence: CONFIDENCE_AUTOCOMPLETE, source: "autocomplete" } : null;
  }

  function classifyByRegex(descriptor) {
    const sources = textSources(descriptor);
    for (const rule of REGEX_TABLE) {
      if (rule.fieldTypes && !rule.fieldTypes.includes(descriptor.fieldType)) continue;
      if (sources.some((text) => rule.pattern.test(text))) {
        return { path: rule.path, confidence: CONFIDENCE_REGEX, source: "regex" };
      }
    }
    return null;
  }

  function classify(descriptor) {
    return classifyByAutocomplete(descriptor) || classifyByRegex(descriptor);
  }

  // Paths where several fields on ONE page legitimately hold the same
  // answer, so the one-winner-per-path rule below must not apply. Found
  // live on a Lever form: a location typeahead AND a country dropdown
  // both ask where you are - dropping one (the way duplicate "email"
  // claims are rightly dropped) silently unfilled the dropdown.
  const NON_EXCLUSIVE_PATHS = new Set(["derived.location"]);

  // assignments: [{ descriptor, path, confidence, source }]. Returns the
  // same shape with lower-confidence duplicates on the same path removed
  // (kept assignment marked unchanged; dropped ones are simply absent).
  function rationalize(assignments) {
    const kept = [];
    const bestByPath = new Map();
    for (const a of assignments) {
      if (NON_EXCLUSIVE_PATHS.has(a.path)) {
        kept.push(a);
        continue;
      }
      const existing = bestByPath.get(a.path);
      if (!existing || a.confidence > existing.confidence) {
        bestByPath.set(a.path, a);
      }
    }
    return kept.concat(Array.from(bestByPath.values()));
  }

  function resolveValue(profile, path) {
    if (path === "contact.full_name") {
      const first = profile.contact && profile.contact.first_name;
      const last = profile.contact && profile.contact.last_name;
      return [first, last].filter(Boolean).join(" ");
    }
    // Extrapolated/interpolated answers (years of experience, current
    // employer, education level, EEO defaults, ...) live in
    // shared/derivation.js - a derived path returning undefined simply
    // leaves the field for the cache/LLM tiers, never guesses.
    if (path.startsWith("derived.")) {
      return AF.derive ? AF.derive.derivedValue(profile, path) : undefined;
    }
    let node = profile;
    for (const part of path.split(".")) {
      if (node === null || node === undefined) return undefined;
      node = node[part];
    }
    return node;
  }

  AF.heuristics = { classify, rationalize, resolveValue, AUTOCOMPLETE_MAP, REGEX_TABLE };
})(typeof window !== "undefined" ? window : self);
