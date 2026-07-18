// Tier 2: derives answers the profile doesn't hold literally
// (extrapolation/interpolation), plus fuzzy option matching for
// select/listbox/radio option lists. Two hard rules keep this layer
// honest:
//   1. Derive only what follows mechanically from stored facts (dates ->
//      years of experience, degree strings -> education level, address
//      parts -> location string). Anything judgment-shaped is REFUSED and
//      left for the cache/LLM tiers or the human: salary/rate,
//      start date/notice period, age/DOB, references, essay questions.
//   2. A fuzzy match must clear a threshold AND be strictly better than
//      the runner-up - a tie means ambiguity, and ambiguity means leave
//      the field blank rather than guess.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  // ---------------- text normalization + fuzzy matching ----------------

  const STOPWORDS = new Set(["a", "an", "the", "of", "or", "and", "to", "in", "s"]);

  function normalize(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  // Canonical degree tokens so "BSc" / "Bachelor of Engineering" /
  // "Bachelor's Degree" all share a token.
  function canonicalToken(tok) {
    if (/^(bachelor|bachelors|bs|bsc|ba|beng|bse|bcomm|undergraduate)$/.test(tok)) return "bachelors";
    if (/^(master|masters|ms|msc|ma|meng|mba|graduate)$/.test(tok)) return "masters";
    if (/^(phd|doctorate|doctoral|dphil)$/.test(tok)) return "phd";
    if (/^(associate|associates)$/.test(tok)) return "associates";
    return tok;
  }

  function tokens(text) {
    return normalize(text)
      .split(" ")
      .filter((t) => t && !STOPWORDS.has(t))
      .map(canonicalToken);
  }

  // Answers that mean the same thing across forms, matched as a CLASS
  // rather than by string distance. EEO "decline" phrasings vary wildly
  // but always contain one of these markers.
  const DECLINE_MARKERS = [
    "decline",
    "prefer not",
    "not wish to",
    "dont wish",
    "don t wish",
    "rather not say",
    "choose not to",
  ];

  function isDeclineText(text) {
    const norm = normalize(text);
    return DECLINE_MARKERS.some((m) => norm.includes(normalize(m)));
  }

  function jaccard(aTokens, bTokens) {
    if (aTokens.length === 0 || bTokens.length === 0) return 0;
    const a = new Set(aTokens);
    const b = new Set(bTokens);
    let inter = 0;
    for (const t of a) if (b.has(t)) inter += 1;
    return inter / (a.size + b.size - inter);
  }

  function scoreOption(targetTokens, optionText) {
    const optTokens = tokens(optionText);
    if (optTokens.length === 0) return 0;
    const targetSet = new Set(targetTokens);
    const optSet = new Set(optTokens);
    if (targetTokens.length > 0 && targetTokens.join(" ") === optTokens.join(" ")) return 1;
    const base = jaccard(targetTokens, optTokens);
    const targetInOpt = targetTokens.every((t) => optSet.has(t));
    const optInTarget = optTokens.every((t) => targetSet.has(t));
    // Containment either way is a strong signal ("Canada" against
    // "Toronto, Ontario, Canada"; "Bachelor's" against "Bachelor's Degree")
    // - but scaled by overlap so the SHORTEST containing option wins
    // (Chromium's rule: "United States" must beat "United States Minor
    // Outlying Islands" instead of tying with it).
    if (targetInOpt || optInTarget) return 0.6 + 0.4 * base;
    return base;
  }

  // "2-4 years" / "5+ years" / "Less than 1 year" style buckets: string
  // distance is meaningless here ("2-4 years" vs "4-6 years" differ by
  // one character) - parse the numbers and do range containment instead.
  function matchNumericBucket(options, numericValue) {
    const v = Number(numericValue);
    if (!Number.isFinite(v)) return null;
    for (const opt of options) {
      // Raw lowercase, NOT normalize(): normalization strips punctuation
      // and "2-4 years" must keep its hyphen for the range regex.
      const text = String(opt.text || "").toLowerCase().trim();
      let m = text.match(/(\d+)\s*(?:-|to)\s*(\d+)/);
      if (m && v >= Number(m[1]) && v <= Number(m[2])) return opt;
      m = text.match(/(\d+)\s*\+|more than (\d+)|(\d+) or more/);
      if (m && v >= Number(m[1] || m[2] || m[3])) return opt;
      m = text.match(/less than (\d+)|under (\d+)/);
      if (m && v < Number(m[1] || m[2])) return opt;
      m = text.match(/^(\d+)(\s*years?)?$/);
      if (m && v === Number(m[1])) return opt;
    }
    return null;
  }

  const FUZZY_THRESHOLD = 0.5;

  // options: [{ value, text }]. value (the profile-side answer) may be a
  // string or boolean. Returns the matched option or null. Deliberately
  // returns null on ties - see header rule 2.
  function fuzzyMatchOption(options, value) {
    if (value === undefined || value === null || value === "") return null;

    if (typeof value === "string" && isDeclineText(value)) {
      const decline = options.find((o) => isDeclineText(o.text) || isDeclineText(o.value));
      return decline || null;
    }

    if (/^\d+(\.\d+)?$/.test(String(value).trim())) {
      const bucket = matchNumericBucket(options, value);
      if (bucket) return bucket;
      // A pure number that fits no bucket must not fall through to token
      // scoring ("3" has no meaningful token overlap with anything).
      return null;
    }

    const targetTokens = tokens(String(value));
    if (targetTokens.length === 0) return null;

    let best = null;
    let bestScore = 0;
    let runnerUpScore = 0;
    for (const opt of options) {
      if (isDeclineText(opt.text)) continue; // never fuzzy-drift into a decline option
      const s = scoreOption(targetTokens, opt.text);
      if (s > bestScore) {
        runnerUpScore = bestScore;
        bestScore = s;
        best = opt;
      } else if (s > runnerUpScore) {
        runnerUpScore = s;
      }
    }
    if (!best || bestScore < FUZZY_THRESHOLD) return null;
    if (bestScore === runnerUpScore) return null; // ambiguous - leave blank
    return best;
  }

  // ---------------- derived profile values ----------------

  const MONTHS = {
    jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
    jul: 6, aug: 7, sep: 8, sept: 8, oct: 9, nov: 10, dec: 11,
  };

  // Accepts "2024-05", "05/2024", "May 2024", "2024"; "present"-like
  // words mean now. Returns a Date or null.
  function parseDateLoose(raw) {
    if (raw === undefined || raw === null) return null;
    const s = String(raw).trim().toLowerCase();
    if (s === "") return null;
    if (/present|current|now|ongoing|today/.test(s)) return new Date();
    let m = s.match(/^(\d{4})[-/](\d{1,2})/);
    if (m) return new Date(Number(m[1]), Number(m[2]) - 1, 1);
    m = s.match(/^(\d{1,2})[-/](\d{4})$/);
    if (m) return new Date(Number(m[2]), Number(m[1]) - 1, 1);
    m = s.match(/^([a-z]{3,9})\.?,?\s+(\d{4})$/);
    if (m && MONTHS[m[1].slice(0, 3)] !== undefined) {
      return new Date(Number(m[2]), MONTHS[m[1].slice(0, 3)], 1);
    }
    m = s.match(/^(\d{4})$/);
    if (m) return new Date(Number(m[1]), 0, 1);
    return null;
  }

  function experienceIntervals(profile) {
    const out = [];
    for (const job of profile.experience || []) {
      const start = parseDateLoose(job.start);
      if (!start) continue;
      const end = parseDateLoose(job.end) || new Date();
      if (end > start) out.push([start.getTime(), end.getTime()]);
    }
    return out.sort((a, b) => a[0] - b[0]);
  }

  // Total years across MERGED intervals - two overlapping jobs are not
  // double experience.
  function totalExperienceYears(profile) {
    const intervals = experienceIntervals(profile);
    if (intervals.length === 0) return undefined;
    let total = 0;
    let [curStart, curEnd] = intervals[0];
    for (const [s, e] of intervals.slice(1)) {
      if (s <= curEnd) {
        curEnd = Math.max(curEnd, e);
      } else {
        total += curEnd - curStart;
        [curStart, curEnd] = [s, e];
      }
    }
    total += curEnd - curStart;
    const years = total / (365.25 * 24 * 3600 * 1000);
    // Round DOWN: overclaiming experience on an application is worse than
    // underclaiming by a few months.
    return String(Math.floor(years));
  }

  function mostRecentExperience(profile) {
    const jobs = (profile.experience || []).filter((j) => j && (j.company || j.title));
    if (jobs.length === 0) return null;
    const ongoing = jobs.find((j) => {
      const end = String(j.end || "").toLowerCase();
      return end === "" || /present|current|now|ongoing/.test(end);
    });
    if (ongoing) return ongoing;
    let best = jobs[0];
    let bestEnd = parseDateLoose(best.end) || new Date(0);
    for (const j of jobs.slice(1)) {
      const end = parseDateLoose(j.end) || new Date(0);
      if (end > bestEnd) {
        best = j;
        bestEnd = end;
      }
    }
    return best;
  }

  const DEGREE_LEVELS = [
    { level: 5, pattern: /phd|doctor/i, label: "Doctorate" },
    { level: 4, pattern: /master|\bmba\b|\bmsc?\b|\bmeng\b|\bma\b/i, label: "Master's degree" },
    { level: 3, pattern: /bachelor|\bbsc?\b|\bbeng\b|\bba\b|undergrad/i, label: "Bachelor's degree" },
    { level: 2, pattern: /associate/i, label: "Associate degree" },
    { level: 1, pattern: /diploma|certificate/i, label: "Diploma" },
    { level: 0, pattern: /high\s*school|secondary/i, label: "High school or equivalent" },
  ];

  function highestDegree(profile) {
    let best = null;
    for (const edu of profile.education || []) {
      const text = `${edu.degree || ""} ${edu.field || ""}`;
      for (const d of DEGREE_LEVELS) {
        if (d.pattern.test(text) && (!best || d.level > best.level)) best = d;
      }
    }
    return best ? best.label : undefined;
  }

  function graduationYear(profile) {
    let latest;
    for (const edu of profile.education || []) {
      const end = parseDateLoose(edu.end);
      if (end && (!latest || end > latest)) latest = end;
    }
    return latest ? String(latest.getFullYear()) : undefined;
  }

  function locationString(profile) {
    const addr = (profile.contact && profile.contact.address) || {};
    const parts = [addr.city, addr.region, addr.country].filter(Boolean);
    return parts.length > 0 ? parts.join(", ") : undefined;
  }

  function nonEmpty(v) {
    return v === undefined || v === null || v === "" ? undefined : v;
  }

  function derivedValue(profile, path) {
    const recent = () => mostRecentExperience(profile) || {};
    switch (path) {
      case "derived.current_company":
        return nonEmpty(recent().company);
      case "derived.current_title":
        return nonEmpty(recent().title);
      case "derived.years_experience":
        return totalExperienceYears(profile);
      case "derived.highest_degree":
        return highestDegree(profile);
      case "derived.school": {
        const edu = (profile.education || [])[0] || {};
        return nonEmpty(edu.school);
      }
      case "derived.graduation_year":
        return graduationYear(profile);
      case "derived.location":
        return locationString(profile);
      // "Are you authorized to work in X WITHOUT sponsorship?" - a
      // compound question, live on a Lever posting in this repo's test
      // set. Neither stored boolean alone answers it: yes only if
      // authorized AND not needing sponsorship. Shipped bots that route
      // this by first-keyword-hit ('sponsor') answer it backwards.
      case "derived.authorized_without_sponsorship": {
        const wa = profile.work_authorization || {};
        if (wa.authorized === null || wa.authorized === undefined) return undefined;
        if (wa.requires_sponsorship === null || wa.requires_sponsorship === undefined) {
          return undefined;
        }
        return Boolean(wa.authorized) && !wa.requires_sponsorship;
      }
      case "derived.eeo.gender":
      case "derived.eeo.race":
      case "derived.eeo.veteran_status":
      case "derived.eeo.disability": {
        const key = path.split(".")[2];
        const defaults = profile.eeo_defaults || {};
        return nonEmpty(defaults[key]) || "decline";
      }
      default:
        return undefined;
    }
  }

  AF.derive = {
    derivedValue,
    fuzzyMatchOption,
    isDeclineText,
    tokens,
    normalize,
    parseDateLoose,
    totalExperienceYears,
  };
})(typeof window !== "undefined" ? window : self);
