// Rule-based (NO LLM) importer: raw resume text + freeform description
// text -> profile JSON per schema.js, with a found/missing report so the
// user can validate what was extracted before saving.
//
// Negation handling is a JS port of NegEx/ConText (Chapman et al. 2001/
// 2007; trigger lists from chapmanbe/negex negex_triggers.txt, adapted
// from the clinical domain to job-application phrasing): pseudo-negation
// checked first, then negation triggers scoped WITHIN a clause - scope
// never crosses a sentence boundary, and conjunction terms ("but",
// "however", "although") terminate it. "No sponsorship needed" negates;
// "authorized to work, but sponsorship required" does not leak the "but"
// clause's polarity backwards.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  // ---------------- sentence + clause segmentation ----------------

  const ABBREV = /\b(?:U\.S|U\.K|e\.g|i\.e|Dr|Mr|Mrs|Ms|St|Jr|Sr|Inc|Ltd|Co)\./g;

  function splitSentences(text) {
    return String(text || "")
      .replace(ABBREV, (m) => m.replace(/\./g, "§"))
      .split(/(?<=[.!?])\s+|\n+/)
      .map((s) => s.replace(/§/g, ".").trim())
      .filter(Boolean);
  }

  // Conjunction/termination terms end a negation's scope (NegEx [CONJ]).
  const CLAUSE_SPLIT = /\b(?:but|however|although|though|except|unless|whereas|;)\b/i;

  function splitClauses(sentence) {
    return sentence.split(CLAUSE_SPLIT).map((c) => c.trim()).filter(Boolean);
  }

  // NegEx [PREN]/[POST] triggers, medical list pruned to generic negation
  // plus job-domain additions (marked): "unwilling", "not willing",
  // "no ... needed" phrasing all reduce to these tokens at clause level.
  const NEGATION_TRIGGERS = [
    "no", "not", "without", "never", "don't", "dont", "doesn't", "doesnt",
    "didn't", "didnt", "cannot", "can't", "cant", "won't", "wont",
    "unable", "unwilling", "n/a", "none",
  ];

  // NegEx [PSEU]: looks like negation, isn't - must suppress a match.
  const PSEUDO_NEGATIONS = [/no problem/i, /no issue/i, /not only/i, /no worries/i];

  function clauseIsNegated(clause) {
    if (PSEUDO_NEGATIONS.some((p) => p.test(clause))) return false;
    const words = clause.toLowerCase().split(/[^a-z0-9'/]+/);
    return NEGATION_TRIGGERS.some((t) => words.includes(t));
  }

  // Finds the clause containing the fact pattern; returns
  // { clause, negated } or null. Highest-scoring sentence wins when the
  // pattern appears in several (per-sentence scoping is what keeps
  // multi-fact descriptions correct).
  function findFactClause(sentences, pattern) {
    for (const sentence of sentences) {
      if (!pattern.test(sentence)) continue;
      for (const clause of splitClauses(sentence)) {
        if (pattern.test(clause)) {
          return { clause, negated: clauseIsNegated(clause) };
        }
      }
    }
    return null;
  }

  // ---------------- description-text fact extraction ----------------

  const REFERRAL_LEXICON = [
    { pattern: /linked\s*in/i, label: "LinkedIn" },
    { pattern: /indeed/i, label: "Indeed" },
    { pattern: /glassdoor/i, label: "Glassdoor" },
    { pattern: /referr|friend|colleague/i, label: "Referral" },
    { pattern: /company\s+(web)?site|career\s+page/i, label: "Company website" },
    { pattern: /job\s+fair|career\s+fair/i, label: "Job fair" },
    { pattern: /recruiter/i, label: "Recruiter" },
    { pattern: /google|search/i, label: "Online search" },
  ];

  function extractFacts(descriptionText) {
    const facts = { work_authorization: {}, preferences: {} };
    const found = {};
    const sentences = splitSentences(descriptionText);
    if (sentences.length === 0) return { facts, found };

    const sponsor = findFactClause(sentences, /\bsponsor(ship)?\b|\bvisa\b/i);
    if (sponsor) {
      // "no sponsorship needed" -> negated -> NOT required;
      // "will require sponsorship" -> affirmed -> required.
      facts.work_authorization.requires_sponsorship = !sponsor.negated;
      found["work_authorization.requires_sponsorship"] = sponsor.clause;
    }

    const auth = findFactClause(
      sentences,
      /authori[sz]ed?\s+to\s+work|legally\s+(entitled|permitted|eligible)|work\s+permit|\bcitizen(ship)?\b|permanent\s+resident|green\s+card/i
    );
    if (auth) {
      facts.work_authorization.authorized = !auth.negated;
      found["work_authorization.authorized"] = auth.clause;
    }

    const salarySentence = sentences.find((s) => /salary|compensation|looking\s+for\s+\$|\$\s*\d/i.test(s));
    if (salarySentence) {
      if (/negotiable|flexible|open\s+(on|about|to discussing)/i.test(salarySentence)) {
        facts.preferences.salary_expectation = "Negotiable";
        found["preferences.salary_expectation"] = salarySentence.trim();
      } else {
        const range = salarySentence.match(
          /(?:\$|cad|usd)?\s*(\d{2,3}(?:,\d{3})?)\s*k?\s*(?:-|–|to)\s*(?:\$|cad|usd)?\s*(\d{2,3}(?:,\d{3})?)\s*k?/i
        );
        const single = salarySentence.match(/(?:\$|cad|usd)\s*(\d{2,3}(?:,\d{3})?)\s*(k)?/i);
        const raw = range ? range[0] : single ? single[0] : null;
        if (raw) {
          facts.preferences.salary_expectation = raw.trim();
          found["preferences.salary_expectation"] = raw.trim();
        }
      }
    }

    const notice = descriptionText.match(/(\d+)\s*[- ]?\s*(day|week|month)s?['’]?\s*notice/i);
    if (notice) {
      facts.preferences.notice_period = `${notice[1]} ${notice[2]}${Number(notice[1]) > 1 ? "s" : ""}`;
      found["preferences.notice_period"] = notice[0];
    } else if (/no\s+notice\s+(period\s+)?(required|needed)|available\s+(to\s+start\s+)?(immediately|asap|right away)/i.test(descriptionText)) {
      facts.preferences.notice_period = "None";
      facts.preferences.earliest_start = "Immediately";
      found["preferences.earliest_start"] = "available immediately";
    }

    const start = descriptionText.match(
      /(?:start(?:ing)?|available)\s*(?:to\s+start\s*)?(?:on|from|by)?\s*([A-Z][a-z]{2,8}\.?,?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)/
    );
    if (start && !facts.preferences.earliest_start) {
      facts.preferences.earliest_start = start[1];
      found["preferences.earliest_start"] = start[0];
    }

    const reloc = findFactClause(sentences, /\breloc\w*/i);
    if (reloc) {
      facts.preferences.willing_to_relocate = !reloc.negated;
      found["preferences.willing_to_relocate"] = reloc.clause;
    }

    // Work mode is an ENUM, not a polarity: pick the category word that
    // co-occurs with a positive-preference verb in an unnegated clause;
    // two affirmed categories = ambiguous = leave blank.
    const modes = [];
    for (const [word, label] of [[/\bremote\b|\bwfh\b/i, "Remote"], [/\bhybrid\b/i, "Hybrid"], [/on-?site|in-?person|in-?office/i, "On-site"]]) {
      const hit = findFactClause(sentences, word);
      if (hit && !hit.negated && /open\s+to|prefer|looking\s+for|want|interested\s+in|only/i.test(hit.clause)) {
        modes.push({ label, clause: hit.clause });
      }
    }
    if (modes.length === 1) {
      facts.preferences.work_mode = modes[0].label;
      found["preferences.work_mode"] = modes[0].clause;
    }

    const license = findFactClause(sentences, /driver.?s?\s+licen[cs]e/i);
    if (license) {
      facts.preferences.drivers_license = !license.negated;
      found["preferences.drivers_license"] = license.clause;
    }

    const referralSentence = sentences.find((s) =>
      /(heard|found|saw|discovered|came\s+across|learned)\b.{0,30}\b(through|via|on|from)\b/i.test(s)
    );
    if (referralSentence) {
      const entry = REFERRAL_LEXICON.find((e) => e.pattern.test(referralSentence));
      if (entry) {
        facts.preferences.referral_source = entry.label;
        found["preferences.referral_source"] = referralSentence.trim();
      }
    }

    return { facts, found };
  }

  // ---------------- resume-text parsing ----------------

  const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/;
  const PHONE_RE = /(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b/;
  const URL_RE = /(?:https?:\/\/)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:\/[^\s|,;)]*)?/gi;

  const SECTION_HEADERS = [
    { key: "experience", pattern: /^(work\s+)?experience$|^employment(\s+history)?$|^work\s+history$|^professional\s+experience$|^relevant\s+experience$/i },
    { key: "education", pattern: /^education$|^academic(\s+background)?$|^education\s*(&|and)\s*training$/i },
    { key: "skills", pattern: /^(technical\s+)?skills$|^technologies$|^core\s+competencies$|^skills\s*(&|and)\s*(tools|interests|abilities)$|^tools$/i },
    { key: "projects", pattern: /^projects$|^personal\s+projects$|^side\s+projects$/i },
    { key: "summary", pattern: /^summary$|^objective$|^profile$|^about(\s+me)?$/i },
    { key: "other", pattern: /^certifications?$|^awards?$|^publications?$|^volunteer(ing)?$|^interests$|^languages$|^references?$/i },
  ];

  // OpenResume's fallback section-title rule (group-lines-into-sections):
  // a short, letters-only line containing a section keyword counts as a
  // header even when the anchored patterns above miss the exact phrasing
  // ("EXPERIENCE AND LEADERSHIP", "Skills Summary").
  const FALLBACK_SECTION_KEYWORDS = [
    ["experience", "experience"], ["employment", "experience"], ["work history", "experience"],
    ["education", "education"], ["academic", "education"],
    ["skill", "skills"], ["technologies", "skills"], ["competenc", "skills"],
    ["project", "projects"],
    ["summary", "summary"], ["objective", "summary"], ["profile", "summary"],
    ["certification", "other"], ["award", "other"], ["publication", "other"],
    ["volunteer", "other"], ["language", "other"], ["interest", "other"], ["reference", "other"],
  ];

  function classifyHeader(line) {
    const trimmed = line.trim().replace(/[:–-]+$/, "").trim();
    if (trimmed.length === 0 || trimmed.length > 40) return null;
    for (const h of SECTION_HEADERS) {
      if (h.pattern.test(trimmed)) return h.key;
    }
    if (/^[A-Za-z\s&]+$/.test(trimmed) && trimmed.split(/\s+/).length <= 4) {
      const lower = trimmed.toLowerCase();
      for (const [keyword, key] of FALLBACK_SECTION_KEYWORDS) {
        if (lower.includes(keyword)) return key;
      }
    }
    return null;
  }

  // "May 2023 - Present", "05/2023-11/2025", "2023 - 2025" (hyphen,
  // en/em dash), "May 2023 to Present".
  const DATE_TOKEN = String.raw`(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}|\d{1,2}\/\d{4}|\d{4}-\d{2}|\d{4})`;
  const DATE_RANGE_RE = new RegExp(
    `(${DATE_TOKEN})\\s*(?:-|\\u2013|\\u2014|to|\\u2192)\\s*(${DATE_TOKEN}|Present|Current|Now|Ongoing)`,
    "i"
  );

  function looksLikeName(line) {
    const t = line.trim();
    if (!t || /\d|@|http|www\./.test(t)) return false;
    const words = t.split(/\s+/);
    if (words.length < 2 || words.length > 4) return false;
    return words.every((w) => /^[A-Z][A-Za-z'.-]*$/.test(w));
  }

  function splitNameParts(fullName) {
    const SUFFIXES = new Set(["jr", "sr", "ii", "iii", "iv", "v", "phd", "md"]);
    const words = fullName.trim().split(/\s+/).filter((w) => !SUFFIXES.has(w.replace(/[.,]/g, "").toLowerCase()));
    if (words.length === 0) return { first: "", last: "" };
    if (words.length === 1) return { first: words[0], last: "" };
    return { first: words[0], last: words[words.length - 1] };
  }

  function classifyUrls(text) {
    const out = { linkedin: "", github: "", website: "" };
    for (const raw of text.match(URL_RE) || []) {
      const url = raw.replace(/[.,;]$/, "");
      // The dotted-token regex also catches numerics ("3.8/4.0" GPAs,
      // version numbers) - a real host ends in an alphabetic TLD.
      if (!/\.[a-z]{2,}$/i.test(url.split("/")[0])) continue;
      if (/linkedin\.com/i.test(url) && !out.linkedin) out.linkedin = url;
      else if (/github\.com/i.test(url) && !out.github) out.github = url;
      else if (!/@/.test(url) && !out.website && !/linkedin|github/i.test(url)) {
        // Skip bare email-domain fragments the URL regex can shave off
        // an email address; require a path or a non-generic TLD signal.
        if (!EMAIL_RE.test(text) || !text.includes(`@${url.split("/")[0]}`)) out.website = url;
      }
    }
    return out;
  }

  // "City, ST" / "City, Province, Country" on a contact line.
  function extractLocation(headLines) {
    for (const line of headLines) {
      const m = line.match(
        /([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+)?),\s*([A-Z]{2}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:,\s*([A-Z][A-Za-z ]+))?/
      );
      if (m && !EMAIL_RE.test(m[0]) && !/university|college|institute|school|inc|llc|ltd/i.test(m[0])) {
        return { city: m[1], region: m[2], country: m[3] || "" };
      }
    }
    return null;
  }

  function segmentSections(lines) {
    const sections = { _head: [] };
    let current = "_head";
    for (const line of lines) {
      const key = classifyHeader(line);
      if (key) {
        current = key;
        if (!sections[current]) sections[current] = [];
        continue;
      }
      if (!sections[current]) sections[current] = [];
      sections[current].push(line);
    }
    return sections;
  }

  // Entry grouping: a date-range line anchors an entry; the entry's
  // company/title come from the date line itself plus the non-bullet
  // line(s) immediately around it. Bullets (-, *, •) are skipped.
  function parseEntries(sectionLines) {
    const entries = [];
    for (let i = 0; i < sectionLines.length; i += 1) {
      const line = sectionLines[i];
      const range = line.match(DATE_RANGE_RE);
      if (!range) continue;
      const headParts = [];
      const before = line.replace(DATE_RANGE_RE, "").replace(/[|–-]\s*$/, "").trim();
      if (before) headParts.push(before);
      // A date on its own line names the entry on the previous line(s).
      // Bullet-glyph class from OpenResume's bullet-points.ts, plus the
      // ASCII substitutes PDF-to-text extractors commonly emit.
      for (let back = 1; back <= 2 && headParts.join("").length === 0; back += 1) {
        const prev = sectionLines[i - back];
        if (prev && !DATE_RANGE_RE.test(prev) && !/^\s*[-*•⋅∙●○⦁▪♦>·]/.test(prev)) headParts.push(prev.trim());
      }
      entries.push({ head: headParts.join(" | "), start: range[1], end: /present|current|now|ongoing/i.test(range[2]) ? "" : range[2] });
    }
    return entries;
  }

  // "Title, Company" / "Title at Company" / "Company - Title" /
  // "Title | Company": try the explicit separators in order. Word list
  // adapted from OpenResume's JOB_TITLES (extract-work-experience.ts) -
  // the side of a separator containing one of these is the title, the
  // other side is the company.
  const TITLE_WORDS = /engineer|developer|analyst|manager|intern|co-?op|extern|designer|scientist|consultant|specialist|director|lead|head|assistant|associate|coordinator|administrator|technician|architect|founder|president|officer|researcher|teacher|tutor|volunteer|representative|supervisor|clerk|operator|advisor|\bvp\b|\bcto\b|\bceo\b/i;

  function splitTitleCompany(head) {
    const cleaned = head.replace(/\s*\|\s*$/, "").trim();
    const bySep = cleaned.split(/\s+at\s+|\s*[|]\s*|\s*[,–]\s+|\s+-\s+/).map((p) => p.trim()).filter(Boolean);
    if (bySep.length >= 2) {
      const titleIdx = bySep.findIndex((p) => TITLE_WORDS.test(p));
      if (titleIdx >= 0) {
        const title = bySep[titleIdx];
        const company = bySep.find((_, idx) => idx !== titleIdx) || "";
        return { title, company };
      }
      return { title: bySep[0], company: bySep[1] };
    }
    if (TITLE_WORDS.test(cleaned)) return { title: cleaned, company: "" };
    return { title: "", company: cleaned };
  }

  const DEGREE_RE = /\b(ph\.?d|doctorate|m\.?b\.?a|m\.?s\.?c?|m\.?eng|master(?:'?s)?|b\.?s\.?c?|b\.?eng|b\.?a\.?s?c?|bachelor(?:'?s)?|associate(?:'?s)?|diploma|high\s+school)\b[^,|\n]*/i;
  const SCHOOL_RE = /\b([A-Z][A-Za-z.&' -]*(?:University|College|Institute|Polytechnic|School)(?:\s+of\s+[A-Z][A-Za-z ]+)?)\b|\b(University|College|Institute)\s+of\s+([A-Z][A-Za-z ]+)\b/;

  function parseEducation(sectionLines) {
    const entries = parseEntries(sectionLines);
    const text = sectionLines.join("\n");
    const out = [];
    const source = entries.length > 0 ? entries : [{ head: text.split("\n").slice(0, 3).join(" | "), start: "", end: "" }];
    for (const entry of source) {
      const scope = entry.head || text;
      const schoolMatch = scope.match(SCHOOL_RE) || text.match(SCHOOL_RE);
      const degreeMatch = scope.match(DEGREE_RE) || text.match(DEGREE_RE);
      if (!schoolMatch && !degreeMatch) continue;
      // OpenResume's GPA rule: 0-4 scale with 1-2 decimals.
      const gpaMatch = text.match(/gpa[:\s]*([0-4]\.\d{1,2})/i) || scope.match(/\b([0-4]\.\d{1,2})\s*\/\s*4/);
      out.push({
        school: schoolMatch ? (schoolMatch[1] || `${schoolMatch[2]} of ${schoolMatch[3]}`).trim() : "",
        degree: degreeMatch ? degreeMatch[0].trim() : "",
        field: "",
        start: entry.start,
        end: entry.end,
        gpa: gpaMatch ? gpaMatch[1] : "",
      });
    }
    return out;
  }

  function parseSkills(sectionLines) {
    const raw = sectionLines
      .join("\n")
      .replace(/^[-*•]\s*/gm, "")
      .replace(/^[A-Za-z &]+:\s*/gm, "") // "Languages: Python, C" category prefixes
      .split(/[,|•;\n]+/)
      .map((s) => s.trim())
      .filter((s) => s && s.length <= 40);
    return Array.from(new Set(raw));
  }

  function parseResume(resumeText) {
    const found = {};
    const lines = String(resumeText || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const fullText = lines.join("\n");
    const sections = segmentSections(lines);
    const contact = {};

    const email = fullText.match(EMAIL_RE);
    if (email) {
      contact.email = email[0];
      found["contact.email"] = email[0];
    }
    const phone = fullText.match(PHONE_RE);
    if (phone) {
      contact.phone = phone[0].trim();
      found["contact.phone"] = phone[0].trim();
    }
    const urls = classifyUrls(fullText);
    for (const key of ["linkedin", "github", "website"]) {
      if (urls[key]) {
        contact[key] = urls[key];
        found[`contact.${key}`] = urls[key];
      }
    }

    const nameLine = lines.slice(0, 5).find(looksLikeName);
    if (nameLine) {
      const { first, last } = splitNameParts(nameLine);
      contact.first_name = first;
      contact.last_name = last;
      found["contact.name"] = nameLine.trim();
    }

    const location = extractLocation((sections._head || []).slice(0, 6));
    if (location) {
      contact.address = { city: location.city, region: location.region, country: location.country };
      found["contact.address"] = [location.city, location.region, location.country].filter(Boolean).join(", ");
    }

    const experience = [];
    for (const entry of parseEntries(sections.experience || [])) {
      const { title, company } = splitTitleCompany(entry.head);
      if (!title && !company) continue;
      experience.push({ company, title, start: entry.start, end: entry.end });
    }
    if (experience.length > 0) {
      found["experience"] = `${experience.length} entr${experience.length === 1 ? "y" : "ies"} (${experience.map((e) => e.company || e.title).join("; ")})`;
    }

    const education = parseEducation(sections.education || []);
    if (education.length > 0) {
      found["education"] = education.map((e) => `${e.degree || "?"} @ ${e.school || "?"}`).join("; ");
    }

    const skills = parseSkills(sections.skills || []);
    if (skills.length > 0) found["skills"] = `${skills.length} skills`;

    return { contact, experience, education, skills, found };
  }

  // ---------------- top-level import ----------------

  const REPORT_FIELDS = [
    "contact.name", "contact.email", "contact.phone", "contact.address",
    "contact.linkedin", "experience", "education", "skills",
  ];

  function importProfile(resumeText, descriptionText, profileId) {
    const resume = parseResume(resumeText);
    const description = extractFacts(descriptionText);

    const profile = AF.schema.emptyProfile(profileId || "imported");
    Object.assign(profile.contact, resume.contact);
    if (resume.contact.address) Object.assign(profile.contact.address, resume.contact.address);
    profile.contact.address = { ...AF.schema.emptyProfile("x").contact.address, ...(resume.contact.address || {}) };
    profile.experience = resume.experience;
    profile.education = resume.education;
    profile.skills = resume.skills;
    Object.assign(profile.work_authorization, description.facts.work_authorization);
    Object.assign(profile.preferences, description.facts.preferences);

    const foundAll = { ...resume.found, ...description.found };
    const missing = REPORT_FIELDS.filter((f) => !(f in foundAll));
    const warnings = [];
    if (!resumeText || !resumeText.trim()) warnings.push("no resume text given - contact/experience/education left empty");
    if (descriptionText && descriptionText.trim() && Object.keys(description.found).length === 0) {
      warnings.push("description text yielded no facts - work auth / preferences left unanswered");
    }

    return { profile, report: { found: foundAll, missing, warnings } };
  }

  AF.importer = {
    importProfile,
    parseResume,
    extractFacts,
    splitSentences,
    clauseIsNegated,
  };
})(typeof window !== "undefined" ? window : self);
