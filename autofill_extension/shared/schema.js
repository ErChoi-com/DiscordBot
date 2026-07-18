// Applicant profile schema + validation. Loaded as a plain classic script
// (not an ES module) so the same file works unchanged whether it's
// injected into a page as a content script or loaded by a plain <script>
// tag in the popup/options pages. Everything hangs off globalThis.AF.
(function (global) {
  "use strict";

  const AF = (global.AF = global.AF || {});

  function emptyProfile(profileId) {
    return {
      profile_id: profileId || "default",
      contact: {
        first_name: "",
        last_name: "",
        email: "",
        phone: "",
        address: {
          street: "",
          city: "",
          region: "",
          postal_code: "",
          country: "",
        },
        linkedin: "",
        github: "",
        website: "",
      },
      work_authorization: {
        authorized: null,
        requires_sponsorship: null,
      },
      // Facts a form may ask that only the user can state (never derived
      // or guessed) - populated by hand or by the resume/description
      // importer in shared/profile_import.js. Empty/null means "don't
      // answer", never "answer something plausible".
      preferences: {
        salary_expectation: "",
        notice_period: "",
        earliest_start: "",
        willing_to_relocate: null,
        work_mode: "",
        referral_source: "",
        drivers_license: null,
      },
      education: [],
      experience: [],
      skills: [],
      eeo_defaults: {
        gender: "decline",
        veteran_status: "decline",
        race: "decline",
        disability: "decline",
      },
    };
  }

  // Returns { ok: true, profile } or { ok: false, errors: [...] }. Tolerant
  // by design: fills in missing sections rather than rejecting a profile
  // that's mid-edit, but flags anything that isn't the expected shape.
  function validateProfile(raw) {
    const errors = [];
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
      return { ok: false, errors: ["Profile must be a JSON object."] };
    }

    const base = emptyProfile(typeof raw.profile_id === "string" ? raw.profile_id : "default");

    if (raw.contact && typeof raw.contact === "object") {
      Object.assign(base.contact, raw.contact);
      if (raw.contact.address && typeof raw.contact.address === "object") {
        Object.assign(base.contact.address, raw.contact.address);
      }
    } else if (raw.contact !== undefined) {
      errors.push("contact must be an object.");
    }

    if (raw.work_authorization && typeof raw.work_authorization === "object") {
      Object.assign(base.work_authorization, raw.work_authorization);
    } else if (raw.work_authorization !== undefined) {
      errors.push("work_authorization must be an object.");
    }

    if (raw.preferences && typeof raw.preferences === "object") {
      Object.assign(base.preferences, raw.preferences);
    } else if (raw.preferences !== undefined) {
      errors.push("preferences must be an object.");
    }

    for (const key of ["education", "experience", "skills"]) {
      if (raw[key] === undefined) continue;
      if (!Array.isArray(raw[key])) {
        errors.push(`${key} must be an array.`);
        continue;
      }
      base[key] = raw[key];
    }

    if (raw.eeo_defaults && typeof raw.eeo_defaults === "object") {
      Object.assign(base.eeo_defaults, raw.eeo_defaults);
    } else if (raw.eeo_defaults !== undefined) {
      errors.push("eeo_defaults must be an object.");
    }

    return { ok: errors.length === 0, profile: base, errors };
  }

  AF.schema = { emptyProfile, validateProfile };
})(typeof window !== "undefined" ? window : self);
