(function () {
  "use strict";

  const profileListEl = document.getElementById("profileList");
  const profileIdInput = document.getElementById("profileIdInput");
  const jsonTextarea = document.getElementById("profileJson");
  const validationMessage = document.getElementById("validationMessage");
  const savedNotice = document.getElementById("savedNotice");

  let profiles = {};
  let selectedId = null;

  function showValidationError(text) {
    validationMessage.textContent = text;
    validationMessage.hidden = !text;
  }

  function flashSaved() {
    savedNotice.hidden = false;
    setTimeout(() => {
      savedNotice.hidden = true;
    }, 1500);
  }

  function renderList() {
    profileListEl.innerHTML = "";
    const ids = Object.keys(profiles).sort();
    for (const id of ids) {
      const li = document.createElement("li");
      li.className = "profile-list__item" + (id === selectedId ? " profile-list__item--active" : "");
      li.textContent = id;
      li.addEventListener("click", () => selectProfile(id));
      profileListEl.appendChild(li);
    }
  }

  function selectProfile(id) {
    selectedId = id;
    profileIdInput.value = id;
    jsonTextarea.value = JSON.stringify(profiles[id], null, 2);
    showValidationError("");
    renderList();
  }

  async function loadAll() {
    const stored = await chrome.storage.local.get("af_profiles");
    profiles = stored.af_profiles || {};
    const ids = Object.keys(profiles);
    if (ids.length > 0) {
      selectProfile(ids[0]);
    } else {
      renderList();
    }
  }

  async function persistProfiles() {
    await chrome.storage.local.set({ af_profiles: profiles });
    const stored = await chrome.storage.local.get("af_active_profile_id");
    if (!stored.af_active_profile_id || !profiles[stored.af_active_profile_id]) {
      const firstId = Object.keys(profiles)[0];
      if (firstId) await chrome.storage.local.set({ af_active_profile_id: firstId });
    }
  }

  document.getElementById("newProfileButton").addEventListener("click", () => {
    selectedId = null;
    profileIdInput.value = "";
    jsonTextarea.value = JSON.stringify(AF.schema.emptyProfile(""), null, 2);
    showValidationError("");
    renderList();
    profileIdInput.focus();
  });

  document.getElementById("loadTemplateButton").addEventListener("click", () => {
    const id = profileIdInput.value.trim() || "default";
    jsonTextarea.value = JSON.stringify(AF.schema.emptyProfile(id), null, 2);
    showValidationError("");
  });

  document.getElementById("saveButton").addEventListener("click", async () => {
    const id = profileIdInput.value.trim();
    if (!id) {
      showValidationError("Profile ID can't be empty.");
      return;
    }

    let parsed;
    try {
      parsed = JSON.parse(jsonTextarea.value);
    } catch (err) {
      showValidationError(`Invalid JSON: ${err.message}`);
      return;
    }

    const result = AF.schema.validateProfile(parsed);
    if (!result.ok) {
      showValidationError(result.errors.join("\n"));
      return;
    }

    if (selectedId && selectedId !== id) {
      delete profiles[selectedId];
    }
    result.profile.profile_id = id;
    profiles[id] = result.profile;
    selectedId = id;

    await persistProfiles();
    showValidationError("");
    renderList();
    flashSaved();
  });

  document.getElementById("deleteButton").addEventListener("click", async () => {
    if (!selectedId || !profiles[selectedId]) return;
    delete profiles[selectedId];
    selectedId = null;
    await persistProfiles();
    const remaining = Object.keys(profiles);
    if (remaining.length > 0) {
      selectProfile(remaining[0]);
    } else {
      profileIdInput.value = "";
      jsonTextarea.value = "";
      renderList();
    }
  });

  // --- Resume/description importer: rule-based, local, no LLM. The
  // parsed result is deliberately routed THROUGH the JSON editor rather
  // than saved directly - the user reviews what the parser found (and
  // the report of what it didn't) before anything persists.
  const importReport = document.getElementById("importReport");
  document.getElementById("importParseButton").addEventListener("click", () => {
    const resumeText = document.getElementById("importResumeText").value;
    const descriptionText = document.getElementById("importDescriptionText").value;
    if (!resumeText.trim() && !descriptionText.trim()) {
      importReport.textContent = "Nothing to parse - paste resume text first.";
      importReport.hidden = false;
      return;
    }
    const id = profileIdInput.value.trim() || "imported";
    const { profile, report } = AF.importer.importProfile(resumeText, descriptionText, id);
    profileIdInput.value = id;
    jsonTextarea.value = JSON.stringify(profile, null, 2);
    const lines = [];
    for (const [field, snippet] of Object.entries(report.found)) {
      lines.push(`  FOUND ${field}: ${snippet}`);
    }
    for (const field of report.missing) {
      lines.push(`  MISSING ${field}`);
    }
    for (const warning of report.warnings) {
      lines.push(`  NOTE ${warning}`);
    }
    importReport.textContent =
      "Parsed. Review the Profile JSON above, then Save.\n" + lines.join("\n");
    importReport.hidden = false;
    showValidationError("");
  });

  const llmFields = {
    gemini_api_key: document.getElementById("geminiKey"),
    gemini_model: document.getElementById("geminiModel"),
    openrouter_api_key: document.getElementById("openrouterKey"),
    openrouter_model: document.getElementById("openrouterModel"),
    groq_api_key: document.getElementById("groqKey"),
    groq_model: document.getElementById("groqModel"),
  };
  const llmSavedNotice = document.getElementById("llmSavedNotice");

  async function loadLlmConfig() {
    const stored = await chrome.storage.local.get("af_llm_config");
    const config = stored.af_llm_config || {};
    for (const [key, input] of Object.entries(llmFields)) {
      input.value = config[key] || "";
    }
  }

  document.getElementById("saveLlmConfigButton").addEventListener("click", async () => {
    const config = {};
    for (const [key, input] of Object.entries(llmFields)) {
      config[key] = input.value.trim();
    }
    await chrome.storage.local.set({ af_llm_config: config });
    llmSavedNotice.hidden = false;
    setTimeout(() => {
      llmSavedNotice.hidden = true;
    }, 1500);
  });

  loadAll();
  loadLlmConfig();
})();
