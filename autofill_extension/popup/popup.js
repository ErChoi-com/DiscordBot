(async function () {
  "use strict";

  const emptyState = document.getElementById("emptyState");
  const mainState = document.getElementById("mainState");
  const profileSelect = document.getElementById("profileSelect");
  const fillButton = document.getElementById("fillButton");
  const resultPanel = document.getElementById("resultPanel");
  const filledCount = document.getElementById("filledCount");
  const skippedCount = document.getElementById("skippedCount");
  const skippedList = document.getElementById("skippedList");
  const statusMessage = document.getElementById("statusMessage");

  function showStatus(text) {
    statusMessage.textContent = text;
    statusMessage.hidden = !text;
  }

  async function getStoredProfiles() {
    const stored = await chrome.storage.local.get(["af_profiles", "af_active_profile_id"]);
    return { profiles: stored.af_profiles || {}, activeId: stored.af_active_profile_id };
  }

  async function init() {
    const { profiles, activeId } = await getStoredProfiles();
    const ids = Object.keys(profiles);

    if (ids.length === 0) {
      emptyState.hidden = false;
      mainState.hidden = true;
      return;
    }

    emptyState.hidden = true;
    mainState.hidden = false;

    profileSelect.innerHTML = "";
    for (const id of ids) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = id;
      profileSelect.appendChild(option);
    }
    profileSelect.value = ids.includes(activeId) ? activeId : ids[0];
    if (profileSelect.value !== activeId) {
      await chrome.storage.local.set({ af_active_profile_id: profileSelect.value });
    }
  }

  profileSelect.addEventListener("change", async () => {
    await chrome.storage.local.set({ af_active_profile_id: profileSelect.value });
  });

  document.getElementById("openOptions").addEventListener("click", () => {
    chrome.runtime.openOptionsPage();
  });

  document.getElementById("emptyStateSetup").addEventListener("click", () => {
    chrome.runtime.openOptionsPage();
  });

  fillButton.addEventListener("click", async () => {
    fillButton.disabled = true;
    fillButton.textContent = "Filling…";
    resultPanel.hidden = true;
    showStatus("");

    try {
      const response = await chrome.runtime.sendMessage({ type: "FILL_ACTIVE_TAB" });
      if (!response || !response.ok) {
        showStatus((response && response.error) || "Couldn't fill this page.");
      } else {
        const { summary } = response;
        filledCount.textContent = String(summary.filled);
        skippedCount.textContent = String(summary.skipped);
        skippedList.innerHTML = "";
        const skippedFields = summary.fields.filter((f) => !f.ok);
        for (const field of skippedFields) {
          const li = document.createElement("li");
          li.textContent = field.label || `(unlabeled ${field.fieldType})`;
          skippedList.appendChild(li);
        }
        resultPanel.hidden = false;
        if (summary.total === 0) {
          showStatus("No fillable fields found on this page.");
        }
      }
    } catch (err) {
      showStatus("Couldn't reach this page. Try reloading it.");
    } finally {
      fillButton.disabled = false;
      fillButton.textContent = "Fill this page";
    }
  });

  init();
})();
