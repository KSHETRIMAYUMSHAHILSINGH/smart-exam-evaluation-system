const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  descriptiveQuestions: [],
  mcqQuestions: [],
  questionSource: "custom",
  customPrepared: null,
  ocrFile: null,
  ocrMode: "fast",
  ocrTarget: "answer",
  ocrExtractedText: "",
  ocrResultMeta: null,
};

function showToast(message, ms = 3200) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");

  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    toast.classList.add("hidden");
  }, ms);
}

function setLoading(button, loading, text) {
  if (!button) return;

  if (loading) {
    button.dataset.originalText = button.innerHTML;
    button.disabled = true;
    button.innerHTML = `<span>${text}</span><span class="btn-arrow">···</span>`;
  } else {
    button.disabled = false;
    button.innerHTML = button.dataset.originalText || button.innerHTML;
  }
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`/api${path}`, options);

  let data;
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const detail =
      typeof data === "object"
        ? data.detail || data.message || JSON.stringify(data)
        : data;

    throw new Error(detail || `Request failed (${response.status})`);
  }

  return data;
}

async function checkBackend() {
  const pill = $("#backendStatus");
  const text = $("#backendStatusText");

  try {
    await apiFetch("/questions");
    pill.classList.add("online");
    pill.classList.remove("offline");
    text.textContent = "Backend online";
  } catch (error) {
    pill.classList.add("offline");
    pill.classList.remove("online");
    text.textContent = "Backend offline";
  }
}

function setMode(mode) {
  const descriptive = mode === "descriptive";

  $("#descriptiveTab").classList.toggle("active", descriptive);
  $("#mcqTab").classList.toggle("active", !descriptive);

  $("#descriptiveView").classList.toggle("active", descriptive);
  $("#mcqView").classList.toggle("active", !descriptive);

  window.setTimeout(() => refreshScrollReveals(), 40);
}

function getQuestionById(items, id) {
  return items.find((item) => String(item.question_id) === String(id));
}

function populateDescriptiveQuestions() {
  const select = $("#descriptiveQuestion");
  select.innerHTML = "";

  for (const question of state.descriptiveQuestions) {
    const option = document.createElement("option");
    option.value = question.question_id;
    option.textContent = `${question.question_id}: ${question.question_text}`;
    select.appendChild(option);
  }

  if (state.questionSource === "benchmark") {
    updateDescriptiveQuestion();
  }
}

function updateDescriptiveQuestion() {
  const question = getQuestionById(
    state.descriptiveQuestions,
    $("#descriptiveQuestion").value
  );

  if (!question) return;

  $("#questionId").textContent = question.question_id;
  $("#questionText").textContent = question.question_text;

  if (state.questionSource === "benchmark") {
    $("#modelAnswer").textContent =
      question.model_answer || "No reference answer.";
  }
}

function setQuestionSource(source) {
  state.questionSource = source === "benchmark" ? "benchmark" : "custom";

  $$('[data-question-source]').forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.questionSource === state.questionSource
    );
  });

  $("#customQuestionPanel").classList.toggle(
    "hidden",
    state.questionSource !== "custom"
  );

  $("#benchmarkQuestionPanel").classList.toggle(
    "hidden",
    state.questionSource !== "benchmark"
  );

  if (state.questionSource === "custom") {
    if (state.customPrepared) {
      renderCustomPrepared(state.customPrepared);
    } else {
      $("#modelAnswer").textContent =
        "Generate a grading key to create the reference answer.";
      $("#customRubricCard").classList.add("hidden");
    }
  } else {
    updateDescriptiveQuestion();
    $("#customRubricCard").classList.add("hidden");
  }

  $("#evaluationResults").classList.add("hidden");
  window.setTimeout(() => refreshScrollReveals(), 40);
}

function invalidateCustomKey(showMessage = false) {
  state.customPrepared = null;
  $("#customKeyStatus").textContent = "Key not generated";
  $("#customKeyStatus").classList.remove("ready");
  $("#customRubricCard").classList.add("hidden");
  $("#modelAnswer").textContent =
    "Generate a grading key to create the reference answer.";

  const message = $("#customPrepareMessage");
  message.classList.remove("success", "error");

  if (showMessage) {
    message.textContent =
      "Question changed. Generate a new grading key before evaluation.";
    message.classList.remove("hidden");
  } else {
    message.textContent = "";
    message.classList.add("hidden");
  }
}

function renderCustomPrepared(prepared) {
  $("#modelAnswer").textContent =
    prepared.model_answer || "Reference answer unavailable.";

  const criteria = Array.isArray(prepared.criteria)
    ? prepared.criteria
    : [];

  const list = $("#customRubricPreview");
  list.innerHTML = "";

  criteria.forEach((criterion, index) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <span>C${index + 1}</span>
      <p>${escapeHtml(criterion)}</p>
    `;
    list.appendChild(item);
  });

  $("#customRubricCard").classList.toggle("hidden", criteria.length === 0);
  $("#customKeyStatus").textContent =
    criteria.length === 6 ? "Key ready · 6 criteria" : "Key ready";
  $("#customKeyStatus").classList.add("ready");
}

async function prepareCustomQuestion() {
  const questionText = $("#customQuestion").value.trim();

  if (!questionText) {
    showToast("Type a question or scan a question image first.");
    return;
  }

  const button = $("#prepareCustomBtn");
  const message = $("#customPrepareMessage");

  setLoading(button, true, "Generating grading key");
  message.classList.add("hidden");
  message.classList.remove("success", "error");

  try {
    const prepared = await apiFetch("/custom/prepare", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question_text: questionText,
      }),
    });

    if (!prepared.model_answer || !Array.isArray(prepared.criteria)) {
      throw new Error("Backend returned an incomplete grading key.");
    }

    state.customPrepared = prepared;
    renderCustomPrepared(prepared);

    message.textContent =
      "Grading key generated. Review the reference answer if needed, then enter or scan the student's answer.";
    message.classList.add("success");
    message.classList.remove("hidden");

    showToast("Custom grading key ready.");
  } catch (error) {
    state.customPrepared = null;
    $("#customKeyStatus").textContent = "Key generation failed";
    $("#customKeyStatus").classList.remove("ready");

    message.textContent = `Could not generate grading key: ${error.message}`;
    message.classList.add("error");
    message.classList.remove("hidden");
  } finally {
    setLoading(button, false);
  }
}

function populateMcqQuestions() {
  const select = $("#mcqQuestion");
  select.innerHTML = "";

  for (const question of state.mcqQuestions) {
    const option = document.createElement("option");
    option.value = question.question_id;
    option.textContent = `${question.question_id}: ${question.question_text}`;
    select.appendChild(option);
  }

  updateMcqQuestion();
}

function updateMcqQuestion() {
  const question = getQuestionById(
    state.mcqQuestions,
    $("#mcqQuestion").value
  );

  if (!question) return;

  $("#mcqQuestionId").textContent = question.question_id;
  $("#mcqQuestionText").textContent = question.question_text;

  const optionsWrap = $("#mcqOptions");
  optionsWrap.innerHTML = "";

  const options = question.options || {};

  Object.keys(options).forEach((key) => {
    const wrap = document.createElement("div");
    wrap.className = "mcq-option";

    const id = `mcq_${question.question_id}_${key}`;

    wrap.innerHTML = `
      <input id="${id}" type="radio" name="mcqAnswer" value="${escapeHtml(key)}" />
      <label for="${id}">
        <span class="option-key">${escapeHtml(key)}</span>
        <span>${escapeHtml(options[key])}</span>
      </label>
    `;

    optionsWrap.appendChild(wrap);
  });

  $("#mcqResult").classList.add("hidden");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadQuestions() {
  try {
    const [descriptive, mcq] = await Promise.all([
      apiFetch("/questions"),
      apiFetch("/mcq/questions"),
    ]);

    state.descriptiveQuestions = Array.isArray(descriptive) ? descriptive : [];
    state.mcqQuestions = Array.isArray(mcq) ? mcq : [];

    if (state.descriptiveQuestions.length) {
      populateDescriptiveQuestions();
    } else {
      $("#descriptiveQuestion").innerHTML =
        "<option>No benchmark questions found</option>";
    }

    if (state.mcqQuestions.length) {
      populateMcqQuestions();
    } else {
      $("#mcqQuestion").innerHTML = "<option>No MCQs found</option>";
    }

    setQuestionSource("custom");
  } catch (error) {
    showToast(`Could not load questions: ${error.message}`, 5000);
    setQuestionSource("custom");
  }
}

function bindFileUpload() {
  const input = $("#ocrFile");
  const dropzone = $("#dropzone");

  input.addEventListener("change", () => {
    const [file] = input.files;
    setOcrFile(file || null);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) {
      setOcrFile(file);
    }
  });

  $("#removeFileBtn").addEventListener("click", () => {
    input.value = "";
    setOcrFile(null);
  });
}

function setOcrFile(file) {
  state.ocrFile = file;
  const preview = $("#filePreview");
  const extractBtn = $("#extractBtn");

  if (!file) {
    preview.classList.add("hidden");
    extractBtn.disabled = true;
    $("#ocrMessage").classList.add("hidden");
    clearOcrReview();
    return;
  }

  const validTypes = ["image/png", "image/jpeg"];
  if (!validTypes.includes(file.type)) {
    showToast("Please choose a PNG, JPG or JPEG image.");
    state.ocrFile = null;
    return;
  }

  clearOcrReview();

  $("#fileName").textContent = file.name;
  $("#fileMeta").textContent =
    `${(file.size / 1024).toFixed(1)} KB · ${capitalize(state.ocrTarget)}`;

  const reader = new FileReader();
  reader.onload = () => {
    $("#previewImage").src = reader.result;
  };
  reader.readAsDataURL(file);

  preview.classList.remove("hidden");
  extractBtn.disabled = false;
}

function updateOcrTargetUi() {
  $$('[data-ocr-target]').forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.ocrTarget === state.ocrTarget
    );
  });

  const isQuestion = state.ocrTarget === "question";

  $("#dropzoneTitle").textContent =
    isQuestion ? "Drop question image" : "Drop answer image";

  $("#ocrTargetHelp").textContent = isQuestion
    ? "Extract first, review the recognized text, then choose Use as question."
    : "Extract first, review the recognized text, then choose Use as student answer.";

  updateOcrReviewDestination();

  if (state.ocrFile) {
    $("#fileMeta").textContent =
      `${(state.ocrFile.size / 1024).toFixed(1)} KB · ${capitalize(state.ocrTarget)}`;
  }
}

function bindOcrTarget() {
  $$('[data-ocr-target]').forEach((button) => {
    button.addEventListener("click", () => {
      state.ocrTarget = button.dataset.ocrTarget === "question"
        ? "question"
        : "answer";
      updateOcrTargetUi();
      $("#ocrMessage").classList.add("hidden");
    });
  });

  updateOcrTargetUi();
}

function bindOcrMode() {
  $$('[data-ocr-mode]').forEach((button) => {
    button.addEventListener("click", () => {
      $$('[data-ocr-mode]').forEach((item) => item.classList.remove("active"));
      button.classList.add("active");

      state.ocrMode = button.dataset.ocrMode;

      $("#ocrModeHelp").textContent =
        state.ocrMode === "fast"
          ? "Best for clear, normal handwriting. Usually around 1–2 seconds."
          : "Uses the local Qwen3.5 vision model for difficult handwriting. Slower, but context-aware.";
    });
  });
}

function updateOcrReviewDestination() {
  const isQuestion = state.ocrTarget === "question";
  const target = $("#ocrReviewTarget");
  const useButton = $("#ocrUseBtn");

  if (target) {
    target.textContent = isQuestion ? "Question" : "Student answer";
  }

  if (useButton) {
    useButton.textContent = isQuestion ? 'Use as question' : 'Use as student answer';
  }
}

function clearOcrReview() {
  state.ocrExtractedText = "";
  state.ocrResultMeta = null;

  const panel = $("#ocrReviewPanel");
  const editor = $("#ocrExtractedText");
  const status = $("#ocrReviewStatus");

  if (editor) editor.value = "";
  if (panel) panel.classList.add("hidden");
  if (status) status.textContent = "Not applied yet · review before use";

  if ($("#ocrReviewMode")) $("#ocrReviewMode").textContent = "—";
  if ($("#ocrReviewEngine")) $("#ocrReviewEngine").textContent = "—";
  if ($("#ocrReviewTime")) $("#ocrReviewTime").textContent = "—";
}

function renderOcrReview(extractedText, data) {
  state.ocrExtractedText = extractedText;
  state.ocrResultMeta = data || {};

  $("#ocrExtractedText").value = extractedText;
  $("#ocrReviewMode").textContent = capitalize(data.mode || state.ocrMode);
  $("#ocrReviewEngine").textContent = data.engine || "OCR";
  $("#ocrReviewTime").textContent =
    typeof data.elapsed_sec === "number"
      ? `${data.elapsed_sec.toFixed(2)}s`
      : "—";

  $("#ocrReviewStatus").textContent = "Not applied yet · review before use";
  updateOcrReviewDestination();
  $("#ocrReviewPanel").classList.remove("hidden");

  window.setTimeout(() => {
    $("#ocrReviewPanel").scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "nearest",
    });
  }, 80);
}

function applyReviewedOcr() {
  const text = $("#ocrExtractedText").value.trim();

  if (!text) {
    showToast("There is no extracted text to use.");
    return;
  }

  state.ocrExtractedText = text;

  if (state.ocrTarget === "question") {
    setQuestionSource("custom");
    $("#customQuestion").value = text;
    invalidateCustomKey(false);
    $("#ocrReviewStatus").textContent =
      "Applied to question · generate a new grading key";
    showToast("Reviewed OCR text applied as the question.");
  } else {
    $("#studentAnswer").value = text;
    $("#ocrReviewStatus").textContent =
      "Applied to student answer · ready for evaluation";
    showToast("Reviewed OCR text applied as the student answer.");
  }
}

async function copyReviewedOcr() {
  const text = $("#ocrExtractedText").value.trim();

  if (!text) {
    showToast("There is no extracted text to copy.");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    showToast("Extracted text copied.");
  } catch (error) {
    const editor = $("#ocrExtractedText");
    editor.focus();
    editor.select();
    const copied = document.execCommand("copy");
    showToast(copied ? "Extracted text copied." : "Copy failed. Select the text manually.");
  }
}

async function extractOcr() {
  if (!state.ocrFile) {
    showToast(
      state.ocrTarget === "question"
        ? "Choose a question image first."
        : "Choose an answer image first."
    );
    return;
  }

  const button = $("#extractBtn");
  const message = $("#ocrMessage");

  button.disabled = true;
  button.textContent =
    state.ocrMode === "fast" ? "Running Fast OCR..." : "Running Accurate OCR...";

  message.classList.add("hidden");
  message.classList.remove("success", "error");

  const formData = new FormData();
  formData.append("file", state.ocrFile, state.ocrFile.name);
  formData.append("mode", state.ocrMode);

  try {
    const data = await apiFetch("/ocr", {
      method: "POST",
      body: formData,
    });

    const extractedText = (data.extracted_text || "").trim();

    if (!extractedText) {
      message.textContent = "OCR could not detect readable text from this image.";
      message.classList.remove("hidden");
      return;
    }

    renderOcrReview(extractedText, data);

    const elapsed =
      typeof data.elapsed_sec === "number"
        ? ` · ${data.elapsed_sec.toFixed(2)}s`
        : "";

    const engine = data.engine || "OCR";
    const returnedMode = data.mode || state.ocrMode;

    message.textContent =
      `OCR complete. Review and correct the extracted text below before using it. ${capitalize(returnedMode)} · ${engine}${elapsed}.`;

    message.classList.add("success");
    message.classList.remove("hidden");
    showToast("OCR complete — review the extracted text.");
  } catch (error) {
    message.textContent = `OCR failed: ${error.message}`;
    message.classList.add("error");
    message.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = "Extract text";
  }
}

function capitalize(value) {
  const text = String(value || "");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function renderKeywordChips(target, values) {
  target.innerHTML = "";

  const items = Array.isArray(values) && values.length ? values : ["None"];

  items.forEach((value) => {
    const chip = document.createElement("span");
    chip.textContent = value;
    target.appendChild(chip);
  });
}

function renderRubric(items) {
  const target = $("#rubricList");
  target.innerHTML = "";

  const rubricItems = Array.isArray(items) ? items : [];

  if (!rubricItems.length) {
    target.innerHTML =
      '<div class="message">No rubric evidence was returned.</div>';
    return;
  }

  rubricItems.forEach((item) => {
    const wrap = document.createElement("div");
    wrap.className = "rubric-item";

    let statusText = "Not supported";
    let statusClass = "";

    if (item.guard_rejected) {
      statusText = "Evidence guard";
      statusClass = "rejected";
    } else if (item.supported) {
      statusText = "Supported";
      statusClass = "supported";
    }

    const evidence = item.evidence
      ? `<small>Evidence: ${escapeHtml(item.evidence)}</small>`
      : "";

    wrap.innerHTML = `
      <div class="rubric-top">
        <strong>C${escapeHtml(item.criterion_id)}</strong>
        <span class="rubric-status ${statusClass}">${escapeHtml(statusText)}</span>
      </div>
      <p>${escapeHtml(item.criterion || "")}</p>
      ${evidence}
    `;

    target.appendChild(wrap);
  });
}

function addAlert(message, type = "") {
  const div = document.createElement("div");
  div.className = `alert ${type}`.trim();
  div.textContent = message;
  $("#alertsArea").appendChild(div);
}

function renderEvaluation(result) {
  $("#evaluationResults").classList.remove("hidden");

  animateScoreValue($("#finalScore"), Number(result.final_score ?? 0));
  $("#graderMeta").textContent =
    `${result.grader_name || "Qwen"} · ${result.grader_version || "V3.3.7 frozen"}`;

  $("#metricCore").textContent = `${result.core ?? 0}/5`;
  $("#metricRubric").textContent = `${result.supported_count ?? 0}/6`;
  $("#metricPenalty").textContent = `-${result.contradiction_penalty ?? 0}`;
  $("#metricTime").textContent =
    `${Number(result.elapsed_sec ?? 0).toFixed(2)}s`;

  const alerts = $("#alertsArea");
  alerts.innerHTML = "";

  if (result.evidence_guard_applied) {
    const rejected = (result.rejected_criterion_ids || [])
      .map((item) => `C${item}`)
      .join(", ");

    addAlert(
      `Evidence validation removed unsupported rubric claims: ${rejected || "none"}. ` +
      `Raw frozen-Qwen score: ${result.raw_qwen_score ?? 0}/10.`,
      "warning"
    );
  }

  if (result.non_answer) {
    const reason = result.non_answer_reason
      ? ` (${result.non_answer_reason})`
      : "";

    addAlert(`The response was classified as a non-answer${reason}.`, "warning");
  }

  if (result.wrong_topic) {
    addAlert("The response was classified as wrong-topic.", "warning");
  }

  if (
    Number(result.contradiction_penalty || 0) > 0 &&
    result.contradiction_quote
  ) {
    addAlert(
      `Contradictory claim detected: ${result.contradiction_quote}`,
      "danger"
    );
  }

  renderRubric(result.rubric_results);

  const trainedScore = result.trained_score;
  const comparisonFallback = $("#comparisonFallback");
  const isCustomQuestion =
    String(result.question_id || "").toUpperCase() === "CUSTOM";

  if (isCustomQuestion) {
    // Preserve the original V6 three-column model-comparison layout.
    // Only the content changes for unseen custom questions.
    $("#comparisonEyebrow").textContent = "GRADING SUMMARY";
    $("#comparisonTitle").textContent = "Qwen grading result";
    $("#comparisonTag").textContent = "QWEN";

    $("#comparisonLabel1").textContent = "Raw Qwen score";
    $("#comparisonLabel2").textContent = "Final score";
    $("#comparisonLabel3").textContent = "Rubric support";

    const rawScore = Number(result.raw_qwen_score);
    const finalScore = Number(result.final_score);
    const rejected = Array.isArray(result.rejected_criterion_ids)
      ? result.rejected_criterion_ids.map((item) => `C${item}`)
      : [];

    $("#trainedScore").textContent = Number.isFinite(rawScore)
      ? `${rawScore.toFixed(2)}/10`
      : "—";
    $("#trainedModelName").textContent = "Before evidence validation";

    $("#semanticScore").textContent = Number.isFinite(finalScore)
      ? `${finalScore.toFixed(2)}/10`
      : "—";
    $("#comparisonMeta2").textContent = "Evidence-adjusted";

    $("#baselineScore").textContent = `${result.supported_count ?? 0}/6`;
    $("#comparisonMeta3").textContent = "Validated criteria";

    $("#matchedKeywordsLabel").textContent = "Core correctness";
    renderKeywordChips($("#matchedKeywords"), [`${result.core ?? 0}/5`]);

    $("#missingKeywordsLabel").textContent = "Rejected evidence claims";
    renderKeywordChips($("#missingKeywords"), rejected);

    comparisonFallback.textContent = rejected.length
      ? `Evidence validation rejected ${rejected.length} unsupported rubric claim${rejected.length === 1 ? "" : "s"}. The final score above is the score used.`
      : "All claimed rubric support passed evidence validation. The trained benchmark model is intentionally not applied to unseen custom questions.";
    comparisonFallback.classList.remove("hidden");
  } else {
    // Original V6 benchmark presentation.
    $("#comparisonEyebrow").textContent = "PROJECT COMPARISON";
    $("#comparisonTitle").textContent = "Model scores";
    $("#comparisonTag").textContent = "ML";
    $("#comparisonLabel1").textContent = "Our trained model";
    $("#comparisonLabel2").textContent = "Semantic baseline";
    $("#comparisonLabel3").textContent = "Keyword baseline";
    $("#comparisonMeta2").textContent = "";
    $("#comparisonMeta3").textContent = "";
    $("#matchedKeywordsLabel").textContent = "Matched legacy keywords";
    $("#missingKeywordsLabel").textContent = "Missing legacy keywords";

    if (trainedScore !== null && trainedScore !== undefined) {
      comparisonFallback.classList.add("hidden");

      $("#trainedScore").textContent = `${trainedScore}/10`;
      $("#trainedModelName").textContent =
        result.trained_model_name || "Trained model";

      $("#semanticScore").textContent =
        result.semantic_score !== null && result.semantic_score !== undefined
          ? `${result.semantic_score}/10`
          : "Unavailable";

      $("#baselineScore").textContent =
        result.baseline_score !== null && result.baseline_score !== undefined
          ? `${result.baseline_score}/10`
          : "Unavailable";

      renderKeywordChips($("#matchedKeywords"), result.matched_keywords);
      renderKeywordChips($("#missingKeywords"), result.missing_keywords);
    } else {
      $("#trainedScore").textContent = "—";
      $("#trainedModelName").textContent = "Unavailable";
      $("#semanticScore").textContent = "—";
      $("#baselineScore").textContent = "—";
      renderKeywordChips($("#matchedKeywords"), []);
      renderKeywordChips($("#missingKeywords"), []);
      comparisonFallback.textContent =
        "The legacy trained-model comparison was unavailable, but the frozen Qwen final grade completed successfully.";
      comparisonFallback.classList.remove("hidden");
    }
  }

  $("#evaluationResults").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

async function evaluateAnswer() {
  const answer = $("#studentAnswer").value.trim();

  if (!answer) {
    showToast("Type an answer or scan an answer image first.");
    return;
  }

  const button = $("#evaluateBtn");
  setLoading(button, true, "Evaluating answer");

  try {
    let result;

    if (state.questionSource === "custom") {
      const questionText = $("#customQuestion").value.trim();

      if (!questionText) {
        throw new Error("Enter or scan a question first.");
      }

      if (!state.customPrepared) {
        throw new Error(
          "Generate the grading key for this question before evaluating the answer."
        );
      }

      const payload = {
        question_text: questionText,
        model_answer: state.customPrepared.model_answer,
        criteria: state.customPrepared.criteria,
        student_answer: answer,
      };

      if ($("#recordHumanScore").checked) {
        payload.human_score = Number($("#humanScore").value);
      }

      result = await apiFetch("/custom/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    } else {
      const payload = {
        question_id: $("#descriptiveQuestion").value,
        student_answer: answer,
      };

      if ($("#recordHumanScore").checked) {
        payload.human_score = Number($("#humanScore").value);
      }

      result = await apiFetch("/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    }

    renderEvaluation(result);
  } catch (error) {
    showToast(`Evaluation failed: ${error.message}`, 5000);
  } finally {
    setLoading(button, false);
  }
}

async function submitMcq() {
  const selected = $('input[name="mcqAnswer"]:checked');

  if (!selected) {
    showToast("Select an option first.");
    return;
  }

  const payload = {
    question_id: $("#mcqQuestion").value,
    student_answer: selected.value,
  };

  const button = $("#submitMcqBtn");
  setLoading(button, true, "Checking answer");

  try {
    const result = await apiFetch("/mcq/evaluate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const question = getQuestionById(
      state.mcqQuestions,
      $("#mcqQuestion").value
    );

    const target = $("#mcqResult");
    target.classList.remove("hidden", "correct", "incorrect");

    if (result.is_correct) {
      target.classList.add("correct");
      target.innerHTML = `
        <strong>Correct</strong>
        <p>Score: ${escapeHtml(result.score)}/${escapeHtml(result.max_score)}</p>
      `;
    } else {
      target.classList.add("incorrect");

      const correctAnswer = result.correct_answer;
      const correctText =
        correctAnswer && question?.options
          ? question.options[correctAnswer] || ""
          : "";

      target.innerHTML = `
        <strong>Incorrect</strong>
        <p>Score: ${escapeHtml(result.score)}/${escapeHtml(result.max_score)}</p>
        ${
          correctAnswer
            ? `<p>Correct answer: ${escapeHtml(correctAnswer)}. ${escapeHtml(correctText)}</p>`
            : ""
        }
      `;
    }
  } catch (error) {
    showToast(`MCQ evaluation failed: ${error.message}`, 5000);
  } finally {
    setLoading(button, false);
  }
}

function bindUi() {
  $("#descriptiveTab").addEventListener("click", () => setMode("descriptive"));
  $("#mcqTab").addEventListener("click", () => setMode("mcq"));

  $$('[data-question-source]').forEach((button) => {
    button.addEventListener("click", () => {
      setQuestionSource(button.dataset.questionSource);
    });
  });

  $("#customQuestion").addEventListener("input", () => {
    if (state.customPrepared) {
      invalidateCustomKey(true);
    }
  });

  $("#prepareCustomBtn").addEventListener("click", prepareCustomQuestion);

  $("#descriptiveQuestion").addEventListener(
    "change",
    updateDescriptiveQuestion
  );

  $("#mcqQuestion").addEventListener("change", updateMcqQuestion);

  $("#recordHumanScore").addEventListener("change", (event) => {
    $("#humanScoreWrap").classList.toggle("hidden", !event.target.checked);
  });

  $("#evaluateBtn").addEventListener("click", evaluateAnswer);
  $("#submitMcqBtn").addEventListener("click", submitMcq);
  $("#extractBtn").addEventListener("click", extractOcr);
  $("#ocrUseBtn").addEventListener("click", applyReviewedOcr);
  $("#ocrCopyBtn").addEventListener("click", copyReviewedOcr);
  $("#ocrRerunBtn").addEventListener("click", extractOcr);
  $("#ocrClearBtn").addEventListener("click", clearOcrReview);
  $("#ocrExtractedText").addEventListener("input", (event) => {
    state.ocrExtractedText = event.target.value;
    $("#ocrReviewStatus").textContent = "Edited · not applied yet";
  });

  bindFileUpload();
  bindOcrTarget();
  bindOcrMode();
  setQuestionSource("custom");
}


// ============================================================
// V6 VISUAL SYSTEM
// ============================================================

const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
).matches;

function scrollToId(id) {
  const element = document.querySelector(id);
  if (!element) return;
  element.scrollIntoView({
    behavior: prefersReducedMotion ? "auto" : "smooth",
    block: "start",
  });
}

function setupNavigationButtons() {
  const mapping = [
    ["#openConsoleBtn", "#console"],
    ["#heroConsoleBtn", "#console"],
    ["#closingConsoleBtn", "#console"],
    ["#heroSystemBtn", "#capabilities"],
  ];

  mapping.forEach(([buttonId, target]) => {
    const button = document.querySelector(buttonId);
    if (button) {
      button.addEventListener("click", () => scrollToId(target));
    }
  });

  const btt = document.querySelector("#backToTop");
  if (btt) {
    btt.addEventListener("click", () => scrollToId("#top"));
  }
}

function setupPreloader() {
  const preloader = document.querySelector("#preloader");
  const number = document.querySelector("#preNumber");
  const fill = document.querySelector("#preFill");

  if (!preloader || !number || !fill) {
    document.body.classList.add("loaded");
    return;
  }

  if (prefersReducedMotion) {
    preloader.classList.add("done");
    document.body.classList.add("loaded");
    return;
  }

  const started = performance.now();
  const duration = 1550;

  function frame(now) {
    const p = Math.min((now - started) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const value = Math.round(eased * 100);

    number.textContent = String(value);
    fill.style.width = `${value}%`;

    if (p < 1) {
      requestAnimationFrame(frame);
      return;
    }

    window.setTimeout(() => {
      preloader.classList.add("done");
      document.body.classList.add("loaded");
    }, 180);
  }

  requestAnimationFrame(frame);
}

function setupScrollUi() {
  const progress = document.querySelector("#scrollProgress");
  const nav = document.querySelector("#mainNav");
  const btt = document.querySelector("#backToTop");

  function update() {
    const y = window.scrollY;
    const max =
      document.documentElement.scrollHeight - window.innerHeight;

    if (progress) {
      progress.style.width =
        `${max > 0 ? Math.min(100, (y / max) * 100) : 0}%`;
    }

    if (nav) {
      nav.classList.toggle("scrolled", y > 55);
    }

    if (btt) {
      btt.classList.toggle("show", y > 650);
    }
  }

  window.addEventListener("scroll", update, { passive: true });
  update();
}

function setupClock() {
  const clock = document.querySelector("#localClock");
  if (!clock) return;

  function update() {
    clock.textContent = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  update();
  window.setInterval(update, 1000);
}

function setupRotator() {
  const rotator = document.querySelector("#heroRotator");
  if (!rotator || prefersReducedMotion) return;

  let index = 0;
  const count = rotator.children.length;

  window.setInterval(() => {
    index = (index + 1) % count;
    rotator.style.transform = `translateY(-${index * 100}%)`;
  }, 2700);
}

function setupHeroParallax() {
  const content = document.querySelector("#heroContent");
  const hero = document.querySelector("#hero");

  if (!content || !hero || prefersReducedMotion) return;

  let tx = 0;
  let ty = 0;
  let cx = 0;
  let cy = 0;

  hero.addEventListener("pointermove", (event) => {
    const rect = hero.getBoundingClientRect();
    tx = ((event.clientX - rect.left) / rect.width - 0.5) * 14;
    ty = ((event.clientY - rect.top) / rect.height - 0.5) * 12;
  });

  hero.addEventListener("pointerleave", () => {
    tx = 0;
    ty = 0;
  });

  function animate() {
    cx += (tx - cx) * 0.055;
    cy += (ty - cy) * 0.055;

    content.style.transform =
      `translate3d(${-cx}px, ${-cy}px, 0) ` +
      `rotateX(${cy * 0.12}deg) rotateY(${-cx * 0.12}deg)`;

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

function setupCursor() {
  const dot = document.querySelector("#cursorDot");
  const ring = document.querySelector("#cursorRing");

  const isTouch =
    "ontouchstart" in window ||
    navigator.maxTouchPoints > 0 ||
    window.innerWidth <= 760;

  if (!dot || !ring || isTouch || prefersReducedMotion) {
    document.body.style.cursor = "auto";
    return;
  }

  const trailCount = 5;
  const trails = [];
  let queue = Array.from({ length: trailCount }, () => ({
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
  }));

  for (let i = 0; i < trailCount; i += 1) {
    const el = document.createElement("div");
    el.className = "cursor-trail";
    const size = Math.max(1, 2.8 - i * 0.35);
    el.style.width = `${size}px`;
    el.style.height = `${size}px`;
    el.style.opacity = String((1 - i / trailCount) * 0.23);
    document.body.appendChild(el);
    trails.push(el);
  }

  let mx = window.innerWidth / 2;
  let my = window.innerHeight / 2;
  let rx = mx;
  let ry = my;

  window.addEventListener("pointermove", (event) => {
    mx = event.clientX;
    my = event.clientY;

    dot.style.left = `${mx}px`;
    dot.style.top = `${my}px`;

    queue = [{ x: mx, y: my }, ...queue.slice(0, trailCount - 1)];
  });

  window.addEventListener("pointerdown", () => {
    document.body.classList.add("cursor-press");
  });

  window.addEventListener("pointerup", () => {
    document.body.classList.remove("cursor-press");
  });

  document.addEventListener("pointerover", (event) => {
    if (
      event.target.closest(
        "button,a,input,textarea,select,summary,.tilt-card,.architecture-card,.mcq-option"
      )
    ) {
      document.body.classList.add("cursor-hover");
    }
  });

  document.addEventListener("pointerout", (event) => {
    if (
      event.target.closest(
        "button,a,input,textarea,select,summary,.tilt-card,.architecture-card,.mcq-option"
      )
    ) {
      document.body.classList.remove("cursor-hover");
    }
  });

  function frame() {
    rx += (mx - rx) * 0.13;
    ry += (my - ry) * 0.13;

    ring.style.left = `${rx}px`;
    ring.style.top = `${ry}px`;

    trails.forEach((trail, index) => {
      const point = queue[index] || queue[queue.length - 1];
      trail.style.left = `${point.x}px`;
      trail.style.top = `${point.y}px`;
    });

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

function setupHeroCanvas() {
  const canvas = document.querySelector("#heroCanvas");
  if (!canvas || prefersReducedMotion) return;

  const ctx = canvas.getContext("2d");
  let w = 0;
  let h = 0;
  let dpr = 1;
  let points = [];
  let mouse = { x: -9999, y: -9999 };
  const hero = document.querySelector("#hero");

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = Math.max(1, rect.width);
    h = Math.max(1, rect.height);

    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const amount = window.innerWidth < 760 ? 42 : 82;

    points = Array.from({ length: amount }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.19,
      vy: (Math.random() - 0.5) * 0.19,
      baseVx: (Math.random() - 0.5) * 0.19,
      baseVy: (Math.random() - 0.5) * 0.19,
      r: 0.7 + Math.random() * 0.7,
    }));
  }

  hero?.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = event.clientX - rect.left;
    mouse.y = event.clientY - rect.top;
  });

  hero?.addEventListener("pointerleave", () => {
    mouse = { x: -9999, y: -9999 };
  });

  hero?.addEventListener("pointerdown", (event) => {
    const rect = canvas.getBoundingClientRect();
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;

    points.forEach((p) => {
      const dx = p.x - cx;
      const dy = p.y - cy;
      const distance = Math.hypot(dx, dy);

      if (distance < 180 && distance > 0) {
        const power = ((180 - distance) / 180) * 2.5;
        p.vx += (dx / distance) * power;
        p.vy += (dy / distance) * power;
      }
    });
  });

  function draw() {
    ctx.clearRect(0, 0, w, h);

    points.forEach((p) => {
      const dx = p.x - mouse.x;
      const dy = p.y - mouse.y;
      const distance = Math.hypot(dx, dy);

      if (distance < 145 && distance > 0) {
        const force = ((145 - distance) / 145) * 0.42;
        p.vx += (dx / distance) * force;
        p.vy += (dy / distance) * force;
      }

      p.vx += (p.baseVx - p.vx) * 0.025;
      p.vy += (p.baseVy - p.vy) * 0.025;

      const speed = Math.hypot(p.vx, p.vy);
      if (speed > 2.4) {
        p.vx = (p.vx / speed) * 2.4;
        p.vy = (p.vy / speed) * 2.4;
      }

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      const proximity =
        distance < 240 ? Math.max(0, 1 - distance / 240) : 0;

      ctx.beginPath();
      ctx.arc(
        p.x,
        p.y,
        p.r + proximity * 1.4,
        0,
        Math.PI * 2
      );
      ctx.fillStyle =
        `rgba(255,255,255,${0.17 + proximity * 0.37})`;
      ctx.fill();
    });

    for (let i = 0; i < points.length; i += 1) {
      for (let j = i + 1; j < points.length; j += 1) {
        const a = points[i];
        const b = points[j];
        const distance = Math.hypot(a.x - b.x, a.y - b.y);

        if (distance < 118) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle =
            `rgba(255,255,255,${0.105 * (1 - distance / 118)})`;
          ctx.lineWidth = 0.55;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(draw);
}

function setupScoreCanvas() {
  const canvas = document.querySelector("#scoreCanvas");
  if (!canvas || prefersReducedMotion) return;

  const ctx = canvas.getContext("2d");

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  let angle = 0;

  function draw() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const x = w * 0.78;
    const y = h * 0.37;

    ctx.strokeStyle = "rgba(255,255,255,.09)";
    ctx.lineWidth = 1;

    [48, 77, 108].forEach((radius, idx) => {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();

      const px = x + Math.cos(angle * (idx + 1) * 0.7) * radius;
      const py = y + Math.sin(angle * (idx + 1) * 0.7) * radius;

      ctx.beginPath();
      ctx.arc(px, py, idx === 0 ? 2.4 : 1.6, 0, Math.PI * 2);
      ctx.fillStyle =
        idx === 0
          ? "rgba(184,151,62,.75)"
          : "rgba(255,255,255,.32)";
      ctx.fill();
    });

    angle += 0.006;
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(draw);
}

function setupTiltAndGlow() {
  if (prefersReducedMotion || window.innerWidth <= 760) return;

  document.querySelectorAll(".tilt-card").forEach((card) => {
    const glow = card.querySelector(".card-glow");

    card.addEventListener("pointermove", (event) => {
      const rect = card.getBoundingClientRect();

      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const rx = ((y / rect.height) - 0.5) * -2.1;
      const ry = ((x / rect.width) - 0.5) * 2.1;

      card.style.transform =
        `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg)`;

      if (glow) {
        glow.style.left = `${x}px`;
        glow.style.top = `${y}px`;
      }
    });

    card.addEventListener("pointerleave", () => {
      card.style.transform =
        "perspective(1000px) rotateX(0deg) rotateY(0deg)";
    });
  });
}

function setupClosingGlow() {
  const section = document.querySelector(".closing");
  const glow = document.querySelector("#closingGlow");

  if (!section || !glow || prefersReducedMotion) return;

  section.addEventListener("pointermove", (event) => {
    const rect = section.getBoundingClientRect();
    glow.style.left = `${event.clientX - rect.left}px`;
    glow.style.top = `${event.clientY - rect.top}px`;
  });
}

function setupRippleButtons() {
  document.querySelectorAll(".ripple-btn").forEach((button) => {
    button.addEventListener("click", (event) => {
      const rect = button.getBoundingClientRect();
      const ripple = document.createElement("span");
      ripple.className = "ripple";

      const size = Math.max(rect.width, rect.height);
      ripple.style.width = `${size}px`;
      ripple.style.height = `${size}px`;
      ripple.style.left =
        `${event.clientX - rect.left - size / 2}px`;
      ripple.style.top =
        `${event.clientY - rect.top - size / 2}px`;

      button.appendChild(ripple);

      window.setTimeout(() => ripple.remove(), 600);
    });
  });
}

let revealObserver = null;

function refreshScrollReveals() {
  if (prefersReducedMotion) {
    document.querySelectorAll(".sr").forEach((el) => {
      el.classList.add("visible");
    });
    return;
  }

  if (!revealObserver) {
    revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;

          entry.target.classList.add("visible");
          revealObserver.unobserve(entry.target);
        });
      },
      {
        threshold: 0.08,
        rootMargin: "0px 0px -7% 0px",
      }
    );
  }

  document.querySelectorAll(".sr:not(.visible)").forEach((el) => {
    revealObserver.observe(el);
  });
}

function setupArchitectureHover() {
  document.querySelectorAll(".architecture-card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      if (prefersReducedMotion) return;

      const rect = card.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      card.style.background =
        `radial-gradient(circle at ${x * 100}% 25%, rgba(184,151,62,.045), transparent 38%), #0a0a0a`;
    });

    card.addEventListener("pointerleave", () => {
      card.style.background = "";
    });
  });
}

function animateScoreValue(element, target, duration = 720) {
  if (!element || !Number.isFinite(target)) {
    if (element) element.textContent = String(target || 0);
    return;
  }

  if (prefersReducedMotion) {
    element.textContent = String(target);
    return;
  }

  const start = performance.now();
  const decimalPlaces =
    Number.isInteger(target) ? 0 : Math.min(2, String(target).split(".")[1]?.length || 1);

  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;

    element.textContent = value.toFixed(decimalPlaces);

    if (progress < 1) {
      requestAnimationFrame(frame);
    } else {
      element.textContent = String(target);
    }
  }

  requestAnimationFrame(frame);
}

function setupVisualSystem() {
  setupPreloader();
  setupNavigationButtons();
  setupScrollUi();
  setupClock();
  setupRotator();
  setupHeroParallax();
  setupCursor();
  setupHeroCanvas();
  setupScoreCanvas();
  setupTiltAndGlow();
  setupClosingGlow();
  setupRippleButtons();
  setupArchitectureHover();
  refreshScrollReveals();
}


async function init() {
  bindUi();
  setupVisualSystem();
  await checkBackend();
  await loadQuestions();
}

init();
