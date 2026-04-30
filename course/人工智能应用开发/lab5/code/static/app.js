const config = {
  entity_name: "实验提交",
  fields: [
    { label: "提交编号", key: "id", type: "text", editable: false, multiline: false },
    { label: "学生编号", key: "student_id", type: "text", editable: true, multiline: false },
    { label: "实验编号", key: "lab_id", type: "text", editable: true, multiline: false },
    { label: "提交时间", key: "submitted_at", type: "text", editable: true, multiline: false },
    { label: "标题", key: "title", type: "text", editable: true, multiline: false },
    { label: "说明", key: "description", type: "text", editable: true, multiline: true },
    { label: "附件链接", key: "attachment_urls", type: "text", editable: true, multiline: false },
    { label: "状态", key: "status", type: "text", editable: false, multiline: false },
  ],
  create_fields: [
    { label: "学生编号", key: "student_id", type: "text", editable: true, multiline: false },
    { label: "实验编号", key: "lab_id", type: "text", editable: true, multiline: false },
    { label: "提交时间", key: "submitted_at", type: "text", editable: true, multiline: false },
    { label: "标题", key: "title", type: "text", editable: true, multiline: false },
    { label: "说明", key: "description", type: "text", editable: true, multiline: true },
    { label: "附件链接", key: "attachment_urls", type: "text", editable: true, multiline: false },
  ],
  statuses: ["draft", "submitted", "under_review", "approved", "rejected", "needs_resubmission"],
  list_columns: ["提交编号", "学生姓名", "实验标题", "状态", "分数", "提交时间"],
  filters: ["状态", "实验编号", "学生编号", "评审教师编号"],
};

const statusLabels = {
  all: "全部",
  draft: "草稿",
  submitted: "已提交",
  under_review: "评审中",
  approved: "已通过",
  rejected: "已驳回",
  needs_resubmission: "需重新提交",
  graded: "已评分",
};

const metricsEl = document.getElementById("metrics");
const createForm = document.getElementById("createForm");
const createMessage = document.getElementById("createMessage");
const recordsEl = document.getElementById("records");
const filterStrip = document.getElementById("filterStrip");
const emptyState = document.getElementById("emptyState");
const detailPanel = document.getElementById("detailPanel");
const detailTitle = document.getElementById("detailTitle");
const detailMeta = document.getElementById("detailMeta");
const detailStatus = document.getElementById("detailStatus");
const recordFields = document.getElementById("recordFields");
const reviewForm = document.getElementById("reviewForm");
const reviewerInput = document.getElementById("reviewerInput");
const statusSelect = document.getElementById("statusSelect");
const reviewCommentInput = document.getElementById("reviewCommentInput");
const reviewMessage = document.getElementById("reviewMessage");

const state = {
  records: [],
  metrics: [],
  activeId: null,
  activeFilter: "all",
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败。");
  return data;
}

function displayStatus(status) {
  return statusLabels[status] || status || "-";
}

function statusClass(status) {
  const value = String(status || "").toLowerCase();
  if (value.includes("approve") || value.includes("通过")) return "status-approved";
  if (value.includes("reject") || value.includes("驳回")) return "status-rejected";
  if (value.includes("review") || value.includes("评审")) return "status-review";
  return "";
}

function filteredRecords() {
  if (state.activeFilter === "all") {
    return state.records;
  }
  return state.records.filter((item) => String(item.status) === state.activeFilter);
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return Array.isArray(value) ? value.join("、") : value;
}

function primaryTitle(record) {
  return displayValue(record[config.fields[0]?.key] || record.id);
}

function secondaryText(record) {
  const parts = config.fields.slice(1, 3).map((field) => record[field.key]).filter(Boolean);
  return parts.length ? parts.join(" | ") : record.id;
}

function numericSummary(record) {
  const numericField = config.fields.find((field) => field.type === "number");
  if (!numericField) {
    return "";
  }
  return `${numericField.label}：${displayValue(record[numericField.key])}`;
}

function buildForm() {
  createForm.innerHTML = "";
  config.create_fields.forEach((field) => {
    const label = document.createElement("label");
    if (field.multiline) {
      label.className = "full";
    }
    const span = document.createElement("span");
    span.textContent = field.label;
    const useTextarea = field.multiline;
    const input = useTextarea ? document.createElement("textarea") : document.createElement("input");
    if (input.tagName === "INPUT") {
      input.type = field.type === "number" ? "number" : "text";
    }
    input.name = field.key;
    input.placeholder = `请输入${field.label}`;
    label.appendChild(span);
    label.appendChild(input);
    createForm.appendChild(label);
  });
  const action = document.createElement("div");
  action.className = "full action-row";
  action.innerHTML = '<button type="submit">创建记录</button>';
  createForm.appendChild(action);
}

function buildStatusSelect() {
  statusSelect.innerHTML = "";
  config.statuses.forEach((status) => {
    const option = document.createElement("option");
    option.value = status;
    option.textContent = displayStatus(status);
    statusSelect.appendChild(option);
  });
}

function buildFilterStrip() {
  filterStrip.innerHTML = "";
  const filters = ["all", ...config.statuses];
  filters.forEach((filter) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `filter-chip${state.activeFilter === filter ? " active" : ""}`;
    chip.textContent = displayStatus(filter);
    chip.addEventListener("click", () => {
      state.activeFilter = filter;
      buildFilterStrip();
      renderRecords();
    });
    filterStrip.appendChild(chip);
  });
}

function renderMetrics() {
  metricsEl.innerHTML = "";
  state.metrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `<span class="metric-label">${metric.label}</span><strong class="metric-value">${metric.value}</strong>`;
    metricsEl.appendChild(card);
  });
}

function renderRecords() {
  recordsEl.innerHTML = "";
  const records = filteredRecords();
  records.forEach((record) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `record-item${record.id === state.activeId ? " active" : ""}`;
    button.innerHTML = `
      <div class="record-head">
        <strong class="record-title">${primaryTitle(record)}</strong>
        <span class="status-badge ${statusClass(record.status)}">${displayStatus(record.status)}</span>
      </div>
      <div class="record-subtitle">${secondaryText(record)}</div>
      <div class="record-scoreline">
        <span>${numericSummary(record)}</span>
        <span>${displayValue(record.reviewer)}</span>
      </div>
      <div class="record-meta">
        <span>${displayValue(record.created_at)}</span>
        <span>${record.id}</span>
      </div>
    `;
    button.addEventListener("click", () => {
      state.activeId = record.id;
      renderRecords();
      renderDetail();
    });
    recordsEl.appendChild(button);
  });
}

function renderDetail() {
  const record = state.records.find((item) => item.id === state.activeId);
  if (!record) {
    emptyState.classList.remove("hidden");
    detailPanel.classList.add("hidden");
    return;
  }
  emptyState.classList.add("hidden");
  detailPanel.classList.remove("hidden");
  detailTitle.textContent = primaryTitle(record);
  detailMeta.textContent = `${config.entity_name} / ${record.id} / 创建于 ${record.created_at || "-"}`;
  detailStatus.textContent = displayStatus(record.status);
  detailStatus.className = `status-badge ${statusClass(record.status)}`;
  recordFields.innerHTML = "";

  config.fields.forEach((field) => {
    const card = document.createElement("article");
    card.className = `detail-block${field.multiline ? " wide" : ""}`;
    const fieldValue = field.key === "status" ? displayStatus(record[field.key]) : displayValue(record[field.key]);
    card.innerHTML = `<strong>${field.label}</strong><div>${fieldValue}</div>`;
    recordFields.appendChild(card);
  });

  const reviewCard = document.createElement("article");
  reviewCard.className = "detail-block wide";
  reviewCard.innerHTML = `
    <strong>评审摘要</strong>
    <div>评审教师：${displayValue(record.reviewer)}</div>
    <div>评审时间：${displayValue(record.reviewed_at)}</div>
    <div>评审意见：${displayValue(record.review_comment)}</div>
  `;
  recordFields.appendChild(reviewCard);

  reviewerInput.value = record.reviewer || "";
  reviewCommentInput.value = record.review_comment || "";
  statusSelect.value = record.status;
}

async function loadDashboard() {
  const data = await requestJson("/api/dashboard");
  state.records = data.records;
  state.metrics = data.metrics;
  if (!state.activeId && state.records.length > 0) {
    state.activeId = state.records[0].id;
  }
  renderMetrics();
  buildFilterStrip();
  renderRecords();
  renderDetail();
}

createForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {};
  config.create_fields.forEach((field) => {
    const input = createForm.querySelector(`[name="${field.key}"]`);
    payload[field.key] = field.type === "number" ? Number(input.value || 0) : input.value.trim();
  });
  try {
    const result = await requestJson("/api/records", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeId = result.record.id;
    createForm.reset();
    createMessage.textContent = "记录已创建。";
    await loadDashboard();
  } catch (error) {
    createMessage.textContent = error.message;
  }
});

reviewForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.activeId) return;
  try {
    await requestJson(`/api/records/${state.activeId}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewer: reviewerInput.value.trim(),
        review_comment: reviewCommentInput.value.trim(),
        status: statusSelect.value,
      }),
    });
    reviewMessage.textContent = "评审已保存。";
    await loadDashboard();
  } catch (error) {
    reviewMessage.textContent = error.message;
  }
});

buildForm();
buildStatusSelect();
loadDashboard();
