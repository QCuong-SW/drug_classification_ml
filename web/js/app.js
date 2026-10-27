/**
 * ML PHARMA LAB — Drug Classification Frontend Application
 * Interacts with FastAPI backend to render each Python ML Stage dynamically.
 */

// Global State
const state = {
  activeTab: "tab-overview",
  dataset: [],
  filteredDataset: [],
  currentPage: 1,
  pageSize: 15,
  charts: {},
  cachedStages: {}
};

// Drug Badges and Colors Mapping
const DRUG_COLORS = {
  drugY: { primary: "#0284c7", bg: "#e0f2fe", border: "#7dd3fc" },
  drugX: { primary: "#10b981", bg: "#d1fae5", border: "#6ee7b7" },
  drugA: { primary: "#ef4444", bg: "#fee2e2", border: "#fca5a5" },
  drugB: { primary: "#f59e0b", bg: "#fef3c7", border: "#fcd34d" },
  drugC: { primary: "#8b5cf6", bg: "#ede9fe", border: "#c4b5fd" }
};

// DOM Content Loaded
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initPredictorControls();
  fetchInitialData();
});

/* ==========================================================================
   Navigation & Tabs
   ========================================================================== */
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      switchToTab(tabId);
    });
  });
}

function switchToTab(tabId) {
  state.activeTab = tabId;

  // Update Nav Items
  document.querySelectorAll(".nav-item").forEach(item => {
    if (item.getAttribute("data-tab") === tabId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update Views
  document.querySelectorAll(".view-tab").forEach(tab => {
    if (tab.id === tabId) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  // Update Breadcrumbs
  const screenTitles = {
    "tab-overview": "Tổng quan & Trình giả lập Dự đoán",
    "tab-stage1": "Stage 01: Khám phá và Thấu hiểu Dữ liệu (stage1_understand_data.py)",
    "tab-stage2": "Stage 02: Xây dựng Mô hình Cơ sở (stage2_baseline_model.py)",
    "tab-stage3": "Stage 03: Đánh giá Đa chiều & Mổ xẻ Mô hình (stage3_evaluation.py)",
    "tab-stage4": "Stage 04: So sánh và Biện luận Mô hình (stage4_model_comparison.py)",
    "tab-stage5": "Stage 05: Quy trình Chuẩn Chuyên nghiệp với Pipeline (stage5_pipeline_workflow.py)",
    "tab-stage6": "Stage 06: Trực quan hóa và Đúc kết Tri thức (stage6_visualization.py)"
  };

  const breadcrumbEl = document.getElementById("current-screen-title");
  if (breadcrumbEl && screenTitles[tabId]) {
    breadcrumbEl.textContent = screenTitles[tabId];
  }

  // Lazy load data for stages if not already loaded
  if (tabId === "tab-stage1" && !state.cachedStages[1]) loadStage1();
  if (tabId === "tab-stage2" && !state.cachedStages[2]) loadStage2();
  if (tabId === "tab-stage3" && !state.cachedStages[3]) loadStage3();
  if (tabId === "tab-stage4" && !state.cachedStages[4]) loadStage4();
  if (tabId === "tab-stage5" && !state.cachedStages[5]) loadStage5();
  if (tabId === "tab-stage6" && !state.cachedStages[6]) loadStage6();

  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ==========================================================================
   Initial Data Fetching
   ========================================================================== */
async function fetchInitialData() {
  try {
    // 1. Fetch backend status
    const statusRes = await fetch("/api/status");
    if (statusRes.ok) {
      const statusData = await statusRes.json();
      const statusLbl = document.getElementById("status-label");
      if (statusLbl) statusLbl.textContent = `Backend Online (${statusData.dataset_rows} mẫu)`;
    }

    // 2. Load Stage 1 & Dataset for initial charts
    await loadStage1();

    // 3. Trigger initial sample prediction
    runPrediction();

  } catch (err) {
    console.error("Lỗi khi kết nối backend:", err);
    showToast("Không thể kết nối đến server Python API. Đang thử lại...", "warning");
  }
}

/* ==========================================================================
   Stage 1: Khám phá Dữ liệu
   ========================================================================== */
async function loadStage1() {
  try {
    const res = await fetch("/api/stage1");
    if (!res.ok) throw new Error("Failed to load stage 1");
    const data = await res.json();
    state.cachedStages[1] = data;

    // Populate Metrics
    document.getElementById("st1-rows").textContent = data.shape.rows;
    document.getElementById("st1-cols").textContent = data.shape.columns;
    document.getElementById("st1-missing").textContent = `${data.null_count} (0.0%)`;
    document.getElementById("st1-duplicates").textContent = `${data.duplicates} dòng`;

    // Render Target Distribution Chart & Bars
    renderTargetDistChart(data.target_distribution);
    renderTargetDistBars(data.target_distribution);

    // Fetch full dataset for the browser table
    const datasetRes = await fetch("/api/dataset?limit=250");
    if (datasetRes.ok) {
      const ds = await datasetRes.json();
      state.dataset = ds.records;
      state.filteredDataset = [...state.dataset];
      initDatasetTable();
    }
  } catch (err) {
    console.error("Error loading Stage 1:", err);
  }
}

function renderTargetDistChart(distList) {
  const canvas = document.getElementById("chart-target-dist");
  if (!canvas) return;

  const labels = distList.map(item => item.drug);
  const counts = distList.map(item => item.count);
  const colors = labels.map(drug => DRUG_COLORS[drug]?.primary || "#64748b");

  if (state.charts["targetDist"]) {
    state.charts["targetDist"].destroy();
  }

  state.charts["targetDist"] = new Chart(canvas, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Số lượng mẫu",
        data: counts,
        backgroundColor: colors,
        borderRadius: 6,
        barPercentage: 0.6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const item = distList[ctx.dataIndex];
              return ` ${item.count} mẫu (${item.percentage}%)`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: "#f1f5f9" }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

function renderTargetDistBars(distList) {
  const container = document.getElementById("st1-target-bars");
  if (!container) return;

  container.innerHTML = distList.map(item => {
    const color = DRUG_COLORS[item.drug]?.primary || "#0284c7";
    return `
      <div class="prob-bar-row">
        <span class="prob-bar-label">${item.drug}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${item.percentage}%; background-color: ${color}"></div>
        </div>
        <span class="prob-bar-value">${item.percentage}%</span>
      </div>
    `;
  }).join("");
}

function initDatasetTable() {
  const searchInput = document.getElementById("dataset-search");
  const prevBtn = document.getElementById("btn-prev-page");
  const nextBtn = document.getElementById("btn-next-page");

  searchInput.addEventListener("input", (e) => {
    const term = e.target.value.toLowerCase().trim();
    if (!term) {
      state.filteredDataset = [...state.dataset];
    } else {
      state.filteredDataset = state.dataset.filter(row => {
        return (
          row.Drug.toLowerCase().includes(term) ||
          row.BP.toLowerCase().includes(term) ||
          row.Sex.toLowerCase().includes(term) ||
          row.Cholesterol.toLowerCase().includes(term) ||
          String(row.Age).includes(term)
        );
      });
    }
    state.currentPage = 1;
    renderDatasetPage();
  });

  prevBtn.addEventListener("click", () => {
    if (state.currentPage > 1) {
      state.currentPage--;
      renderDatasetPage();
    }
  });

  nextBtn.addEventListener("click", () => {
    const maxPage = Math.ceil(state.filteredDataset.length / state.pageSize);
    if (state.currentPage < maxPage) {
      state.currentPage++;
      renderDatasetPage();
    }
  });

  renderDatasetPage();
}

function renderDatasetPage() {
  const tbody = document.getElementById("raw-dataset-tbody");
  const pageInfo = document.getElementById("page-info");
  const prevBtn = document.getElementById("btn-prev-page");
  const nextBtn = document.getElementById("btn-next-page");
  if (!tbody) return;

  const total = state.filteredDataset.length;
  const maxPage = Math.ceil(total / state.pageSize) || 1;
  const startIdx = (state.currentPage - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, total);

  const pageRecords = state.filteredDataset.slice(startIdx, endIdx);

  tbody.innerHTML = pageRecords.map(row => {
    const drugClass = row.Drug.toLowerCase();
    return `
      <tr>
        <td><code>#${row.id}</code></td>
        <td><strong>${row.Age}</strong> tuổi</td>
        <td><span class="badge-neutral">${row.Sex === "F" ? "Nữ (F)" : "Nam (M)"}</span></td>
        <td><span class="badge-neutral">${row.BP}</span></td>
        <td><span class="badge-neutral">${row.Cholesterol}</span></td>
        <td><strong>${row.Na_to_K.toFixed(3)}</strong></td>
        <td><span class="badge-drug ${drugClass}">${row.Drug}</span></td>
      </tr>
    `;
  }).join("");

  pageInfo.textContent = `Hiển thị ${total === 0 ? 0 : startIdx + 1} - ${endIdx} trong ${total} bản ghi`;
  prevBtn.disabled = state.currentPage <= 1;
  nextBtn.disabled = state.currentPage >= maxPage;
}

/* ==========================================================================
   Stage 2: Baseline Model
   ========================================================================== */
async function loadStage2() {
  try {
    const res = await fetch("/api/stage2");
    if (!res.ok) throw new Error("Failed to load stage 2");
    const data = await res.json();
    state.cachedStages[2] = data;

    // Accuracy
    document.getElementById("st2-accuracy").textContent = `${data.baseline_accuracy}%`;

    // Stratified table
    const stratTbody = document.querySelector("#st2-stratified-table tbody");
    if (stratTbody) {
      stratTbody.innerHTML = data.stratified_comparison.map(item => {
        const diff = Math.abs(item.train_pct - item.test_pct).toFixed(1);
        return `
          <tr>
            <td><span class="badge-drug ${item.class.toLowerCase()}">${item.class}</span></td>
            <td><strong>${item.train_pct}%</strong></td>
            <td><strong>${item.test_pct}%</strong></td>
            <td><span class="text-success">±${diff}% (Rất chuẩn)</span></td>
          </tr>
        `;
      }).join("");
    }

    // Encoded badges
    const encodedBadges = document.getElementById("st2-encoded-features");
    if (encodedBadges) {
      encodedBadges.innerHTML = data.encoded_features.map(f => {
        return `<span class="feat-badge num-badge">${f}</span>`;
      }).join(" ");
    }

    // Detailed Test Predictions Table
    const testTbody = document.getElementById("st2-test-tbody");
    if (testTbody) {
      testTbody.innerHTML = data.test_samples.map(sample => {
        const correctBadge = sample.is_correct
          ? `<span class="badge-success">✓ ĐÚNG</span>`
          : `<span class="badge-danger">✗ SAI</span>`;

        const probBars = Object.entries(sample.probabilities).map(([drug, prob]) => {
          const color = DRUG_COLORS[drug]?.primary || "#64748b";
          return `
            <div style="display:flex; align-items:center; gap:6px; font-size:0.7rem; margin-bottom:2px;">
              <span style="width:38px; font-weight:700;">${drug}:</span>
              <div style="flex:1; height:5px; background:#e2e8f0; border-radius:3px; overflow:hidden;">
                <div style="width:${prob}%; height:100%; background:${color};"></div>
              </div>
              <span style="width:36px; text-align:right;">${prob}%</span>
            </div>
          `;
        }).join("");

        return `
          <tr>
            <td><code>#${sample.sample_no}</code></td>
            <td style="font-size:0.78rem;">
              Tuổi: <strong>${sample.patient.Age}</strong>, Sex: ${sample.patient.Sex}, BP: <strong>${sample.patient.BP}</strong>, Chol: ${sample.patient.Cholesterol}, Na/K: <strong>${sample.patient.Na_to_K.toFixed(2)}</strong>
            </td>
            <td><span class="badge-drug ${sample.actual.toLowerCase()}">${sample.actual}</span></td>
            <td><span class="badge-drug ${sample.predicted.toLowerCase()}">${sample.predicted}</span></td>
            <td>${correctBadge}</td>
            <td style="min-width: 170px;">${probBars}</td>
          </tr>
        `;
      }).join("");
    }

  } catch (err) {
    console.error("Error loading Stage 2:", err);
  }
}

/* ==========================================================================
   Stage 3: Đánh giá Đa chiều
   ========================================================================== */
async function loadStage3() {
  try {
    const res = await fetch("/api/stage3");
    if (!res.ok) throw new Error("Failed to load stage 3");
    const data = await res.json();
    state.cachedStages[3] = data;

    // Overall Metrics
    document.getElementById("st3-acc").textContent = `${data.overall_metrics.accuracy}%`;
    document.getElementById("st3-macro-prec").textContent = `${data.overall_metrics.macro_precision}%`;
    document.getElementById("st3-macro-rec").textContent = `${data.overall_metrics.macro_recall}%`;
    document.getElementById("st3-macro-f1").textContent = `${data.overall_metrics.macro_f1}%`;
    document.getElementById("st3-weighted-f1").textContent = `${data.overall_metrics.weighted_f1}%`;

    // Render Confusion Matrix
    renderConfusionMatrix(data.classes, data.confusion_matrix);

    // Classification Report Table
    const reportTbody = document.getElementById("classification-report-tbody");
    if (reportTbody) {
      reportTbody.innerHTML = data.class_metrics.map(m => {
        return `
          <tr>
            <td><span class="badge-drug ${m.class.toLowerCase()}">${m.class}</span></td>
            <td><strong>${m.precision}%</strong></td>
            <td><strong>${m.recall}%</strong></td>
            <td><strong>${m.f1_score}%</strong></td>
            <td>${m.support} bệnh nhân</td>
          </tr>
        `;
      }).join("");
    }

    // Error Analysis Deep Dive
    const errorCountBadge = document.getElementById("st3-error-count");
    if (errorCountBadge) {
      errorCountBadge.textContent = `${data.misclassified_count} ca đoán sai / ${data.total_test_samples} mẫu`;
    }

    const errorsContainer = document.getElementById("st3-errors-container");
    if (errorsContainer) {
      errorsContainer.innerHTML = data.errors.map(err => {
        const probDetails = Object.entries(err.probabilities).map(([drug, p]) => {
          return `<code>${drug}: ${p}%</code>`;
        }).join(" • ");

        return `
          <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:16px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
              <strong>Bệnh nhân dòng #${err.original_row} (Tập Test #${err.test_index + 1}):</strong>
              <div>
                Thực tế: <span class="badge-drug ${err.actual_drug.toLowerCase()}">${err.actual_drug}</span>
                ➜ Mô hình đoán: <span class="badge-drug ${err.predicted_drug.toLowerCase()}">${err.predicted_drug}</span>
              </div>
            </div>
            <div style="font-size:0.8rem; color:#475569; margin-bottom:6px;">
              <strong>Thông số lâm sàng:</strong> Tuổi: ${err.patient_features.Age}, Giới tính: ${err.patient_features.Sex}, BP: ${err.patient_features.BP}, Cholesterol: ${err.patient_features.Cholesterol}, Na/K: ${err.patient_features.Na_to_K}
            </div>
            <div style="font-size:0.78rem; color:#64748b; margin-bottom:8px;">
              <strong>Xác suất mô hình tính ra:</strong> ${probDetails}
            </div>
            <p style="font-size:0.82rem; color:#b91c1c; margin:0;">
              💡 <em>${err.explanation}</em>
            </p>
          </div>
        `;
      }).join("");
    }

  } catch (err) {
    console.error("Error loading Stage 3:", err);
  }
}

function renderConfusionMatrix(classes, matrix) {
  const table = document.getElementById("cm-table");
  if (!table) return;

  let html = `<thead><tr><th>Thực tế \\ Đoán</th>`;
  classes.forEach(c => {
    html += `<th>${c}</th>`;
  });
  html += `</tr></thead><tbody>`;

  for (let i = 0; i < classes.length; i++) {
    html += `<tr><th>${classes[i]}</th>`;
    for (let j = 0; j < classes.length; j++) {
      const val = matrix[i][j];
      let cellClass = "cm-cell-zero";
      if (val > 0) {
        cellClass = (i === j) ? "cm-cell-correct" : "cm-cell-incorrect";
      }
      html += `<td class="${cellClass}">${val}</td>`;
    }
    html += `</tr>`;
  }

  html += `</tbody>`;
  table.innerHTML = html;
}

/* ==========================================================================
   Stage 4: So sánh Mô hình
   ========================================================================== */
async function loadStage4() {
  try {
    const res = await fetch("/api/stage4");
    if (!res.ok) throw new Error("Failed to load stage 4");
    const data = await res.json();
    state.cachedStages[4] = data;

    // Benchmark Table
    const tbody = document.getElementById("benchmark-tbody");
    if (tbody) {
      tbody.innerHTML = data.benchmark.map(m => {
        const gapBadge = m.train_acc === 100 && m.test_acc < 100
          ? `<span class="badge-warning">Overfit nhẹ (${m.overfitting_gap}%)</span>`
          : `<span class="badge-success">Không overfit</span>`;

        return `
          <tr>
            <td><strong>${m.model}</strong></td>
            <td>${m.train_acc}%</td>
            <td><strong>${m.cv_acc_mean}%</strong> <span style="font-size:0.7rem; color:#94a3b8;">±${m.cv_acc_std}</span></td>
            <td><strong>${m.cv_f1_mean}%</strong></td>
            <td><strong class="text-success">${m.test_acc}%</strong></td>
            <td><strong>${m.test_f1}%</strong></td>
            <td>${gapBadge}</td>
          </tr>
        `;
      }).join("");
    }

    // Chart Model Comparison
    renderModelComparisonChart(data.benchmark);

  } catch (err) {
    console.error("Error loading Stage 4:", err);
  }
}

function renderModelComparisonChart(benchmarks) {
  const canvas = document.getElementById("chart-model-comparison");
  if (!canvas) return;

  const models = benchmarks.map(b => b.model);
  const cvAcc = benchmarks.map(b => b.cv_acc_mean);
  const testAcc = benchmarks.map(b => b.test_acc);
  const testF1 = benchmarks.map(b => b.test_f1);

  if (state.charts["modelComp"]) {
    state.charts["modelComp"].destroy();
  }

  state.charts["modelComp"] = new Chart(canvas, {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        {
          label: "5-Fold CV Accuracy (%)",
          data: cvAcc,
          backgroundColor: "#93c5fd",
          borderRadius: 4
        },
        {
          label: "Test Accuracy (%)",
          data: testAcc,
          backgroundColor: "#10b981",
          borderRadius: 4
        },
        {
          label: "Test F1-Macro (%)",
          data: testF1,
          backgroundColor: "#0284c7",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 85,
          max: 100,
          grid: { color: "#f1f5f9" }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

/* ==========================================================================
   Stage 5: Pipeline & GridSearch
   ========================================================================== */
async function loadStage5() {
  try {
    const res = await fetch("/api/stage5");
    if (!res.ok) throw new Error("Failed to load stage 5");
    const data = await res.json();
    state.cachedStages[5] = data;
    // Data is already statically and dynamically structured in HTML template
  } catch (err) {
    console.error("Error loading Stage 5:", err);
  }
}

/* ==========================================================================
   Stage 6: Trực quan hóa
   ========================================================================== */
async function loadStage6() {
  try {
    const res = await fetch("/api/stage6");
    if (!res.ok) throw new Error("Failed to load stage 6");
    const data = await res.json();
    state.cachedStages[6] = data;
  } catch (err) {
    console.error("Error loading Stage 6:", err);
  }
}

/* ==========================================================================
   Interactive Patient Predictor
   ========================================================================== */
function initPredictorControls() {
  const rangeAge = document.getElementById("range-age");
  const valAge = document.getElementById("val-age");
  const rangeNatok = document.getElementById("range-natok");
  const valNatok = document.getElementById("val-natok");
  const btnRun = document.getElementById("btn-run-prediction");

  if (rangeAge && valAge) {
    rangeAge.addEventListener("input", (e) => {
      valAge.textContent = `${e.target.value} tuổi`;
      debouncePrediction();
    });
  }

  if (rangeNatok && valNatok) {
    rangeNatok.addEventListener("input", (e) => {
      valNatok.textContent = `${parseFloat(e.target.value).toFixed(1)}`;
      debouncePrediction();
    });
  }

  // Radio button listeners
  document.querySelectorAll("input[type=radio]").forEach(r => {
    r.addEventListener("change", () => debouncePrediction());
  });

  const modelSelect = document.getElementById("predict-model-select");
  if (modelSelect) {
    modelSelect.addEventListener("change", () => debouncePrediction());
  }

  if (btnRun) {
    btnRun.addEventListener("click", () => runPrediction());
  }

  // Preset Buttons
  const btnPreset138 = document.getElementById("btn-preset-case138");
  if (btnPreset138) {
    btnPreset138.addEventListener("click", () => {
      // Bệnh nhân #138: Age 51, Male, BP HIGH, Cholesterol NORMAL, Na_to_K 11.343 -> Actual DrugB
      setPredictorValues(51, "M", "HIGH", "NORMAL", 11.3);
      showToast("Đã nạp hồ sơ bệnh nhân #138 (Ca khó kiểm định giữa DrugA và DrugB)", "info");
      runPrediction();
    });
  }

  const btnPresetDrugY = document.getElementById("btn-preset-drugy");
  if (btnPresetDrugY) {
    btnPresetDrugY.addEventListener("click", () => {
      setPredictorValues(30, "F", "HIGH", "HIGH", 25.5);
      showToast("Đã nạp hồ sơ Na/K cao (25.5 > 15 ➜ Điển hình cho DrugY)", "info");
      runPrediction();
    });
  }
}

function setPredictorValues(age, sex, bp, chol, natok) {
  document.getElementById("range-age").value = age;
  document.getElementById("val-age").textContent = `${age} tuổi`;

  document.getElementById("range-natok").value = natok;
  document.getElementById("val-natok").textContent = `${natok.toFixed(1)}`;

  const sexRadio = document.querySelector(`input[name="pred-sex"][value="${sex}"]`);
  if (sexRadio) sexRadio.checked = true;

  const bpRadio = document.querySelector(`input[name="pred-bp"][value="${bp}"]`);
  if (bpRadio) bpRadio.checked = true;

  const cholRadio = document.querySelector(`input[name="pred-chol"][value="${chol}"]`);
  if (cholRadio) cholRadio.checked = true;
}

let debounceTimer = null;
function debouncePrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    runPrediction();
  }, 120);
}

async function runPrediction() {
  try {
    const age = parseFloat(document.getElementById("range-age")?.value || 45);
    const natok = parseFloat(document.getElementById("range-natok")?.value || 15.5);
    const sex = document.querySelector('input[name="pred-sex"]:checked')?.value || "F";
    const bp = document.querySelector('input[name="pred-bp"]:checked')?.value || "HIGH";
    const cholesterol = document.querySelector('input[name="pred-chol"]:checked')?.value || "HIGH";
    const model_name = document.getElementById("predict-model-select")?.value || "pipeline_rf";

    const payload = { age, sex, bp, cholesterol, na_to_k: natok, model_name };

    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Prediction API call failed");
    const result = await res.json();

    // Render results
    renderPredictionResult(result);

  } catch (err) {
    console.error("Lỗi suy luận dự đoán:", err);
  }
}

function renderPredictionResult(res) {
  const drugBadge = document.getElementById("result-drug-badge");
  const drugName = document.getElementById("result-drug-name");
  const confidence = document.getElementById("result-confidence");
  const explanation = document.getElementById("result-explanation");
  const probBarsContainer = document.getElementById("result-prob-bars");

  if (!drugBadge || !drugName) return;

  const colorToken = DRUG_COLORS[res.prediction] || DRUG_COLORS["drugY"];

  drugName.textContent = res.prediction;
  drugBadge.style.backgroundColor = colorToken.bg;
  drugBadge.style.borderColor = colorToken.border;
  drugName.style.color = colorToken.primary;

  confidence.textContent = `Độ tin cậy: ${res.confidence}%`;
  confidence.style.color = colorToken.primary;

  explanation.textContent = res.explanation;

  // Render horizontal probability bars
  if (probBarsContainer && res.probabilities) {
    probBarsContainer.innerHTML = Object.entries(res.probabilities).map(([drug, prob]) => {
      const color = DRUG_COLORS[drug]?.primary || "#64748b";
      return `
        <div class="prob-bar-row">
          <span class="prob-bar-label">${drug}</span>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${prob}%; background-color: ${color}"></div>
          </div>
          <span class="prob-bar-value">${prob}%</span>
        </div>
      `;
    }).join("");
  }
}

/* ==========================================================================
   Live Script Runner
   ========================================================================== */
async function runStageScript(stageNum) {
  const terminal = document.getElementById(`terminal-stage${stageNum}`);
  const outputEl = document.getElementById(`terminal-output-${stageNum}`);

  if (terminal && outputEl) {
    terminal.style.display = "block";
    outputEl.textContent = `[PYTHON RUNNER] Đang khởi chạy: python src/stage${stageNum}_*.py ...\n`;
    terminal.scrollIntoView({ behavior: "smooth" });
  }

  showToast(`Đang chạy trực tiếp Stage ${stageNum}...`, "info");

  try {
    const res = await fetch(`/api/run-stage/${stageNum}`, { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Script execution failed");
    }

    if (outputEl) {
      outputEl.textContent = data.output;
    }

    showToast(`✓ Đã hoàn thành thực thi Stage ${stageNum}!`, "success");

    // Invalidate cached data and re-fetch for this stage
    delete state.cachedStages[stageNum];
    if (stageNum === 1) loadStage1();
    if (stageNum === 2) loadStage2();
    if (stageNum === 3) loadStage3();
    if (stageNum === 4) loadStage4();
    if (stageNum === 5) loadStage5();
    if (stageNum === 6) {
      loadStage6();
      // Reload images with timestamp to avoid browser cache
      for (let i = 1; i <= 4; i++) {
        const img = document.getElementById(`fig-img-${i}`);
        if (img) {
          const currentSrc = img.src.split("?")[0];
          img.src = `${currentSrc}?t=${Date.now()}`;
        }
      }
    }

  } catch (err) {
    console.error(`Error running stage ${stageNum}:`, err);
    if (outputEl) {
      outputEl.textContent += `\n[LỖI]: ${err.message}\n`;
    }
    showToast(`Lỗi khi chạy Stage ${stageNum}: ${err.message}`, "danger");
  }
}

function closeTerminal(stageNum) {
  const terminal = document.getElementById(`terminal-stage${stageNum}`);
  if (terminal) terminal.style.display = "none";
}

/* ==========================================================================
   Image Modal (Lightbox)
   ========================================================================== */
function openImageModal(imgSrc, title) {
  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img-element");
  const modalTitle = document.getElementById("modal-img-title");

  if (modal && modalImg) {
    modalImg.src = imgSrc;
    if (modalTitle) modalTitle.textContent = title || "Xem biểu đồ";
    modal.classList.add("open");
  }
}

function closeImageModal() {
  const modal = document.getElementById("img-modal");
  if (modal) modal.classList.remove("open");
}

/* ==========================================================================
   Toast Notification System
   ========================================================================== */
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";

  if (type === "success") toast.style.borderLeftColor = "var(--success)";
  if (type === "warning") toast.style.borderLeftColor = "var(--warning)";
  if (type === "danger") toast.style.borderLeftColor = "var(--danger)";

  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(20px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
