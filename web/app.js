/* -----------------------------------------------------------------------------
 * Autonomous Coding Agent Dashboard Logic (Phase 16I)
 * ----------------------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
  const taskForm = document.getElementById("task-form");
  const taskGoalInput = document.getElementById("task-goal");
  const workspaceInput = document.getElementById("workspace-root");
  const runBtn = document.getElementById("run-btn");
  const recentTasksBody = document.getElementById("recent-tasks-body");
  const refreshHistoryBtn = document.getElementById("refresh-history-btn");
  const activeStatusBadge = document.getElementById("active-status-badge");
  const activeTaskContainer = document.getElementById("active-task-container");
  const finalReportCard = document.getElementById("final-report-card");
  const reportOutcomeBadge = document.getElementById("report-outcome-badge");
  const reportContent = document.getElementById("report-content");
  const traceTimeline = document.getElementById("trace-timeline");
  const healthStatus = document.getElementById("health-status");

  // Health check on load
  checkHealth();
  // Fetch initial history
  fetchTaskHistory();

  // Task form submission handler
  taskForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const goal = taskGoalInput.value.trim();
    const workspace = workspaceInput.value.trim() || ".";

    if (!goal) return;

    setLoading(true);
    updateActiveTaskUI({ goal, workspace_root: workspace, status: "RUNNING" });

    try {
      const response = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal, workspace_root: workspace }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Task submission failed.");
      }

      const taskData = await response.json();
      renderActiveTaskSuccess(taskData);
      renderFinalReport(taskData.final_report, taskData.task_id);
      fetchTaskHistory();
    } catch (err) {
      renderActiveTaskError(err.message);
    } finally {
      setLoading(false);
    }
  });

  // Refresh history button
  refreshHistoryBtn.addEventListener("click", fetchTaskHistory);

  // Health check fetcher
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        healthStatus.textContent = "System Ready";
      } else {
        healthStatus.textContent = "System Offline";
      }
    } catch {
      healthStatus.textContent = "System Offline";
    }
  }

  // Fetch task history list
  async function fetchTaskHistory() {
    try {
      const res = await fetch("/api/tasks");
      if (!res.ok) return;
      const tasks = await res.json();
      renderTaskHistory(tasks);
    } catch (err) {
      console.error("Failed to load task history:", err);
    }
  }

  // Render recent tasks table
  function renderTaskHistory(tasks) {
    if (!tasks || tasks.length === 0) {
      recentTasksBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="3">No previous execution history found</td>
        </tr>
      `;
      return;
    }

    recentTasksBody.innerHTML = tasks.map(t => {
      const statusBadge = getStatusBadgeHTML(t.final_outcome || t.status);
      const shortGoal = escapeHTML(t.goal.length > 35 ? t.goal.substring(0, 35) + "..." : t.goal);
      return `
        <tr data-task-id="${escapeHTML(t.task_id)}" class="task-row">
          <td class="font-mono">${escapeHTML(t.task_id)}</td>
          <td>${shortGoal}</td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }).join("");

    // Add click listeners to rows to view details
    document.querySelectorAll(".task-row").forEach(row => {
      row.addEventListener("click", () => {
        const taskId = row.getAttribute("data-task-id");
        loadTaskDetails(taskId);
      });
    });
  }

  // Load specific task details
  async function loadTaskDetails(taskId) {
    try {
      const res = await fetch(`/api/tasks/${taskId}`);
      if (!res.ok) return;
      const data = await res.json();
      renderFinalReport(data.final_report, data.task_id);
      renderTraceEvents(data.execution_trace || []);
    } catch (err) {
      console.error("Failed to fetch task details:", err);
    }
  }

  // UI state updates for active execution
  function updateActiveTaskUI(task) {
    activeStatusBadge.className = "badge badge-running";
    activeStatusBadge.textContent = "RUNNING";

    activeTaskContainer.innerHTML = `
      <div class="active-task-detail">
        <p><strong>Goal:</strong> ${escapeHTML(task.goal)}</p>
        <p><strong>Workspace:</strong> <code>${escapeHTML(task.workspace_root)}</code></p>
        
        <div class="stage-checklist">
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Understanding</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Planning</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Retrieval</div>
          <div class="stage-item stage-active"><span class="stage-icon">→</span> Testing & Modification</div>
          <div class="stage-item stage-pending"><span class="stage-icon">○</span> Verification</div>
        </div>
      </div>
    `;
  }

  function renderActiveTaskSuccess(data) {
    const outcome = data.final_outcome || "SUCCESS";
    activeStatusBadge.className = `badge badge-${outcome.toLowerCase()}`;
    activeStatusBadge.textContent = outcome;

    activeTaskContainer.innerHTML = `
      <div class="active-task-detail">
        <p><strong>Goal:</strong> ${escapeHTML(data.goal)}</p>
        <p><strong>Task ID:</strong> <code>${escapeHTML(data.task_id)}</code></p>

        <div class="stage-checklist">
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Understanding</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Planning</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Retrieval</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Testing & Modification</div>
          <div class="stage-item stage-complete"><span class="stage-icon">✓</span> Verification</div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-val">${data.final_report.tool_calls || 0}</div>
            <div class="metric-label">Tool Calls</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">${data.final_report.recovery_retries || 0}</div>
            <div class="metric-label">Retries</div>
          </div>
          <div class="metric-card">
            <div class="metric-val">${data.final_report.files_modified_count || 0}</div>
            <div class="metric-label">Files Modified</div>
          </div>
        </div>
      </div>
    `;
  }

  function renderActiveTaskError(errorMsg) {
    activeStatusBadge.className = "badge badge-failed";
    activeStatusBadge.textContent = "FAILED";

    activeTaskContainer.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <p class="text-danger">Execution Failed: ${escapeHTML(errorMsg)}</p>
      </div>
    `;
  }

  // Render Final Report Card
  function renderFinalReport(report, taskId) {
    if (!report) return;
    finalReportCard.classList.remove("hidden");

    const outcome = report.status || "UNKNOWN";
    reportOutcomeBadge.className = `badge badge-${outcome.toLowerCase()}`;
    reportOutcomeBadge.textContent = outcome;

    reportContent.innerHTML = `
      <div>
        <p><strong>Status:</strong> ${escapeHTML(outcome)}</p>
        <p><strong>Task ID:</strong> <code>${escapeHTML(taskId || "N/A")}</code></p>
        <p><strong>Goal:</strong> ${escapeHTML(report.goal || "N/A")}</p>
        <p><strong>Tests:</strong> ${escapeHTML(report.tests || "N/A")}</p>
        <p><strong>Goal Verification:</strong> ${escapeHTML(report.goal_verification || "N/A")}</p>
      </div>
      <div>
        <p><strong>Files Modified:</strong> ${report.files_modified_count || 0}</p>
        <p><strong>Tool Calls:</strong> ${report.tool_calls || 0}</p>
        <p><strong>Validation Attempts:</strong> ${report.validation_attempts || 0}</p>
        <p><strong>Recovery Retries:</strong> ${report.recovery_retries || 0}</p>
        <p><strong>Execution Time:</strong> ${report.execution_time_seconds ? report.execution_time_seconds.toFixed(1) + "s" : "N/A"}</p>
      </div>
    `;

    if (report.reason) {
      reportContent.innerHTML += `
        <div style="grid-column: 1 / -1; margin-top: 12px;" class="alert alert-warning">
          <strong>Reason:</strong> ${escapeHTML(report.reason)}
        </div>
      `;
    }
  }

  function renderTraceEvents(events) {
    if (!events || events.length === 0) {
      traceTimeline.innerHTML = `<p class="text-muted">No trace events recorded.</p>`;
      return;
    }

    traceTimeline.innerHTML = events.map(ev => `
      <div class="trace-event">
        <span class="trace-type">[${escapeHTML(ev.event_type || "event")}]</span> 
        <span class="trace-status">${escapeHTML(ev.status || "")}</span>
        <span>${ev.metadata ? escapeHTML(JSON.stringify(ev.metadata)) : ""}</span>
      </div>
    `).join("");
  }

  function setLoading(loading) {
    const btnText = runBtn.querySelector(".btn-text");
    const spinner = runBtn.querySelector(".spinner");

    if (loading) {
      runBtn.disabled = true;
      btnText.textContent = "RUNNING...";
      if (spinner) spinner.classList.remove("hidden");
    } else {
      runBtn.disabled = false;
      btnText.textContent = "RUN AGENT";
      if (spinner) spinner.classList.add("hidden");
    }
  }

  function getStatusBadgeHTML(status) {
    const st = (status || "").toLowerCase();
    return `<span class="badge badge-${st}">${escapeHTML(status || "UNKNOWN")}</span>`;
  }

  function escapeHTML(str) {
    if (!str) return "";
    return str.replace(/[&<>'"]/g, 
      tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
    );
  }
});
