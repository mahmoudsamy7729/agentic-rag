(function (window) {
  "use strict";

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatDateTime(value) {
    if (!value) return "-";
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
  }

  function formatDate(value) {
    if (!value) return "-";
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleDateString();
  }

  function formatMetric(value, digits) {
    if (value === null || value === undefined || value === "") return "-";
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) return "-";
    return numeric.toFixed(digits === undefined ? 3 : digits);
  }

  function formatPercent(value, digits) {
    if (value === null || value === undefined || value === "") return "-";
    var numeric = Number(value);
    if (!Number.isFinite(numeric)) return "-";
    return (numeric * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function formatJudgeScore(value) {
    if (value === null || value === undefined || value === "") return "-";
    return String(value) + "/5";
  }

  function formatCount(value) {
    var numeric = Number(value || 0);
    return Number.isFinite(numeric) ? numeric.toLocaleString() : "-";
  }

  function compareIdsToQuery(ids) {
    return "/evaluations-compare-ui?runs=" + encodeURIComponent(ids.join(","));
  }

  function parseCompareIds() {
    var params = new URLSearchParams(window.location.search);
    var raw = (params.get("runs") || "").trim();
    if (!raw) return [];
    return raw.split(",").map(function (item) { return item.trim(); }).filter(Boolean);
  }

  function uniq(values) {
    return Array.from(new Set((values || []).filter(Boolean)));
  }

  function renderBadge(status) {
    var safe = escapeHtml(status || "unknown");
    var normalized = String(status || "queued").toLowerCase();
    return '<span class="badge badge-' + safeBadgeClass(normalized) + '">' + safe + "</span>";
  }

  function safeBadgeClass(status) {
    if (status === "running") return "running";
    if (status === "completed") return "completed";
    if (status === "failed") return "failed";
    return "queued";
  }

  function chips(values, emptyLabel) {
    var items = (values || []).filter(Boolean);
    if (!items.length) {
      return '<span class="chip">' + escapeHtml(emptyLabel || "None") + "</span>";
    }
    return items.map(function (value) {
      return '<span class="chip">' + escapeHtml(value) + "</span>";
    }).join("");
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function createMetricChart(rows, formatter) {
    if (!rows || !rows.length) {
      return '<div class="empty-state">No data available.</div>';
    }
    var max = rows.reduce(function (current, row) {
      return Math.max(current, Number(row.value || 0));
    }, 0);
    return rows.map(function (row) {
      var numeric = Number(row.value || 0);
      var pct = max > 0 ? clamp((numeric / max) * 100, 0, 100) : 0;
      return [
        '<div class="chart-row">',
        '  <div class="chart-label-line">',
        "    <span>" + escapeHtml(row.label) + "</span>",
        "    <strong>" + escapeHtml(formatter ? formatter(numeric) : formatMetric(numeric)) + "</strong>",
        "  </div>",
        '  <div class="chart-bar"><span style="width:' + pct.toFixed(2) + '%"></span></div>',
        "</div>",
      ].join("");
    }).join("");
  }

  function average(values) {
    var nums = (values || []).filter(function (value) {
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    }).map(Number);
    if (!nums.length) return null;
    return nums.reduce(function (total, value) { return total + value; }, 0) / nums.length;
  }

  function compositeCaseScore(item) {
    var pieces = [
      item.hit_at_k,
      item.recall_at_k,
      item.precision_at_k,
      item.mrr,
      item.keyword_coverage,
      item.context_relevance_score === null || item.context_relevance_score === undefined
        ? null
        : Number(item.context_relevance_score) / 5,
    ];
    return average(pieces);
  }

  function isLowScore(item) {
    var score = compositeCaseScore(item);
    if (score === null) return false;
    return score < 0.55;
  }

  function downloadUrl(datasetSha) {
    return "/evaluation-datasets/" + encodeURIComponent(datasetSha) + "/download";
  }

  function setBusy(button, busy, busyText) {
    if (!button) return;
    if (!button.dataset.defaultText) {
      button.dataset.defaultText = button.textContent || "";
    }
    button.disabled = Boolean(busy);
    button.textContent = busy ? (busyText || "Working...") : button.dataset.defaultText;
  }

  window.evalAdmin = {
    average: average,
    chips: chips,
    compareIdsToQuery: compareIdsToQuery,
    compositeCaseScore: compositeCaseScore,
    createMetricChart: createMetricChart,
    downloadUrl: downloadUrl,
    escapeHtml: escapeHtml,
    formatCount: formatCount,
    formatDate: formatDate,
    formatDateTime: formatDateTime,
    formatJudgeScore: formatJudgeScore,
    formatMetric: formatMetric,
    formatPercent: formatPercent,
    isLowScore: isLowScore,
    parseCompareIds: parseCompareIds,
    renderBadge: renderBadge,
    setBusy: setBusy,
    uniq: uniq,
  };
})(window);
