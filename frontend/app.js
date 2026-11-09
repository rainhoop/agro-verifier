(() => {
  "use strict";

  /* ======================================================================
     Constants and state
     ====================================================================== */

  const CROPS = ["wheat", "barley", "sunflower", "rapso"];
  const RISK_ORDER = {
    crop_mismatch: 0,
    not_found: 1,
    needs_check: 2,
    doubtful: 3,
    confirmed: 4,
  };
  const DEFAULT_AREA_MIN = 10;
  const DEFAULT_AREA_MAX = 5000;
  const STORAGE_KEY = "agro-verifier-state-v1";
  const SATELLITE_URL =
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  const SCHEME_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
  const SATELLITE_ATTRIBUTION =
    'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics';
  const SCHEME_ATTRIBUTION =
    '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';
  const REDUCED_MOTION =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const COARSE_POINTER =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches;

  const state = {
    lang: "ru",
    i18n: {},
    config: null,
    demoFields: [],
    demoMeta: null,
    demoLang: null,
    demoLoadPromise: null,
    selectedDemoId: null,
    selectedField: null,
    geojson: null,
    drawnAreaHa: null,
    drawnVertices: 0,
    map: null,
    satelliteLayer: null,
    schemeLayer: null,
    basemap: "satellite",
    demoLayers: new Map(),
    finalLayer: null,
    portfolioMarker: null,
    searchMarker: null,
    mode: null,
    points: [],
    rectAnchor: null,
    drawGroup: null,
    markerGroup: null,
    markers: [],
    shapeLayers: [],
    cursorLine: null,
    rectPreview: null,
    rectAnchorMarker: null,
    chart: null,
    lastResult: null,
    lastResultParams: null,
    temporalIndex: 0,
    temporalMode: "ndvi",
    temporalPlayback: null,
    chatSessionId: null,
    chatFreeTextEnabled: false,
    chatBusy: false,
    chatMessages: [],
    loading: false,
    verifySeq: 0,
    noticeTimer: null,
    toastTimer: null,
    drawHintTimer: null,
    drawSheetOffset: 0,
    tileErrorShown: false,
    searchSeq: 0,
    searchTimer: null,
    searchResults: [],
    searchIndex: -1,
    activeTab: "check",
    activeResultView: "result",
    batchRows: [],
    batchMeta: null,
    batchInputRows: null,
    batchFile: null,
    batchLoading: false,
    portfolioLoaded: false,
    lastBatchAt: null,
    sortKey: "verdict",
    sortDir: "asc",
    portfolioView: [],
    effectTimer: null,
    restoredMapView: null,
    restoredTemporal: null,
  };

  /* ======================================================================
     Small helpers
     ====================================================================== */

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function locale() {
    return state.lang === "kz" ? "kk-KZ" : "ru-RU";
  }

  function t(key, vars) {
    const dict = state.i18n[state.lang] || {};
    let value = Object.prototype.hasOwnProperty.call(dict, key) ? dict[key] : key;
    if (typeof value !== "string") return value;
    if (vars) {
      Object.keys(vars).forEach((name) => {
        value = value.split("{" + name + "}").join(String(vars[name]));
      });
    }
    return value;
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtNum(value, digits = 1) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return number.toLocaleString(locale(), {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtInt(value) {
    return fmtNum(value, 0);
  }

  function fmtDate(iso, withYear = true) {
    if (!iso) return "—";
    const date = new Date(String(iso).slice(0, 10) + "T00:00:00");
    if (Number.isNaN(date.getTime())) return "—";
    const options = { day: "2-digit", month: "2-digit" };
    if (withYear) options.year = "numeric";
    return date.toLocaleDateString(locale(), options);
  }

  function fmtPercent(probability) {
    const number = Number(probability);
    if (!Number.isFinite(number)) return "—";
    return Math.round(number * 100) + " %";
  }

  function shiftLabel(days) {
    if (days == null || days === "") return "—";
    const number = Number(days);
    if (!Number.isFinite(number)) return "—";
    const sign = number > 0 ? "+" : number < 0 ? "−" : "";
    return sign + Math.abs(number) + " " + t("features.days");
  }

  function parseNumber(value, fallback = 0) {
    const number = Number(String(value == null ? "" : value).replace(",", "."));
    return Number.isFinite(number) ? number : fallback;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function showToast(message) {
    const node = $("#toast");
    if (!node) return;
    node.textContent = message;
    node.classList.remove("hidden");
    if (state.toastTimer) window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => {
      node.classList.add("hidden");
    }, 4200);
  }

  function setHidden(node, hidden) {
    if (!node) return;
    node.classList.toggle("hidden", Boolean(hidden));
  }

  async function fetchJSON(url, options = {}) {
    const headers = Object.assign({ Accept: "application/json" }, options.headers || {});
    const response = await fetch(url, Object.assign({}, options, { headers }));
    if (!response.ok) {
      let message = response.status + " " + (response.statusText || "");
      try {
        const payload = await response.json();
        if (payload && payload.detail) {
          message =
            typeof payload.detail === "string"
              ? payload.detail
              : JSON.stringify(payload.detail);
        }
      } catch (error) {
        /* keep the status message */
      }
      const error = new Error(message.trim());
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function areaConfig() {
    const area = (state.config && state.config.area) || {};
    return {
      min: Number.isFinite(Number(area.min_ha)) ? Number(area.min_ha) : DEFAULT_AREA_MIN,
      max: Number.isFinite(Number(area.max_ha)) ? Number(area.max_ha) : DEFAULT_AREA_MAX,
    };
  }

  const saveAppStateSoon = debounce(saveAppState, 250);

  /* ======================================================================
     Domain helpers: geodesic area and simple polygon test
     ====================================================================== */

  const EARTH_RADIUS_M = 6371008.8;

  function ringAreaHa(points) {
    if (!Array.isArray(points) || points.length < 3) return 0;
    let total = 0;
    for (let index = 0; index < points.length; index += 1) {
      const first = points[index];
      const second = points[(index + 1) % points.length];
      const lon1 = (first.lng * Math.PI) / 180;
      const lon2 = (second.lng * Math.PI) / 180;
      const lat1 = (first.lat * Math.PI) / 180;
      const lat2 = (second.lat * Math.PI) / 180;
      total += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
    }
    return Math.abs((total * EARTH_RADIUS_M * EARTH_RADIUS_M) / 2) / 10000;
  }

  function orientation(a, b, c) {
    const value = (b.lng - a.lng) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lng - a.lng);
    if (Math.abs(value) < 1e-14) return 0;
    return value > 0 ? 1 : 2;
  }

  function onSegment(a, b, c) {
    return (
      Math.min(a.lng, c.lng) - 1e-14 <= b.lng &&
      b.lng <= Math.max(a.lng, c.lng) + 1e-14 &&
      Math.min(a.lat, c.lat) - 1e-14 <= b.lat &&
      b.lat <= Math.max(a.lat, c.lat) + 1e-14
    );
  }

  function segmentsIntersect(a, b, c, d) {
    const o1 = orientation(a, b, c);
    const o2 = orientation(a, b, d);
    const o3 = orientation(c, d, a);
    const o4 = orientation(c, d, b);
    if (o1 !== o2 && o3 !== o4) return true;
    if (o1 === 0 && onSegment(a, c, b)) return true;
    if (o2 === 0 && onSegment(a, d, b)) return true;
    if (o3 === 0 && onSegment(c, a, d)) return true;
    if (o4 === 0 && onSegment(c, b, d)) return true;
    return false;
  }

  function polygonIsSimple(points) {
    const count = points.length;
    if (count < 4) return true;
    for (let first = 0; first < count; first += 1) {
      const a = points[first];
      const b = points[(first + 1) % count];
      for (let second = first + 1; second < count; second += 1) {
        if (second === first + 1 || (first === 0 && second === count - 1)) continue;
        const c = points[second];
        const d = points[(second + 1) % count];
        if (segmentsIntersect(a, b, c, d)) return false;
      }
    }
    return true;
  }

  function polylineIsSimple(points) {
    for (let first = 0; first < points.length - 1; first += 1) {
      const a = points[first];
      const b = points[first + 1];
      for (let second = first + 2; second < points.length - 1; second += 1) {
        const c = points[second];
        const d = points[second + 1];
        if (segmentsIntersect(a, b, c, d)) return false;
      }
    }
    return true;
  }

  function roundCoordinate(value) {
    return Math.round(value * 1e7) / 1e7;
  }

  function geojsonFromPoints(points) {
    const ring = points.map((point) => [
      roundCoordinate(point.lng),
      roundCoordinate(point.lat),
    ]);
    ring.push(ring[0].slice());
    return { type: "Polygon", coordinates: [ring] };
  }

  /* ======================================================================
     DOM references
     ====================================================================== */

  const els = {
    tabButtons: $$(".tab"),
    panels: { check: $("#tab-check"), portfolio: $("#tab-portfolio") },
    resultView: $("#result-view"),
    resultViewButtons: $$(".result-tab"),
    assistantView: $("#assistant-view"),
    assistantContext: $("#assistant-context"),
    assistantQuestions: $$("#assistant-questions [data-chat-question]"),
    assistantMessages: $("#assistant-messages"),
    assistantLoading: $("#assistant-loading"),
    assistantError: $("#assistant-error"),
    assistantForm: $("#assistant-form"),
    assistantInput: $("#assistant-input"),
    assistantSend: $("#assistant-send"),
    assistantLimit: $("#assistant-limit"),
    langButtons: $$(".lang-switch [data-lang]"),
    sidePanel: $("#side-panel"),
    fieldCard: $("#field-card"),
    fieldName: $("#field-name"),
    fieldMeta: $("#field-meta"),
    fieldHonesty: $("#field-honesty"),
    fieldSource: $("#field-source"),
    fieldError: $("#field-error"),
    drawPolygon: $("#draw-polygon"),
    drawRectangle: $("#draw-rectangle"),
    demoToggle: $("#demo-toggle"),
    demoPanel: $("#demo-panel"),
    demoClose: $("#demo-close"),
    demoList: $("#demo-list"),
    clearField: $("#clear-field"),
    cropSelect: $("#crop-select"),
    sowingDate: $("#sowing-date"),
    verifyForm: $("#verify-form"),
    verifyButton: $("#verify-button"),
    verifySpinner: $("#verify-spinner"),
    resultStep: $("#result-step"),
    resultEmpty: $("#result-empty"),
    resultLoading: $("#result-loading"),
    resultError: $("#result-error"),
    resultErrorText: $("#result-error-text"),
    resultContent: $("#result-content"),
    retryButton: $("#retry-button"),
    verdictDot: $("#verdict-dot"),
    verdictTitle: $("#verdict-title"),
    scoreValue: $("#score-value"),
    verdictSummary: $("#verdict-summary"),
    modelConfidence: $("#model-confidence"),
    modelConfidenceNote: $("#model-confidence-note"),
    modelCrop: $("#model-crop"),
    mlSource: $("#ml-source"),
    guardNote: $("#guard-note"),
    sourceRow: $(".source-row"),
    sourceLabel: $("#source-label"),
    sourceNote: $("#source-note"),
    chartCanvas: $("#ndvi-chart"),
    temporalPanel: $("#temporal-panel"),
    temporalPlay: $("#temporal-play"),
    temporalSlider: $("#temporal-slider"),
    temporalDate: $("#temporal-date"),
    temporalValue: $("#temporal-value"),
    temporalStage: $("#temporal-stage"),
    temporalSource: $("#temporal-source"),
    temporalRestored: $("#temporal-restored"),
    temporalMarkers: $("#temporal-markers"),
    temporalMonths: $("#temporal-months"),
    temporalModeGroup: $(".temporal-mode"),
    temporalModeButtons: $$(".temporal-mode [data-temporal-mode]"),
    temporalLegend: $("#temporal-legend"),
    temporalLegendTitle: $("#temporal-legend-title"),
    temporalLegendColors: $("#temporal-legend-colors"),
    temporalLegendMin: $("#temporal-legend-min"),
    temporalLegendMax: $("#temporal-legend-max"),
    featuresGrid: $("#features-grid"),
    explanation: $("#explanation"),
    explanationSource: $("#explanation-source"),
    reasons: $("#reasons"),
    recommendation: $("#recommendation"),
    top3List: $("#top3-list"),
    weatherSource: $("#weather-source"),
    weatherChips: $("#weather-chips"),
    weatherContext: $("#weather-context"),
    pdfButton: $("#pdf-button"),
    howContent: $("#how-content"),
    resultDisclaimer: $("#result-disclaimer"),
    map: $("#map"),
    mapShell: $("#map-shell"),
    mapHint: $("#map-hint"),
    mapNotice: $("#map-notice"),
    drawToolbar: $("#draw-toolbar"),
    drawCount: $("#draw-count"),
    undoPoint: $("#undo-point"),
    finishDraw: $("#finish-draw"),
    cancelDraw: $("#cancel-draw"),
    drawSheet: $("#draw-sheet"),
    drawSheetHandle: $("#draw-sheet-handle"),
    drawSheetTitle: $("#draw-sheet-title"),
    drawSheetCount: $("#draw-sheet-count"),
    sheetUndo: $("#sheet-undo"),
    sheetFinish: $("#sheet-finish"),
    sheetCancel: $("#sheet-cancel"),
    searchInput: $("#place-search"),
    searchResults: $("#search-results"),
    basemapButtons: $$(".basemap-toggle [data-basemap]"),
    loadSample: $("#load-sample"),
    csvInput: $("#csv-input"),
    csvTemplate: $("#csv-template"),
    exportCsv: $("#export-csv"),
    verdictFilter: $("#verdict-filter"),
    portfolioBody: $("#portfolio-body"),
    portfolioCount: $("#portfolio-count"),
    portfolioSource: $("#portfolio-source"),
    portfolioNote: $("#portfolio-note"),
    portfolioTimestamp: $("#portfolio-timestamp"),
    portfolioLoading: $("#portfolio-loading"),
    portfolioEmpty: $("#portfolio-empty"),
    portfolioTableWrap: $(".table-wrap"),
    portfolioTable: $("#portfolio-table"),
    effectVisit: $("#effect-visit"),
    effectRemote: $("#effect-remote"),
    effectFields: $("#effect-fields"),
    effectSavings: $("#effect-savings"),
    effectFormula: $("#effect-formula"),
    printReport: $("#print-report"),
  };

  /* ======================================================================
     Internationalisation
     ====================================================================== */

  function applyStaticI18n() {
    document.documentElement.lang = state.lang;
    $$("[data-i18n]").forEach((node) => {
      const value = t(node.dataset.i18n);
      if (typeof value === "string") node.textContent = value;
    });
    $$("[data-i18n-placeholder]").forEach((node) => {
      const value = t(node.dataset.i18nPlaceholder);
      if (typeof value === "string") node.placeholder = value;
    });
    $$("[data-i18n-aria-label]").forEach((node) => {
      const value = t(node.dataset.i18nAriaLabel);
      if (typeof value === "string") node.setAttribute("aria-label", value);
    });
    els.langButtons.forEach((button) => {
      const active = button.dataset.lang === state.lang;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    els.searchInput.setAttribute("aria-label", t("map.search"));
    els.csvTemplate.href = "/api/sample-batch?format=csv&lang=" + state.lang;

    if (state.selectedField && !state.mode) {
      updateFieldCardDemo(state.selectedField);
    } else if (state.geojson && !state.mode) {
      updateDrawnFieldCard();
    } else if (state.mode) {
      updateDrawingCard();
    } else {
      resetFieldCardEmpty();
    }
    if (state.batchRows.length) renderPortfolio();
  }

  /* ======================================================================
     Initialisation
     ====================================================================== */

  document.addEventListener("DOMContentLoaded", init);

  async function init() {
    restoreAppState();
    bindEvents();
    renderHowContent();
    updateEffect();

    const i18nPromise = fetchJSON("/i18n.json").catch((error) => {
      console.error("i18n unavailable", error);
      return null;
    });
    const configPromise = fetchJSON("/api/config").catch(() => null);
    let requestedChatSession = "";
    try {
      requestedChatSession =
        state.chatSessionId ||
        window.localStorage.getItem("agro-chat-session") ||
        window.sessionStorage.getItem("agro-chat-session") ||
        "";
    } catch (error) {
      console.warn("Chat session storage is unavailable");
    }
    const chatConfigPromise = fetchJSON(
      "/api/chat/config" +
        "?lang=" + encodeURIComponent(state.lang) +
        (requestedChatSession
          ? "&session_id=" + encodeURIComponent(requestedChatSession)
          : "")
    ).catch(() => null);

    const [i18n, config, chatConfig] = await Promise.all([
      i18nPromise,
      configPromise,
      chatConfigPromise,
    ]);
    if (i18n && i18n.ru) state.i18n = i18n;
    state.config = config;
    state.chatSessionId = (chatConfig && chatConfig.session_id) || state.chatSessionId;
    if (state.chatSessionId) {
      try {
        window.localStorage.setItem("agro-chat-session", state.chatSessionId);
        window.sessionStorage.setItem("agro-chat-session", state.chatSessionId);
      } catch (error) {
        console.warn("Chat session storage is unavailable", error);
      }
    }
    state.chatFreeTextEnabled = Boolean(chatConfig && chatConfig.free_text_enabled);
    applyStaticI18n();
    updateAssistantUI();
    saveAppState();
    initMap();
    if (state.map) {
      await restoreSavedDemoField();
      restoreFieldState();
      updateBasemapButtons();
      els.verdictFilter.value = state.verdictFilter || "";
      if (state.batchRows.length) renderPortfolio();
      switchResultView(state.activeResultView);
      renderAssistantMessages();
    }

    updateVerifyState();
    if (new URLSearchParams(window.location.search).get("demo") === "1") {
      toggleDemoPanel(true);
    }

    const initialTab =
      window.location.hash === "#portfolio"
        ? "portfolio"
        : state.activeTab;
    switchTab(initialTab, { history: false });
    updateEffect();
    saveAppState();
  }

  async function loadDemos() {
    if (state.demoLoadPromise) return state.demoLoadPromise;
    const lang = state.lang;
    state.demoLoadPromise = fetchJSON("/api/demo-fields?lang=" + lang)
      .then((payload) => {
        state.demoFields = Array.isArray(payload.fields) ? payload.fields : [];
        state.demoMeta = payload.meta || null;
        state.demoLang = lang;
        if (!els.demoPanel.classList.contains("hidden")) renderDemoList();
        return payload;
      })
      .finally(() => {
        state.demoLoadPromise = null;
      });
    return state.demoLoadPromise;
  }

  async function restoreSavedDemoField() {
    if (!state.selectedDemoId) return;
    try {
      await loadDemos();
    } catch (error) {
      console.error("Saved demo field could not be restored", error);
    }
  }

  /* ======================================================================
     Events
     ====================================================================== */

  function bindEvents() {
    els.tabButtons.forEach((button) => {
      button.addEventListener("click", () => switchTab(button.dataset.tab));
      button.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const index = els.tabButtons.indexOf(button);
        const direction = event.key === "ArrowRight" ? 1 : -1;
        const next = els.tabButtons[(index + direction + els.tabButtons.length) % els.tabButtons.length];
        switchTab(next.dataset.tab);
        next.focus();
      });
    });

    els.langButtons.forEach((button) => {
      button.addEventListener("click", () => setLanguage(button.dataset.lang));
    });

    els.drawPolygon.addEventListener("click", () => startDrawing("polygon"));
    els.drawRectangle.addEventListener("click", () => startDrawing("rectangle"));
    els.clearField.addEventListener("click", clearField);
    els.demoToggle.addEventListener("click", () => toggleDemoPanel());
    els.demoClose.addEventListener("click", () => toggleDemoPanel(false));
    els.demoList.addEventListener("click", (event) => {
      const button = event.target.closest("[data-demo-id]");
      if (!button) return;
      selectDemo(button.dataset.demoId, { zoom: true });
    });

    els.verifyForm.addEventListener("submit", (event) => {
      event.preventDefault();
      verify({ scroll: true });
    });
    els.retryButton.addEventListener("click", () => verify({ scroll: false }));
    els.resultViewButtons.forEach((button) => {
      button.addEventListener("click", () => switchResultView(button.dataset.resultView));
    });
    els.temporalSlider.addEventListener("input", () => {
      state.temporalIndex = Number(els.temporalSlider.value);
      updateTemporalSelection();
      saveAppState();
    });
    els.temporalPlay.addEventListener("click", toggleTemporalPlayback);
    els.temporalModeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        setTemporalMode(button.dataset.temporalMode);
        saveAppState();
      });
    });
    els.assistantQuestions.forEach((button) => {
      button.addEventListener("click", () =>
        sendAssistantQuestion(button.dataset.chatQuestion, t("chat.question." + button.dataset.chatQuestion))
      );
    });
    els.assistantForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const question = els.assistantInput.value.trim();
      if (question && state.chatFreeTextEnabled) sendAssistantQuestion(null, question);
    });
    els.assistantInput.addEventListener("input", updateAssistantUI);
    els.assistantInput.addEventListener("input", saveAppStateSoon);
    els.assistantMessages.addEventListener("click", onAssistantAction);

    els.undoPoint.addEventListener("click", undoPoint);
    els.finishDraw.addEventListener("click", finishDrawing);
    els.cancelDraw.addEventListener("click", () => cancelDrawing());
    els.sheetUndo.addEventListener("click", undoPoint);
    els.sheetFinish.addEventListener("click", finishDrawing);
    els.sheetCancel.addEventListener("click", () => cancelDrawing());
    els.drawSheetHandle.addEventListener("pointerdown", startDrawSheetDrag);
    els.drawSheetHandle.addEventListener("pointermove", moveDrawSheet);
    els.drawSheetHandle.addEventListener("pointerup", endDrawSheetDrag);
    els.drawSheetHandle.addEventListener("pointercancel", endDrawSheetDrag);
    els.drawSheetHandle.addEventListener("click", (event) => event.stopPropagation());
    els.drawSheetHandle.addEventListener("keydown", onDrawSheetKeydown);

    els.cropSelect.addEventListener("change", updateVerifyState);
    els.sowingDate.addEventListener("change", updateVerifyState);
    [els.cropSelect, els.sowingDate].forEach((input) => {
      input.addEventListener("change", saveAppState);
    });
    els.pdfButton.addEventListener("click", printReport);

    els.basemapButtons.forEach((button) => {
      button.addEventListener("click", () => setBasemap(button.dataset.basemap));
    });

    els.searchInput.addEventListener("input", onSearchInput);
    els.searchInput.addEventListener("input", saveAppStateSoon);
    els.searchInput.addEventListener("keydown", onSearchKeydown);
    els.searchResults.addEventListener("click", (event) => {
      const button = event.target.closest("[data-result-index]");
      if (!button) return;
      const result = state.searchResults[Number(button.dataset.resultIndex)];
      if (result) chooseSearchResult(result);
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".search-box")) hideSearchResults();
    });

    document.addEventListener("keydown", onGlobalKeydown);

    els.loadSample.addEventListener("click", () => loadSample());
    els.csvInput.addEventListener("change", onCsvChange);
    els.exportCsv.addEventListener("click", exportPortfolioCsv);
    els.verdictFilter.addEventListener("change", () => {
      renderPortfolio();
      saveAppState();
    });
    els.portfolioTable.querySelector("thead").addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      const header = button.closest("th[data-sort]");
      if (header) onSort(header.dataset.sort);
    });
    els.portfolioBody.addEventListener("click", (event) => {
      const row = event.target.closest("tr[data-row-id]");
      if (!row) return;
      openPortfolioRow(row.dataset.rowId);
    });
    els.portfolioBody.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const row = event.target.closest("tr[data-row-id]");
      if (!row) return;
      event.preventDefault();
      openPortfolioRow(row.dataset.rowId);
    });

    [els.effectVisit, els.effectRemote, els.effectFields].forEach((input) => {
      input.addEventListener("input", () => {
        updateEffect();
        saveAppStateSoon();
      });
    });

    window.addEventListener("resize", debounce(() => {
      if (state.map) state.map.invalidateSize();
      if (state.chart) state.chart.resize();
    }, 150));
    window.addEventListener("pagehide", saveAppState);
    window.addEventListener("beforeunload", saveAppState);
  }

  function debounce(callback, delay) {
    let timer = null;
    return function debounced(...args) {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => callback.apply(null, args), delay);
    };
  }

  function restoreAppState() {
    let saved;
    try {
      saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null");
    } catch (error) {
      console.warn("Saved app state could not be read", error);
      return;
    }
    if (!saved || typeof saved !== "object" || saved.version !== 1) return;

    if (saved.lang === "ru" || saved.lang === "kz") state.lang = saved.lang;
    if (saved.basemap === "satellite" || saved.basemap === "scheme") {
      state.basemap = saved.basemap;
    }
    if (saved.activeTab === "check" || saved.activeTab === "portfolio") {
      state.activeTab = saved.activeTab;
    }
    if (saved.activeResultView === "result" || saved.activeResultView === "assistant") {
      state.activeResultView = saved.activeResultView;
    }
    if (typeof saved.selectedDemoId === "string") {
      state.selectedDemoId = saved.selectedDemoId;
    }
    if (
      saved.geojson &&
      saved.geojson.type === "Polygon" &&
      Array.isArray(saved.geojson.coordinates)
    ) {
      state.geojson = saved.geojson;
    }
    if (saved.mode === "polygon" || saved.mode === "rectangle") {
      state.mode = saved.mode;
    }
    if (Array.isArray(saved.points)) {
      state.points = saved.points
        .filter(
          (point) =>
            Array.isArray(point) &&
            point.length === 2 &&
            point.every(Number.isFinite) &&
            Math.abs(point[0]) <= 90 &&
            Math.abs(point[1]) <= 180
        )
        .map(([lat, lng]) => L.latLng(lat, lng));
    }
    if (
      Array.isArray(saved.rectAnchor) &&
      saved.rectAnchor.length === 2 &&
      saved.rectAnchor.every(Number.isFinite) &&
      Math.abs(saved.rectAnchor[0]) <= 90 &&
      Math.abs(saved.rectAnchor[1]) <= 180
    ) {
      state.rectAnchor = L.latLng(saved.rectAnchor[0], saved.rectAnchor[1]);
    }
    if (CROPS.includes(saved.crop)) els.cropSelect.value = saved.crop;
    if (
      typeof saved.sowingDate === "string" &&
      /^\d{4}-\d{2}-\d{2}$/.test(saved.sowingDate)
    ) {
      els.sowingDate.value = saved.sowingDate;
    }
    if (
      saved.lastResult &&
      typeof saved.lastResult === "object" &&
      saved.lastResult.field &&
      Array.isArray(saved.lastResult.series)
    ) {
      state.lastResult = saved.lastResult;
    }
    if (saved.lastResultParams && typeof saved.lastResultParams === "object") {
      state.lastResultParams = saved.lastResultParams;
    }
    if (Array.isArray(saved.chatMessages)) {
      state.chatMessages = saved.chatMessages
        .filter(
          (message) =>
            message &&
            (message.role === "user" || message.role === "assistant") &&
            typeof message.text === "string"
        )
        .slice(-30);
    }
    if (typeof saved.chatSessionId === "string") {
      state.chatSessionId = saved.chatSessionId;
    }
    if (typeof saved.assistantDraft === "string") {
      els.assistantInput.value = saved.assistantDraft.slice(0, 500);
    }
    if (typeof saved.searchDraft === "string") {
      els.searchInput.value = saved.searchDraft.slice(0, 200);
    }
    if (Array.isArray(saved.batchRows)) state.batchRows = saved.batchRows;
    if (saved.batchMeta && typeof saved.batchMeta === "object") {
      state.batchMeta = saved.batchMeta;
    }
    state.portfolioLoaded = state.batchRows.length > 0;
    if (typeof saved.lastBatchAt === "string" && Number.isFinite(Date.parse(saved.lastBatchAt))) {
      state.lastBatchAt = new Date(saved.lastBatchAt);
    }
    if (["id", "verdict", "score", "crop"].includes(saved.sortKey)) {
      state.sortKey = saved.sortKey;
    }
    if (saved.sortDir === "asc" || saved.sortDir === "desc") {
      state.sortDir = saved.sortDir;
    }
    if (typeof saved.verdictFilter === "string") {
      state.verdictFilter = saved.verdictFilter;
    }
    if (saved.mapView && Array.isArray(saved.mapView.center)) {
      const [lat, lng] = saved.mapView.center;
      const zoom = Number(saved.mapView.zoom);
      if (
        Number.isFinite(lat) &&
        Number.isFinite(lng) &&
        Math.abs(lat) <= 90 &&
        Math.abs(lng) <= 180 &&
        Number.isFinite(zoom) &&
        zoom >= 3 &&
        zoom <= 19
      ) {
        state.restoredMapView = { center: [lat, lng], zoom };
      }
    }
    if (
      saved.temporal &&
      (saved.temporal.mode === "ndvi" || saved.temporal.mode === "deviation") &&
      Number.isInteger(saved.temporal.index)
    ) {
      state.restoredTemporal = saved.temporal;
    }
    if (saved.effect && typeof saved.effect === "object") {
      ["effectVisit", "effectRemote", "effectFields"].forEach((key) => {
        const value = saved.effect[key];
        if (typeof value === "string" && /^\d{1,12}$/.test(value)) {
          els[key].value = value;
        }
      });
    }
    if (
      saved.lastResult &&
      (!state.selectedDemoId && !state.geojson ||
        !state.lastResultParams ||
        state.lastResultParams.crop !== els.cropSelect.value ||
        state.lastResultParams.sowing_date !== els.sowingDate.value)
    ) {
      state.lastResult = null;
      state.lastResultParams = null;
      state.chatMessages = [];
    }
  }

  function saveAppState() {
    const center = state.map && state.map.getCenter();
    const snapshot = {
      version: 1,
      lang: state.lang,
      basemap: state.basemap,
      activeTab: state.activeTab,
      activeResultView: state.activeResultView,
      selectedDemoId: state.selectedDemoId,
      geojson: state.geojson,
      mode: state.mode,
      points: state.points.map((point) => [point.lat, point.lng]),
      rectAnchor: state.rectAnchor
        ? [state.rectAnchor.lat, state.rectAnchor.lng]
        : null,
      crop: els.cropSelect.value,
      sowingDate: els.sowingDate.value,
      lastResult: state.lastResult,
      lastResultParams: state.lastResultParams,
      temporal: {
        mode: state.temporalMode,
        index: state.temporalIndex,
      },
      chatMessages: state.chatMessages.slice(-30),
      chatSessionId: state.chatSessionId,
      assistantDraft: els.assistantInput.value.slice(0, 500),
      searchDraft: els.searchInput.value.slice(0, 200),
      batchRows: state.batchRows,
      batchMeta: state.batchMeta,
      lastBatchAt: state.lastBatchAt ? state.lastBatchAt.toISOString() : null,
      sortKey: state.sortKey,
      sortDir: state.sortDir,
      verdictFilter: els.verdictFilter.value,
      mapView: center
        ? { center: [center.lat, center.lng], zoom: state.map.getZoom() }
        : state.restoredMapView,
      effect: {
        effectVisit: els.effectVisit.value,
        effectRemote: els.effectRemote.value,
        effectFields: els.effectFields.value,
      },
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
    } catch (error) {
      console.warn("App state could not be saved in this browser", error);
    }
  }

  function restoreFieldState() {
    if (state.selectedDemoId) {
      const field = state.demoFields.find((item) => item.id === state.selectedDemoId);
      if (field) {
        state.selectedField = field;
        drawDemoLayers();
        updateFieldCardDemo(field);
      } else {
        state.selectedDemoId = null;
      }
    }
    if (state.geojson && !state.selectedDemoId) {
      const ring = state.geojson.coordinates[0];
      if (Array.isArray(ring) && ring.length >= 4) {
        const points = ring.slice(0, -1).map(([lng, lat]) => L.latLng(lat, lng));
        if (points.every((point) => Number.isFinite(point.lat) && Number.isFinite(point.lng))) {
          state.drawnAreaHa = ringAreaHa(points);
          state.drawnVertices = points.length;
          state.finalLayer = L.polygon(points, {
            color: "#ffffff",
            weight: 2.2,
            opacity: 0.95,
            fillColor: "#ffffff",
            fillOpacity: 0.12,
            interactive: false,
            className: "drawn-polygon",
          }).addTo(state.map);
          updateDrawnFieldCard();
        } else {
          state.geojson = null;
        }
      } else {
        state.geojson = null;
      }
    }
    if (state.mode) {
      const mode = state.mode;
      const points = state.points;
      const anchor = state.rectAnchor;
      state.mode = null;
      state.points = [];
      state.rectAnchor = null;
      startDrawing(mode);
      state.points = points;
      state.rectAnchor = anchor;
      if (mode === "rectangle" && anchor) {
        state.rectAnchorMarker = L.marker(anchor, {
          icon: L.divIcon({
            className: "vertex-icon first",
            html: "<span></span>",
            iconSize: [22, 22],
            iconAnchor: [11, 11],
          }),
          interactive: false,
          keyboard: false,
        }).addTo(state.map);
      }
      renderVertexMarkers();
      refreshShapeLayers();
      updateDrawingUI();
      saveAppState();
    }
    if (state.lastResult && !state.mode && (state.selectedField || state.geojson)) {
      renderResult(state.lastResult);
      updateDemoStyles();
      if (state.restoredTemporal) {
        setTemporalMode(state.restoredTemporal.mode);
        const maxIndex = Math.max(0, state.lastResult.series.length - 1);
        state.temporalIndex = clamp(state.restoredTemporal.index, 0, maxIndex);
        els.temporalSlider.value = String(state.temporalIndex);
        updateTemporalSelection();
      }
      if (state.activeResultView === "assistant") renderAssistantMessages();
    } else {
      state.lastResult = null;
      state.lastResultParams = null;
      state.chatMessages = [];
    }
    updateVerifyState();
  }

  async function setLanguage(lang) {
    if (lang !== "ru" && lang !== "kz") return;
    if (lang === state.lang) return;
    state.lang = lang;
    try {
      const sessionParam = state.chatSessionId
        ? "&session_id=" + encodeURIComponent(state.chatSessionId)
        : "";
      const chatConfig = await fetchJSON(
        "/api/chat/config?lang=" + encodeURIComponent(state.lang) + sessionParam
      );
      state.chatFreeTextEnabled = Boolean(chatConfig.free_text_enabled);
    } catch (error) {
      console.error("Chat configuration unavailable for selected language", error);
    }
    if (state.selectedDemoId || !els.demoPanel.classList.contains("hidden")) {
      try {
        await loadDemos();
        if (state.demoLang !== state.lang) await loadDemos();
      } catch (error) {
        console.error(error);
      }
    }
    if (state.selectedDemoId) {
      state.selectedField =
        state.demoFields.find((field) => field.id === state.selectedDemoId) || null;
    }
    state.chatMessages = [];
    switchResultView("result");
    applyStaticI18n();
    if (!els.demoPanel.classList.contains("hidden")) renderDemoList();
    updateDemoStyles();
    renderAssistantMessages();
    updateAssistantUI();
    if (state.mode) updateDrawingUI();
    if (state.lastResult && !state.mode) {
      verify({ scroll: false, silent: true });
    }
    if (state.batchRows.length || state.batchFile) reloadBatchForLanguage();
    updateEffect();
    saveAppState();
  }

  /* ======================================================================
     Tabs
     ====================================================================== */

  function switchTab(name, options = {}) {
    if (name !== "check" && name !== "portfolio") return;
    state.activeTab = name;
    els.tabButtons.forEach((button) => {
      const active = button.dataset.tab === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
    });
    Object.keys(els.panels).forEach((key) => {
      const panel = els.panels[key];
      const active = key === name;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
    if (options.history !== false) {
      const hash = name === "portfolio" ? "#portfolio" : "#check";
      if (window.location.hash !== hash) {
        window.history.replaceState(null, "", hash);
      }
    }
    if (name === "check") {
      window.setTimeout(() => {
        if (state.map) state.map.invalidateSize();
        if (state.chart) state.chart.resize();
      }, 60);
    } else if (!state.portfolioLoaded && !state.batchLoading) {
      loadSample({ silent: true });
    }
    saveAppState();
  }

  /* ======================================================================
     Map
     ====================================================================== */

  function initMap() {
    if (typeof L === "undefined") {
      showToast(t("map.offline"));
      return;
    }
    if (state.map) return;

    const map = L.map("map", {
      zoomControl: false,
      minZoom: 3,
      maxZoom: 19,
      attributionControl: true,
      preferCanvas: false,
    });
    map.attributionControl.setPrefix(false);

    state.satelliteLayer = L.tileLayer(SATELLITE_URL, {
      maxZoom: 19,
      maxNativeZoom: 18,
      attribution: SATELLITE_ATTRIBUTION,
      crossOrigin: true,
    });
    state.schemeLayer = L.tileLayer(SCHEME_URL, {
      maxZoom: 19,
      maxNativeZoom: 19,
      attribution: SCHEME_ATTRIBUTION,
      crossOrigin: true,
    });
    if (state.basemap === "scheme") {
      state.schemeLayer.addTo(map);
      map.getContainer().classList.add("scheme");
    } else {
      state.satelliteLayer.addTo(map);
    }
    L.control.zoom({ position: "topright" }).addTo(map);
    if (state.restoredMapView) {
      map.setView(state.restoredMapView.center, state.restoredMapView.zoom, {
        animate: false,
      });
    } else {
      map.fitBounds(
        [
          [50.8, 64.5],
          [56.2, 74.5],
        ],
        { padding: [20, 20], animate: false }
      );
    }

    map.on("click", onMapClick);
    map.on("dblclick", onMapDoubleClick);
    map.on("mousemove", onMapMouseMove);
    map.on("tileerror", onTileError);
    map.on("moveend zoomend", saveAppState);
    state.map = map;
  }

  function onMapDoubleClick(event) {
    if (!state.mode) return;
    if (state.mode !== "polygon") return;
    if (event && event.originalEvent) event.originalEvent.preventDefault();
    if (state.points.length >= 3) {
      finishDrawing();
    } else {
      showMapNotice(t("map.minVertices"));
    }
  }

  function onTileError() {
    if (state.tileErrorShown) return;
    state.tileErrorShown = true;
    showToast(t("map.offline"));
  }

  function setBasemap(name) {
    if (name !== "satellite" && name !== "scheme") return;
    if (!state.map) return;
    if (name === state.basemap) {
      updateBasemapButtons();
      return;
    }
    if (name === "scheme") {
      if (state.map.hasLayer(state.satelliteLayer)) state.map.removeLayer(state.satelliteLayer);
      state.schemeLayer.addTo(state.map);
    } else {
      if (state.map.hasLayer(state.schemeLayer)) state.map.removeLayer(state.schemeLayer);
      state.satelliteLayer.addTo(state.map);
    }
    state.basemap = name;
    state.map.getContainer().classList.toggle("scheme", name === "scheme");
    updateBasemapButtons();
    saveAppState();
  }

  function updateBasemapButtons() {
    els.basemapButtons.forEach((button) => {
      const active = button.dataset.basemap === state.basemap;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  function drawDemoLayers() {
    if (!state.map || typeof L === "undefined") return;
    state.demoLayers.forEach((layer) => {
      if (state.map.hasLayer(layer)) state.map.removeLayer(layer);
    });
    state.demoLayers.clear();

    state.demoFields
      .filter((field) => field.id === state.selectedDemoId)
      .forEach((field) => {
        if (!Array.isArray(field.polygon) || field.polygon.length < 3) return;
        const index = state.demoFields.indexOf(field);
        const latlngs = field.polygon.map((coordinate) => [coordinate[1], coordinate[0]]);
        const layer = L.polygon(latlngs, demoStyle(false, index));
        layer.bindTooltip("№" + (index + 1), {
          permanent: true,
          direction: "center",
          className: "demo-label",
          opacity: 1,
          interactive: false,
        });
        layer.on("click", () => {
          if (state.mode) return;
          selectDemo(field.id, { zoom: true });
        });
        layer.addTo(state.map);
        state.demoLayers.set(field.id, layer);
      });
    updateDemoStyles();
  }

  function demoStyle(selected) {
    return {
      className: "demo-polygon",
      color: "#ffffff",
      weight: selected ? 2.5 : 1.6,
      opacity: 0.95,
      dashArray: selected ? null : "5 5",
      fill: true,
      fillColor: "#ffffff",
      fillOpacity: selected ? 0.12 : 0.04,
      interactive: true,
    };
  }

  function updateDemoStyles() {
    state.demoFields.forEach((field) => {
      const layer = state.demoLayers.get(field.id);
      if (!layer) return;
      const selected = field.id === state.selectedDemoId;
      layer.setStyle(demoStyle(selected));
    });
    if (state.lastResult && state.lastResult.series) {
      const point = state.lastResult.series[state.temporalIndex];
      if (point) applyTemporalMapColor(point);
    }
  }

  function zoomToField(field) {
    if (!state.map || typeof L === "undefined" || !field) return;
    if (!Array.isArray(field.polygon) || field.polygon.length < 3) return;
    const bounds = L.latLngBounds(
      field.polygon.map((coordinate) => [coordinate[1], coordinate[0]])
    );
    if (bounds.isValid()) {
      state.map.fitBounds(bounds, { padding: [55, 55], maxZoom: 14, animate: false });
    }
  }

  function onMapClick(event) {
    if (!state.mode) return;
    if (state.mode === "polygon") {
      addPolygonPoint(event.latlng);
    } else if (state.mode === "rectangle") {
      handleRectangleClick(event.latlng);
    }
  }

  function onMapMouseMove(event) {
    if (!state.mode) return;
    if (state.mode === "polygon") {
      if (COARSE_POINTER) {
        clearCursorLine();
      } else {
        updateCursorLine(event.latlng);
      }
    } else if (state.mode === "rectangle") {
      updateRectanglePreview(event.latlng);
    }
  }

  /* ======================================================================
     Demo selection and field card
     ====================================================================== */

  function selectDemo(id, options = {}) {
    const field = state.demoFields.find((item) => item.id === id);
    if (!field) return;
    const same = state.selectedDemoId === id;
    if (state.mode) cancelDrawing({ quiet: true });
    abortPendingVerify();
    removePortfolioMarker();
    removeFinalLayer();

    state.selectedDemoId = id;
    state.selectedField = field;
    state.geojson = null;
    state.drawnAreaHa = null;
    state.drawnVertices = 0;

    if (!same) {
      els.cropSelect.value = field.crop || "wheat";
      els.sowingDate.value = field.sowing_date || els.sowingDate.value;
    }
    drawDemoLayers();
    updateFieldCardDemo(field);
    closeDemoPanel();
    if (!same) resetResult();
    updateVerifyState();
    if (options.zoom) zoomToField(field);
    if (options.verifyNow) verify({ scroll: Boolean(options.scroll) });
    saveAppState();
  }

  function updateFieldCardDemo(field) {
    if (!field) {
      resetFieldCardEmpty();
      return;
    }
    els.fieldCard.classList.remove("is-empty");
    els.fieldName.textContent = field.name || t("field.custom");
    const pieces = [];
    if (field.district) pieces.push(field.district);
    pieces.push(
      t("field.meta", {
        crop: field.crop_label || field.crop,
        date: fmtDate(field.sowing_date),
        area: fmtNum(field.area_ha, 1),
      })
    );
    if (field.sowing_date_conditional) pieces.push(t("field.conditionalDate"));
    els.fieldMeta.textContent = pieces.join(" · ");
    els.fieldMeta.title = pieces.join(" · ");
    const note = field.note || t("field.conditional");
    els.fieldHonesty.textContent = note;
    els.fieldHonesty.classList.remove("region-warning");
    els.fieldHonesty.classList.remove("hidden");
    setFieldSource(field.source);
    hideFieldError();
  }

  function updateDrawnFieldCard() {
    if (!state.geojson) {
      resetFieldCardEmpty();
      return;
    }
    els.fieldCard.classList.remove("is-empty");
    els.fieldName.textContent = t("field.custom");
    els.fieldMeta.textContent = t("map.area", {
      area: fmtNum(state.drawnAreaHa, 1),
      n: state.drawnVertices,
    });
    els.fieldMeta.title = els.fieldMeta.textContent;
    els.fieldHonesty.textContent = t("field.drawnNote");
    els.fieldHonesty.classList.remove("region-warning");
    els.fieldHonesty.classList.remove("hidden");
    if (!state.lastResult) setFieldSource(null);
    hideFieldError();
  }

  function updateDrawingCard() {
    els.fieldCard.classList.remove("is-empty");
    els.fieldName.textContent =
      state.mode === "rectangle" ? t("action.rect") : t("map.drawingTitle");
    els.fieldHonesty.textContent = t("field.drawnNote");
    els.fieldHonesty.classList.remove("region-warning");
    els.fieldHonesty.classList.remove("hidden");
    hideFieldError();
    setFieldSource(null);
    updateDrawingUI();
  }

  function resetFieldCardEmpty() {
    els.fieldCard.classList.add("is-empty");
    els.fieldName.textContent = t("field.none");
    els.fieldMeta.textContent = t("field.noneHint");
    els.fieldMeta.title = "";
    els.fieldHonesty.classList.add("hidden");
    els.fieldHonesty.classList.remove("region-warning");
    els.fieldHonesty.textContent = "";
    setFieldSource(null);
    hideFieldError();
  }

  function setFieldSource(source) {
    if (!source) {
      els.fieldSource.classList.add("hidden");
      els.fieldSource.className = "source-chip hidden";
      els.fieldSource.textContent = "";
      return;
    }
    const real = source === "sentinel-2-l2a";
    els.fieldSource.className = "source-chip " + (real ? "real" : "synthetic");
    els.fieldSource.textContent = real ? t("field.real") : t("field.synthetic");
    els.fieldSource.classList.remove("hidden");
  }

  function showFieldError(message) {
    if (!els.fieldError) return;
    els.fieldError.textContent = message;
    els.fieldError.classList.remove("hidden");
  }

  function hideFieldError() {
    if (!els.fieldError) return;
    els.fieldError.classList.add("hidden");
    els.fieldError.textContent = "";
  }

  function toggleDemoPanel(force) {
    const open = typeof force === "boolean" ? force : els.demoPanel.classList.contains("hidden");
    els.demoPanel.classList.toggle("hidden", !open);
    els.demoToggle.setAttribute("aria-expanded", open ? "true" : "false");
    els.demoToggle.classList.toggle("active", open);
    if (open) {
      if (state.demoLang === state.lang) {
        renderDemoList();
      } else {
        els.demoList.textContent = t("demo.loading");
        loadDemos().catch((error) => showToast(error.message || t("result.error")));
      }
    }
    saveAppState();
  }

  function closeDemoPanel() {
    els.demoPanel.classList.add("hidden");
    els.demoToggle.setAttribute("aria-expanded", "false");
    els.demoToggle.classList.remove("active");
  }

  function renderDemoList() {
    if (!els.demoList) return;
    if (!state.demoFields.length) {
      els.demoList.innerHTML =
        '<p class="search-empty">' + esc(t("field.noneHint")) + "</p>";
      return;
    }
    els.demoList.innerHTML = state.demoFields
      .map((field) => {
        const selected = field.id === state.selectedDemoId;
        const real = field.source === "sentinel-2-l2a";
        return (
          '<button type="button" class="demo-item' +
          (selected ? " selected" : "") +
          '" data-demo-id="' +
          esc(field.id) +
          '" aria-pressed="' +
          (selected ? "true" : "false") +
          '">' +
          '<span class="demo-item-main"><strong>' +
          esc(field.name) +
          "</strong><span>" +
          esc(field.crop_label || field.crop) +
          " · " +
          esc(t("demo.scenario", { scenario: field.scenario_label || field.expected_verdict || "" })) +
          "</span></span>" +
          '<span class="source-chip ' +
          (real ? "real" : "synthetic") +
          '">' +
          esc(real ? t("field.real") : t("field.synthetic")) +
          "</span></button>"
        );
      })
      .join("");
  }

  function clearField() {
    if (state.mode) cancelDrawing({ quiet: true });
    abortPendingVerify();
    removePortfolioMarker();
    removeFinalLayer();
    state.selectedDemoId = null;
    state.selectedField = null;
    state.geojson = null;
    state.drawnAreaHa = null;
    state.drawnVertices = 0;
    drawDemoLayers();
    updateDemoStyles();
    resetFieldCardEmpty();
    resetResult();
    updateVerifyState();
    els.cropSelect.value = "wheat";
    els.sowingDate.value = "2026-05-15";
    hideMapNotice();
    saveAppState();
  }

  /* ======================================================================
     Drawing
     ====================================================================== */

  function startDrawing(mode) {
    if (mode !== "polygon" && mode !== "rectangle") return;
    if (!state.map || typeof L === "undefined") {
      showToast(t("map.offline"));
      return;
    }
    if (state.mode === mode) return;
    if (state.mode) cancelDrawing({ quiet: true });
    abortPendingVerify();
    removePortfolioMarker();
    removeFinalLayer();
    closeDemoPanel();
    resetResult();

    state.selectedDemoId = null;
    state.selectedField = null;
    state.geojson = null;
    state.drawnAreaHa = null;
    state.drawnVertices = 0;
    state.mode = mode;
    state.drawSheetOffset = 0;
    state.points = [];
    state.rectAnchor = null;
    updateDemoStyles();

    state.drawGroup = L.layerGroup().addTo(state.map);
    state.markerGroup = L.layerGroup().addTo(state.map);
    document.body.classList.add("is-drawing");
    state.map.getContainer().classList.add("drawing");
    if (state.map.doubleClickZoom) state.map.doubleClickZoom.disable();

    els.drawPolygon.classList.toggle("active", mode === "polygon");
    els.drawRectangle.classList.toggle("active", mode === "rectangle");
    els.mapHint.textContent = t(mode === "polygon" ? "map.drawHint" : "map.rectHint");
    setHidden(els.mapHint, false);
    if (state.drawHintTimer) window.clearTimeout(state.drawHintTimer);
    state.drawHintTimer = window.setTimeout(() => {
      state.drawHintTimer = null;
      if (state.mode) setHidden(els.mapHint, true);
    }, 2500);
    hideMapNotice();
    setHidden(els.drawToolbar, false);
    setHidden(els.drawSheet, false);
    updateDrawSheetPosition();
    setDrawingInputsDisabled(true);
    updateDrawingCard();
    updateVerifyState();
    window.setTimeout(() => {
      if (state.map) state.map.invalidateSize();
    }, 40);
    saveAppState();
  }

  function endDrawingUi() {
    if (state.drawGroup && state.map) state.map.removeLayer(state.drawGroup);
    if (state.markerGroup && state.map) state.map.removeLayer(state.markerGroup);
    if (state.cursorLine && state.map) state.map.removeLayer(state.cursorLine);
    if (state.rectPreview && state.map) state.map.removeLayer(state.rectPreview);
    if (state.rectAnchorMarker && state.map) state.map.removeLayer(state.rectAnchorMarker);
    state.drawGroup = null;
    state.markerGroup = null;
    state.cursorLine = null;
    state.rectPreview = null;
    state.rectAnchorMarker = null;
    state.markers = [];
    state.shapeLayers = [];
    state.points = [];
    state.rectAnchor = null;
    state.mode = null;
    state.drawSheetOffset = 0;
    document.body.style.removeProperty("--draw-sheet-height");
    document.body.style.removeProperty("--draw-sheet-offset");
    els.drawSheet.style.transform = "";
    els.drawSheetHandle.setAttribute("aria-valuenow", "0");
    els.drawSheetHandle.setAttribute("aria-valuetext", t("map.drawPanelBottom"));

    if (state.drawHintTimer) {
      window.clearTimeout(state.drawHintTimer);
      state.drawHintTimer = null;
    }
    document.body.classList.remove("is-drawing");
    if (state.map) {
      state.map.getContainer().classList.remove("drawing");
      if (state.map.doubleClickZoom) state.map.doubleClickZoom.enable();
    }
    setHidden(els.mapHint, true);
    hideMapNotice();
    setHidden(els.drawToolbar, true);
    setHidden(els.drawSheet, true);
    els.drawPolygon.classList.remove("active");
    els.drawRectangle.classList.remove("active");
    setDrawingInputsDisabled(false);
  }

  function cancelDrawing(options = {}) {
    if (!state.mode) return;
    endDrawingUi();
    removeFinalLayer();
    state.geojson = null;
    state.drawnAreaHa = null;
    state.drawnVertices = 0;
    state.selectedDemoId = null;
    state.selectedField = null;
    updateDemoStyles();
    resetFieldCardEmpty();
    resetResult();
    updateVerifyState();
    saveAppState();
    if (!options.quiet) showToast(t("action.cancel"));
  }

  function setDrawingInputsDisabled(disabled) {
    els.cropSelect.disabled = disabled;
    els.sowingDate.disabled = disabled;
  }

  function addPolygonPoint(latlng) {
    hideMapNotice();
    if (!latlng) return;
    const last = state.points[state.points.length - 1];
    if (last && last.distanceTo(latlng) < 2) return;
    const candidate = state.points.concat([latlng]);
    if (candidate.length >= 4 && !polylineIsSimple(candidate)) {
      showMapNotice(t("map.selfIntersect"));
      return;
    }
    state.points.push(latlng);
    clearCursorLine();
    if (state.drawHintTimer) {
      window.clearTimeout(state.drawHintTimer);
      state.drawHintTimer = null;
    }
    setHidden(els.mapHint, true);
    renderVertexMarkers();
    refreshShapeLayers();
    updateDrawingUI();
    saveAppState();
  }

  function drawSheetMaxOffset() {
    const sheetHeight = els.drawSheet.getBoundingClientRect().height;
    return Math.max(0, Math.min(sheetHeight - 64, window.innerHeight * 0.55));
  }

  function updateDrawSheetPosition() {
    if (!els.drawSheet || els.drawSheet.classList.contains("hidden")) return;
    const maxOffset = drawSheetMaxOffset();
    state.drawSheetOffset = Math.max(0, Math.min(state.drawSheetOffset, maxOffset));
    els.drawSheet.style.transform = `translateY(${state.drawSheetOffset}px)`;
    document.body.style.setProperty(
      "--draw-sheet-height",
      `${els.drawSheet.getBoundingClientRect().height}px`
    );
    document.body.style.setProperty("--draw-sheet-offset", `${state.drawSheetOffset}px`);
    const percent = maxOffset ? Math.round((state.drawSheetOffset / maxOffset) * 100) : 0;
    els.drawSheetHandle.setAttribute("aria-valuenow", String(percent));
    els.drawSheetHandle.setAttribute(
      "aria-valuetext",
      percent === 0
        ? t("map.drawPanelBottom")
        : t("map.drawPanelMoved", { percent })
    );
  }

  function startDrawSheetDrag(event) {
    if (!state.mode || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    els.drawSheetHandle.setPointerCapture(event.pointerId);
    els.drawSheetHandle.dataset.dragStartY = String(event.clientY);
    els.drawSheetHandle.dataset.dragStartOffset = String(state.drawSheetOffset);
    els.drawSheet.classList.add("is-moving");
  }

  function moveDrawSheet(event) {
    if (!els.drawSheetHandle.hasPointerCapture(event.pointerId)) return;
    event.stopPropagation();
    const startY = Number(els.drawSheetHandle.dataset.dragStartY);
    const startOffset = Number(els.drawSheetHandle.dataset.dragStartOffset);
    state.drawSheetOffset = startOffset + event.clientY - startY;
    updateDrawSheetPosition();
  }

  function endDrawSheetDrag(event) {
    if (!els.drawSheetHandle.hasPointerCapture(event.pointerId)) return;
    event.stopPropagation();
    els.drawSheetHandle.releasePointerCapture(event.pointerId);
    els.drawSheet.classList.remove("is-moving");
  }

  function onDrawSheetKeydown(event) {
    if (!state.mode) return;
    const maxOffset = drawSheetMaxOffset();
    const step = Math.max(24, Math.round(window.innerHeight * 0.05));
    if (event.key === "ArrowDown" || event.key === "ArrowRight") {
      state.drawSheetOffset += step;
    } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
      state.drawSheetOffset -= step;
    } else if (event.key === "Home") {
      state.drawSheetOffset = 0;
    } else if (event.key === "End") {
      state.drawSheetOffset = maxOffset;
    } else {
      return;
    }
    event.preventDefault();
    updateDrawSheetPosition();
  }

  function undoPoint() {
    if (!state.mode) return;
    hideMapNotice();
    if (state.mode === "rectangle") {
      if (state.rectAnchor) {
        state.rectAnchor = null;
        if (state.rectAnchorMarker && state.map) state.map.removeLayer(state.rectAnchorMarker);
        if (state.rectPreview && state.map) state.map.removeLayer(state.rectPreview);
        state.rectAnchorMarker = null;
        state.rectPreview = null;
        updateDrawingUI();
        saveAppState();
      }
      return;
    }
    if (!state.points.length) return;
    state.points.pop();
    clearCursorLine();
    renderVertexMarkers();
    refreshShapeLayers();
    updateDrawingUI();
    saveAppState();
  }

  function clearCursorLine() {
    if (state.cursorLine && state.map) state.map.removeLayer(state.cursorLine);
    state.cursorLine = null;
  }

  function renderVertexMarkers() {
    if (!state.map || !state.markerGroup || typeof L === "undefined") return;
    state.markerGroup.clearLayers();
    state.markers = [];
    state.points.forEach((point, index) => {
      const marker = L.marker(point, {
        icon: L.divIcon({
          className: "vertex-icon" + (index === 0 ? " first" : ""),
          html: "<span></span>",
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        }),
        draggable: true,
        keyboard: false,
        autoPan: true,
        zIndexOffset: 500,
      });
      marker._vertexIndex = index;
      marker.on("click", (event) => {
        if (event.originalEvent) L.DomEvent.stopPropagation(event.originalEvent);
        if (index === 0) {
          if (state.points.length >= 3) {
            finishDrawing();
          } else {
            showMapNotice(t("map.minVertices"));
          }
        }
      });
      marker.on("drag", (event) => {
        state.points[event.target._vertexIndex] = event.target.getLatLng();
        refreshShapeLayers();
        updateDrawingUI();
      });
      marker.on("dragend", () => {
        if (!polygonIsSimple(state.points)) {
          showMapNotice(t("map.selfIntersect"));
        } else {
          hideMapNotice();
        }
        updateDrawingUI();
        saveAppState();
      });
      marker.addTo(state.markerGroup);
      state.markers.push(marker);
    });
  }

  function refreshShapeLayers() {
    if (!state.drawGroup || typeof L === "undefined") return;
    state.shapeLayers.forEach((layer) => state.drawGroup.removeLayer(layer));
    state.shapeLayers = [];
    const points = state.points;
    if (points.length >= 3) {
      const fill = L.polygon(points, {
        stroke: false,
        fill: true,
        fillColor: "#ffffff",
        fillOpacity: 0.14,
        interactive: false,
      });
      state.drawGroup.addLayer(fill);
      state.shapeLayers.push(fill);
    }
    if (points.length >= 2) {
      const line = L.polyline(points, {
        color: "#ffffff",
        weight: 2.5,
        opacity: 0.96,
        lineCap: "round",
        lineJoin: "round",
        interactive: false,
        className: "drawing-outline",
      });
      state.drawGroup.addLayer(line);
      state.shapeLayers.push(line);
    }
    if (points.length >= 3) {
      const closing = L.polyline([points[points.length - 1], points[0]], {
        color: "#ffffff",
        weight: 1.6,
        opacity: 0.9,
        dashArray: "5 5",
        interactive: false,
        className: "drawing-closing-line",
      });
      state.drawGroup.addLayer(closing);
      state.shapeLayers.push(closing);
    }
    if (state.cursorLine) state.cursorLine.bringToFront();
  }

  function updateCursorLine(latlng) {
    if (COARSE_POINTER || !state.map || !state.points.length) {
      clearCursorLine();
      return;
    }
    if (!state.cursorLine) {
      state.cursorLine = L.polyline([], {
        color: "#ffffff",
        weight: 1.5,
        opacity: 0.9,
        dashArray: "3 6",
        interactive: false,
        className: "drawing-cursor-line",
      }).addTo(state.map);
    }
    state.cursorLine.setLatLngs([state.points[state.points.length - 1], latlng]);
    state.cursorLine.bringToFront();
  }

  function updateRectanglePreview(latlng) {
    if (!state.map || !state.rectAnchor) return;
    const bounds = L.latLngBounds(state.rectAnchor, latlng);
    if (!state.rectPreview) {
      state.rectPreview = L.rectangle(bounds, {
        color: "#ffffff",
        weight: 1.6,
        dashArray: "5 5",
        fillColor: "#ffffff",
        fillOpacity: 0.12,
        interactive: false,
      }).addTo(state.map);
    } else {
      state.rectPreview.setBounds(bounds);
    }
  }

  function handleRectangleClick(latlng) {
    hideMapNotice();
    if (!state.rectAnchor) {
      state.rectAnchor = latlng;
      state.rectAnchorMarker = L.marker(latlng, {
        icon: L.divIcon({
          className: "vertex-icon first",
          html: "<span></span>",
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        }),
        interactive: false,
        keyboard: false,
      }).addTo(state.map);
      updateDrawingUI();
      saveAppState();
      return;
    }
    const anchor = state.rectAnchor;
    const south = Math.min(anchor.lat, latlng.lat);
    const north = Math.max(anchor.lat, latlng.lat);
    const west = Math.min(anchor.lng, latlng.lng);
    const east = Math.max(anchor.lng, latlng.lng);
    const points = [
      L.latLng(south, west),
      L.latLng(south, east),
      L.latLng(north, east),
      L.latLng(north, west),
    ];
    const area = ringAreaHa(points);
    const limits = areaConfig();
    if (area < limits.min) {
      showMapNotice(t("map.areaMin"));
      return;
    }
    if (area > limits.max) {
      showMapNotice(t("map.areaMax"));
      return;
    }
    if (state.rectPreview && state.map) state.map.removeLayer(state.rectPreview);
    if (state.rectAnchorMarker && state.map) state.map.removeLayer(state.rectAnchorMarker);
    state.rectPreview = null;
    state.rectAnchorMarker = null;
    state.rectAnchor = null;
    state.points = points;
    refreshShapeLayers();
    updateDrawingUI();
    finishDrawing();
  }

  function updateDrawingUI() {
    if (!state.mode) return;
    const limits = areaConfig();
    const points = state.points;
    const count = state.mode === "rectangle" ? (state.rectAnchor ? 1 : 0) : points.length;
    const area = points.length >= 3 ? ringAreaHa(points) : 0;
    const simple = polygonIsSimple(points);
    const title = t(state.mode === "rectangle" ? "action.rect" : "map.drawingTitle");
    if (els.fieldName) els.fieldName.textContent = title;
    if (els.drawSheetTitle) els.drawSheetTitle.textContent = title;

    const areaLabel = area
      ? t("map.area", { area: fmtNum(area, 1), n: points.length })
      : t("map.vertexCount", { n: count });
    els.drawCount.textContent = area
      ? areaLabel
      : t("map.points", { n: count });
    els.drawSheetCount.textContent = areaLabel;
    if (els.fieldMeta) {
      els.fieldMeta.textContent =
        t("map.vertexCount", { n: count }) +
        " · " +
        t("map.drawRange", {
          min: fmtNum(limits.min, 0),
          max: fmtNum(limits.max, 0),
        });
    }

    const undoDisabled =
      state.mode === "rectangle" ? !state.rectAnchor : points.length === 0;
    const finishDisabled =
      state.mode !== "polygon" || points.length < 3 || !simple;
    [els.undoPoint, els.sheetUndo].forEach((button) => {
      if (button) button.disabled = undoDisabled;
    });
    [els.finishDraw, els.sheetFinish].forEach((button) => {
      if (button) button.disabled = finishDisabled;
    });
  }

  function finishDrawing() {
    if (!state.mode) return;
    const points = state.points.slice();
    if (state.mode === "rectangle" && !points.length) {
      showMapNotice(t("map.rectHint"));
      return;
    }
    if (points.length < 3) {
      showMapNotice(t("map.minVertices"));
      return;
    }
    if (!polygonIsSimple(points)) {
      showMapNotice(t("map.selfIntersect"));
      return;
    }
    const area = ringAreaHa(points);
    const limits = areaConfig();
    if (area < limits.min) {
      showMapNotice(t("map.areaMin"));
      return;
    }
    if (area > limits.max) {
      showMapNotice(t("map.areaMax"));
      return;
    }

    const geojson = geojsonFromPoints(points);
    const vertices = points.length;
    const bounds =
      typeof L !== "undefined" && L.latLngBounds ? L.latLngBounds(points) : null;
    endDrawingUi();
    removeFinalLayer();
    state.geojson = geojson;
    state.drawnAreaHa = area;
    state.drawnVertices = vertices;
    state.selectedDemoId = null;
    state.selectedField = null;
    if (state.map && typeof L !== "undefined") {
      state.finalLayer = L.polygon(points, {
        color: "#ffffff",
        weight: 2.2,
        opacity: 0.95,
        fillColor: "#ffffff",
        fillOpacity: 0.12,
        interactive: false,
        className: "drawn-polygon",
      }).addTo(state.map);
      if (bounds && bounds.isValid()) {
        state.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
      }
    }
    updateDemoStyles();
    updateDrawnFieldCard();
    resetResult();
    updateVerifyState();
    saveAppState();
  }

  function removeFinalLayer() {
    if (state.finalLayer && state.map) {
      state.map.removeLayer(state.finalLayer);
    }
    state.finalLayer = null;
  }

  function showMapNotice(message) {
    if (!els.mapNotice) return;
    els.mapNotice.textContent = message;
    els.mapNotice.classList.remove("hidden");
    if (state.noticeTimer) window.clearTimeout(state.noticeTimer);
    state.noticeTimer = window.setTimeout(() => {
      els.mapNotice.classList.add("hidden");
    }, 6000);
  }

  function hideMapNotice() {
    if (!els.mapNotice) return;
    els.mapNotice.classList.add("hidden");
    if (state.noticeTimer) {
      window.clearTimeout(state.noticeTimer);
      state.noticeTimer = null;
    }
  }

  function onGlobalKeydown(event) {
    const target = event.target;
    const editing =
      target &&
      (target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable);

    if (state.mode) {
      if (event.key === "Escape") {
        event.preventDefault();
        cancelDrawing();
        return;
      }
      if (editing) return;
      if (event.key === "Enter") {
        event.preventDefault();
        finishDrawing();
        return;
      }
      if (event.key === "Backspace" || event.key === "Delete") {
        event.preventDefault();
        undoPoint();
        return;
      }
    }

    if (event.key === "Escape") {
      if (!els.demoPanel.classList.contains("hidden")) {
        closeDemoPanel();
        els.demoToggle.focus();
      }
      if (!els.searchResults.classList.contains("hidden")) {
        hideSearchResults();
      }
    }
  }

  /* ======================================================================
     Verification and result rendering
     ====================================================================== */

  function abortPendingVerify() {
    state.verifySeq += 1;
    if (state.loading) {
      state.loading = false;
      setVerifyLoading(false);
      setDrawingInputsDisabled(false);
    }
  }

  function setVerifyLoading(loading) {
    state.loading = Boolean(loading);
    setHidden(els.verifySpinner, !loading);
    els.resultStep.setAttribute("aria-busy", loading ? "true" : "false");
    updateVerifyState();
    updateAssistantUI();
  }

  function updateVerifyState() {
    const hasField = Boolean(state.selectedDemoId || state.geojson);
    const hasParameters = Boolean(els.cropSelect.value && els.sowingDate.value);
    els.verifyButton.disabled =
      state.loading || !hasField || !hasParameters || Boolean(state.mode);
    els.clearField.classList.toggle("hidden", !hasField);
  }

  async function verify(options = {}) {
    if (state.loading || state.mode) return;
    if (!els.sowingDate.value) {
      showFieldError(t("params.date"));
      els.sowingDate.focus();
      return;
    }
    const hasField = Boolean(state.selectedDemoId || state.geojson);
    if (!hasField) {
      showFieldError(t("field.noneHint"));
      showToast(t("field.noneHint"));
      return;
    }
    hideFieldError();
    const payload = {
      crop: els.cropSelect.value || "wheat",
      sowing_date: els.sowingDate.value,
      lang: state.lang,
    };
    if (state.selectedDemoId) {
      payload.demo_id = state.selectedDemoId;
    } else if (state.geojson) {
      payload.geojson = state.geojson;
    }
    const seq = ++state.verifySeq;
    setVerifyLoading(true);
    setDrawingInputsDisabled(true);
    showResultLoading();
    try {
      const result = await fetchJSON("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (seq !== state.verifySeq) return;
      state.lastResult = result;
      state.lastResultParams = {
        crop: payload.crop,
        sowing_date: payload.sowing_date,
      };
      state.chatMessages = [];
      switchResultView("result");
      renderResult(result);
      updateAssistantUI();
      saveAppState();
      if (options.scroll) {
        els.resultStep.scrollIntoView({
          behavior: REDUCED_MOTION ? "auto" : "smooth",
          block: "start",
        });
      }
    } catch (error) {
      if (seq !== state.verifySeq) return;
      state.lastResult = null;
      state.lastResultParams = null;
      state.chatMessages = [];
      showResultError(error.message || t("result.error"));
      renderAssistantMessages();
      updateAssistantUI();
      saveAppState();
      if (!options.silent) showToast(t("result.error"));
    } finally {
      if (seq === state.verifySeq) {
        setVerifyLoading(false);
        setDrawingInputsDisabled(false);
      }
    }
  }

  function showResultLoading() {
    stopTemporalPlayback();
    setHidden(els.temporalPanel, true);
    hideTemporalLegend();
    resizeMapAfterTemporalLayout();
    restoreTemporalMapStyle();
    setHidden(els.resultEmpty, true);
    setHidden(els.resultLoading, false);
    setHidden(els.resultError, true);
    setHidden(els.resultContent, true);
  }

  function showResultError(message) {
    destroyChart();
    stopTemporalPlayback();
    setHidden(els.temporalPanel, true);
    hideTemporalLegend();
    resizeMapAfterTemporalLayout();
    restoreTemporalMapStyle();
    setHidden(els.resultEmpty, true);
    setHidden(els.resultLoading, true);
    setHidden(els.resultContent, true);
    setHidden(els.resultError, false);
    els.resultErrorText.textContent = message || t("result.errorHint");
    els.resultStep.setAttribute("aria-busy", "false");
  }

  function resetResult() {
    abortPendingVerify();
    destroyChart();
    stopTemporalPlayback();
    setHidden(els.temporalPanel, true);
    hideTemporalLegend();
    resizeMapAfterTemporalLayout();
    restoreTemporalMapStyle();
    state.lastResult = null;
    state.lastResultParams = null;
    state.chatMessages = [];
    switchResultView("result");
    setHidden(els.resultLoading, true);
    setHidden(els.resultError, true);
    setHidden(els.resultContent, true);
    setHidden(els.resultEmpty, false);
    els.resultStep.setAttribute("aria-busy", "false");
    renderAssistantMessages();
    updateAssistantUI();
  }

  function destroyChart() {
    if (state.chart) {
      state.chart.destroy();
      state.chart = null;
    }
  }

  function renderResult(result) {
    setHidden(els.resultEmpty, true);
    setHidden(els.resultLoading, true);
    setHidden(els.resultError, true);
    setHidden(els.resultContent, false);

    const verdict = result.verdict || "doubtful";
    els.verdictDot.className = "status-dot v-" + verdict;
    els.verdictTitle.textContent = result.verdict_label || verdict;
    els.scoreValue.textContent = result.score == null ? "—" : String(result.score);
    els.verdictSummary.textContent =
      (result.explanation && result.explanation.summary) || "";

    const probability = result.ml ? result.ml.probability : null;
    const predictedLabel =
      (result.ml && result.ml.predicted_label) || t("features.none");
    const declaredCrop = result.field && result.field.crop_declared;
    const declaredLabel =
      (result.field && result.field.crop_declared_label) || t("features.none");
    const declaredProbability =
      declaredCrop && result.probabilities
        ? result.probabilities[declaredCrop]
        : (result.top3 || []).find((item) => item.crop === declaredCrop)?.probability;
    els.modelConfidence.textContent = t("result.modelSummary", {
      topCrop: predictedLabel,
      topPercent: fmtPercent(probability),
      declaredCrop: declaredLabel,
      declaredPercent: fmtPercent(declaredProbability),
    });
    setHidden(
      els.modelConfidenceNote,
      !result.ml || result.ml.source === "fallback"
    );
    els.mlSource.textContent =
      result.ml && result.ml.source === "fallback"
        ? t("result.mlFallback")
        : "RandomForest";

    const highScoreNonConfirmed =
      verdict !== "confirmed" && Number(result.score) >= 70;
    setHidden(els.guardNote, !highScoreNonConfirmed);

    const synthetic = result.source === "synthetic";
    els.sourceRow.classList.toggle("is-synthetic", synthetic);
    els.sourceLabel.textContent = result.source_label || result.source || "—";
    els.sourceNote.textContent = result.data_source_note || "";
    setFieldSource(result.source);

    renderChart(result);
    renderTemporalPanel(result);
    renderFeatures(result);
    els.explanation.textContent =
      (result.explanation && result.explanation.summary) || "";
    els.explanationSource.textContent = t(
      result.explanation_source === "llm"
        ? "explanation.source.llm"
        : "explanation.source.template"
    );
    renderReasons(result);
    els.recommendation.textContent =
      (result.explanation && result.explanation.recommendation) || "";
    renderTop3(result);
    renderWeather(result);
    renderHowContent();
    els.resultDisclaimer.textContent =
      (result.meta && (state.lang === "kz" ? result.meta.disclaimer_kz : result.meta.disclaimer_ru)) ||
      "";
    if (result.field && result.field.region_warning) {
      els.fieldHonesty.textContent =
        (els.fieldHonesty.textContent ? els.fieldHonesty.textContent + " " : "") +
        t("map.regionWarning");
      els.fieldHonesty.classList.remove("hidden");
      els.fieldHonesty.classList.add("region-warning");
    }
    els.pdfButton.disabled = false;
    els.resultStep.setAttribute("aria-busy", "false");
  }

  function switchResultView(name) {
    const assistant = name === "assistant";
    state.activeResultView = assistant ? "assistant" : "result";
    setHidden(els.resultView, assistant);
    setHidden(els.assistantView, !assistant);
    els.resultViewButtons.forEach((button) => {
      const active = button.dataset.resultView === (assistant ? "assistant" : "result");
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
    });
    updateAssistantUI();
    saveAppState();
  }

  function chatContext(result) {
    const features = result.features || {};
    const scalarValues = (values) =>
      Object.fromEntries(
        Object.entries(values || {}).filter(
          ([, value]) =>
            value == null ||
            ["string", "number", "boolean"].includes(typeof value)
        )
      );
    const weather = scalarValues(result.weather);
    if (result.weather && result.weather.period) {
      weather.period_start = result.weather.period.start || null;
      weather.period_end = result.weather.period.end || null;
    }
    return {
      field_name: (result.field && result.field.name) || "",
      crop: (result.field && result.field.crop_declared_label) || "",
      crop_code: (result.field && result.field.crop_declared) || "",
      sowing_date: (result.field && result.field.sowing_date) || null,
      area_ha:
        result.field && Number.isFinite(Number(result.field.area_ha))
          ? Number(result.field.area_ha)
          : null,
      verdict: result.verdict_label || "",
      score: Number(result.score) || 0,
      source: result.source_label || "",
      model_confidence:
        result.ml && Number.isFinite(Number(result.ml.probability))
          ? Number(result.ml.probability)
          : null,
      model_crop: (result.ml && result.ml.predicted_label) || "",
      model_top3: Array.isArray(result.top3) ? result.top3.slice(0, 5) : [],
      peak_ndvi: features.peak_ndvi == null ? null : Number(features.peak_ndvi),
      growth_start_date: features.growth_start_date || null,
      peak_date: features.peak_date || null,
      harvest_date: features.harvest_date || null,
      vegetation_days:
        features.vegetation_days == null ? null : Number(features.vegetation_days),
      correlation:
        features.correlation_with_reference == null
          ? null
          : Number(features.correlation_with_reference),
      precipitation_index_pct:
        features.precipitation_index_pct == null
          ? null
          : Number(features.precipitation_index_pct),
      features: scalarValues(features),
      weather,
      season: Array.isArray(result.series)
        ? result.series.slice(0, 40).map((point, index) => ({
            date: point.date,
            ndvi: point.ndvi,
            raw_ndvi:
              Array.isArray(result.ndvi_raw) && result.ndvi_raw[index] != null
                ? result.ndvi_raw[index]
                : null,
            reference_ndvi:
              Array.isArray(result.reference) &&
              result.reference[index] &&
              result.reference[index].ndvi != null
                ? result.reference[index].ndvi
                : null,
            measured: Boolean(point.measured),
            valid_fraction: point.valid_fraction,
            stage: point.stage,
          }))
        : [],
      explanation:
        (result.explanation && result.explanation.summary) || "",
      reasons:
        result.explanation && Array.isArray(result.explanation.reasons)
          ? result.explanation.reasons.slice(0, 8)
          : [],
      recommendation:
        (result.explanation && result.explanation.recommendation) || "",
    };
  }

  function updateAssistantUI() {
    const result = state.lastResult;
    const hasResult = Boolean(result);
    if (hasResult) {
      els.assistantContext.textContent = t("chat.context", {
        field: result.field && result.field.name ? result.field.name : t("field.custom"),
        crop:
          (result.field && result.field.crop_declared_label) || t("features.none"),
        verdict: result.verdict_label || t("features.none"),
        score: Number(result.score) || 0,
      });
    } else {
      els.assistantContext.textContent = t("chat.noResult");
    }
    els.assistantQuestions.forEach((button) => {
      button.disabled =
        !hasResult ||
        state.chatBusy ||
        !state.chatSessionId;
    });
    els.assistantInput.disabled =
      !hasResult ||
      state.chatBusy ||
      !state.chatSessionId ||
      !state.chatFreeTextEnabled;
    els.assistantSend.disabled =
      els.assistantInput.disabled || !els.assistantInput.value.trim();
    els.assistantLimit.textContent = state.chatFreeTextEnabled
      ? ""
      : t("chat.freeTextDisabled");
  }

  function renderAssistantMessages() {
    els.assistantMessages.replaceChildren();
    state.chatMessages.forEach((message, index) => {
      const item = document.createElement("article");
      item.className = "assistant-message " + message.role;
      if (message.role === "user") {
        item.textContent = message.text;
      } else {
        const label = document.createElement("span");
        label.className = "assistant-message-label";
        label.textContent =
          t("chat.tab") +
          " · " +
          t(message.explanationSource === "template" ? "chat.template" : "chat.ai");
        const answer = document.createElement("p");
        answer.textContent = message.text;
        item.append(label, answer);
        if (message.sources && message.sources.length) {
          const sources = document.createElement("div");
          sources.className = "assistant-sources";
          message.sources.forEach((source) => {
            const chip = document.createElement("span");
            chip.className = "assistant-source";
            chip.textContent = source;
            sources.append(chip);
          });
          item.append(sources);
        }
        const actions = document.createElement("div");
        actions.className = "assistant-actions";
        [
          ["copy", t("chat.copy")],
          ["useful", t("chat.useful")],
          ["inaccurate", t("chat.inaccurate")],
        ].forEach(([action, labelText]) => {
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.chatAction = action;
          button.dataset.messageIndex = String(index);
          button.textContent = labelText;
          button.setAttribute(
            "aria-pressed",
            message.feedback === action ? "true" : "false"
          );
          actions.append(button);
        });
        item.append(actions);
      }
      els.assistantMessages.append(item);
    });
  }

  async function sendAssistantQuestion(questionId, question) {
    if (
      !state.lastResult ||
      !state.chatSessionId ||
      !question ||
      state.chatBusy ||
      (!questionId && !state.chatFreeTextEnabled)
    ) {
      return;
    }
    const text = question.trim().slice(0, 500);
    state.chatMessages.push({ role: "user", text });
    saveAppState();
    state.chatBusy = true;
    setHidden(els.assistantError, true);
    setHidden(els.assistantLoading, false);
    if (!questionId) els.assistantInput.value = "";
    renderAssistantMessages();
    updateAssistantUI();
    try {
      const response = await fetchJSON("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: state.chatSessionId,
          question: text,
          question_id: questionId,
          lang: state.lang,
          context: chatContext(state.lastResult),
          history: state.chatMessages
            .slice(-7, -1)
            .map(({ role, text: messageText }) => ({
              role,
              text: messageText,
            })),
        }),
      });
      state.chatMessages.push({
        role: "assistant",
        text: response.answer,
        sources: Array.isArray(response.sources) ? response.sources : [],
        explanationSource: response.explanation_source,
      });
      saveAppState();
    } catch (error) {
      els.assistantError.textContent = error.status === 503
          ? t("chat.unavailable")
          : error.message || t("chat.unavailable");
      setHidden(els.assistantError, false);
    } finally {
      state.chatBusy = false;
      setHidden(els.assistantLoading, true);
      renderAssistantMessages();
      updateAssistantUI();
      saveAppState();
    }
  }

  async function onAssistantAction(event) {
    const button = event.target.closest("[data-chat-action]");
    if (!button) return;
    const message = state.chatMessages[Number(button.dataset.messageIndex)];
    if (!message || message.role !== "assistant") return;
    if (button.dataset.chatAction === "copy") {
      try {
        await navigator.clipboard.writeText(message.text);
        button.textContent = t("chat.copied");
      } catch (error) {
        showToast(t("chat.unavailable"));
      }
      return;
    }
    message.feedback =
      message.feedback === button.dataset.chatAction ? null : button.dataset.chatAction;
    renderAssistantMessages();
    saveAppState();
  }

  function featureRows(result) {
    const features = result.features || {};
    const noGrowthPeak =
      Number((state.config && state.config.verdict && state.config.verdict.no_growth_peak) ?? 0.3);
    const hasMeaningfulSowingShift =
      features.sowing_shift_days != null &&
      features.growth_start_date &&
      Number(features.peak_ndvi) >= noGrowthPeak;
    return [
      {
        label: t("features.growth"),
        value: features.growth_start_date
          ? fmtDate(features.growth_start_date, false)
          : t("features.none"),
      },
      {
        label: t("features.peak"),
        value:
          features.peak_ndvi == null ? t("features.none") : fmtNum(features.peak_ndvi, 2),
      },
      {
        label: t("features.peakDate"),
        value: features.peak_date
          ? fmtDate(features.peak_date, false)
          : t("features.none"),
      },
      {
        label: t("features.duration"),
        value:
          features.vegetation_days == null
            ? t("features.none")
            : features.vegetation_days + " " + t("features.days"),
      },
      {
        label: t("features.decline"),
        value: features.harvest_date
          ? fmtDate(features.harvest_date, false)
          : t("features.declineMissing"),
      },
      {
        label: t("features.correlation"),
        value:
          features.correlation_with_reference == null
            ? t("features.none")
            : fmtNum(features.correlation_with_reference, 2),
      },
      {
        label: t("features.precip"),
        value:
          features.precipitation_index_pct == null
            ? t("features.none")
            : fmtNum(features.precipitation_index_pct, 0) + " %",
      },
      !hasMeaningfulSowingShift
        ? null
        : {
        label: t("features.shift"),
        value: shiftLabel(features.sowing_shift_days),
      },
    ].filter(Boolean);
  }

  function renderFeatures(result) {
    els.featuresGrid.innerHTML = featureRows(result)
      .map(
        (item) =>
          '<div class="feature-item"><dt>' +
          esc(item.label) +
          "</dt><dd>" +
          esc(item.value) +
          "</dd></div>"
      )
      .join("");
  }

  function renderReasons(result) {
    const reasons =
      result.explanation && Array.isArray(result.explanation.reasons)
        ? result.explanation.reasons
        : [];
    els.reasons.innerHTML = reasons
      .map((reason) => "<li>" + esc(reason) + "</li>")
      .join("");
  }

  function renderTop3(result) {
    const top3 = Array.isArray(result.top3) ? result.top3 : [];
    const declared = (result.field && result.field.crop_declared) || "";
    els.top3List.innerHTML = top3
      .map((item) => {
        const labelKey = state.lang === "kz" ? "label_kz" : "label_ru";
        const label = item[labelKey] || item.crop || "";
        const percent = Math.round(Number(item.probability || 0) * 100);
        const declaredChip =
          item.crop === declared
            ? '<span class="declared">' + esc(t("top3.declared")) + "</span>"
            : "";
        return (
          '<div class="top3-item"><div class="top3-head"><strong>' +
          esc(label) +
          "</strong>" +
          declaredChip +
          "<span>" +
          esc(percent + " %") +
          "</span></div>" +
          '<div class="top3-bar"><i style="width:' +
          clamp(percent, 0, 100) +
          '%"></i></div></div>'
        );
      })
      .join("");
  }

  function renderWeather(result) {
    const weather = result.weather || {};
    const block = $(".weather-block");
    if (!weather || !Object.keys(weather).length) {
      if (block) block.classList.add("hidden");
      return;
    }
    if (block) block.classList.remove("hidden");
    const source = weather.source === "climate_norm" ? t("how.weather") : "";
    els.weatherSource.textContent =
      (weather.source_label || "") +
      (weather.cached && weather.source === "open-meteo" ? " · " + t("weather.cached") : "");
    if (source) {
      els.weatherSource.title = source;
    }
    const chips = [];
    if (weather.precipitation_mm != null) {
      chips.push(
        '<span class="weather-chip">' +
          esc(t("weather.precip")) +
          " <strong>" +
          esc(fmtNum(weather.precipitation_mm, 1) + " мм") +
          "</strong></span>"
      );
    }
    if (weather.precipitation_index != null) {
      chips.push(
        '<span class="weather-chip"><strong>' +
          esc(fmtNum(weather.precipitation_index, 0) + " %") +
          "</strong> " +
          esc(t("weather.index")) +
          "</span>"
      );
    }
    if (weather.temperature_mean_c != null) {
      chips.push(
        '<span class="weather-chip">' +
          esc(t("weather.temp")) +
          " <strong>" +
          esc(fmtNum(weather.temperature_mean_c, 1) + " °C") +
          "</strong></span>"
      );
    }
    els.weatherChips.innerHTML = chips.join("");
    els.weatherContext.textContent = "";
    els.weatherContext.classList.add("hidden");
  }

  function renderHowContent() {
    if (!els.howContent) return;
    const config = state.config || {};
    const score = config.score || {};
    const verdict = config.verdict || {};
    const full = score.timing_full_days == null ? 15 : score.timing_full_days;
    const zero = score.timing_zero_days == null ? 60 : score.timing_zero_days;
    const confirmed = score.confirmed_min == null ? 70 : score.confirmed_min;
    const doubtful = score.doubtful_min == null ? 40 : score.doubtful_min;
    const noGrowth = verdict.no_growth_peak == null ? 0.3 : verdict.no_growth_peak;
    const flat = verdict.flat_peak_max == null ? 0.4 : verdict.flat_peak_max;
    const lowPeak = verdict.low_peak_ratio == null ? 0.65 : verdict.low_peak_ratio;
    const mismatch =
      verdict.crop_mismatch_probability == null
        ? 0.6
        : verdict.crop_mismatch_probability;

    els.howContent.innerHTML = [
      "<p>" + esc(t("how.score")) + "</p>",
      "<ul>",
      "<li>" + esc(t("how.shape")) + "</li>",
      "<li>" + esc(t("how.timing", { full: full, zero: zero })) + "</li>",
      "<li>" + esc(t("how.peak")) + "</li>",
      "<li>" + esc(t("how.model")) + "</li>",
      "</ul>",
      "<p>" +
        esc(
          t("how.verdicts", {
            confirmed: confirmed,
            doubtful: doubtful,
            noGrowth: noGrowth,
            flat: flat,
            lowPeak: lowPeak,
            mismatch: mismatch,
          })
        ) +
        "</p>",
      "<p>" + esc(t("how.data")) + "</p>",
      "<p>" + esc(t("how.weather")) + "</p>",
      "<p>" + esc(t("how.limits")) + "</p>",
    ].join("");
  }

  /* ======================================================================
     Chart
     ====================================================================== */

  function monthLabel(iso) {
    const months = t("months");
    const list = Array.isArray(months) ? months : [];
    const month = Number(String(iso).slice(5, 7)) - 1;
    return list[month] || "";
  }

  function nearestDateIndex(labels, iso) {
    if (!iso) return -1;
    const target = new Date(String(iso).slice(0, 10) + "T00:00:00").getTime();
    if (Number.isNaN(target)) return -1;
    let best = -1;
    let bestDiff = Infinity;
    labels.forEach((label, index) => {
      const value = new Date(String(label).slice(0, 10) + "T00:00:00").getTime();
      const diff = Math.abs(value - target);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = index;
      }
    });
    return best;
  }

  function temporalMonth(dateValue) {
    const months = t("months");
    const month = Number(String(dateValue).slice(5, 7)) - 1;
    return Array.isArray(months) ? months[month] || "" : "";
  }

  function temporalRangeLabel(series, index) {
    const start = new Date(series[index].date + "T00:00:00");
    const next = index + 1 < series.length
      ? new Date(series[index + 1].date + "T00:00:00")
      : new Date(start.getTime());
    if (index + 1 < series.length) next.setDate(next.getDate() - 1);
    const startDay = start.getDate();
    const endDay = next.getDate();
    const startMonth = temporalMonth(series[index].date);
    const endMonth = temporalMonth(
      next.getFullYear() +
        "-" +
        String(next.getMonth() + 1).padStart(2, "0") +
        "-" +
        String(next.getDate()).padStart(2, "0")
    );
    if (startMonth === endMonth && startDay !== endDay) {
      return startDay + "–" + endDay + " " + startMonth;
    }
    if (startDay === endDay && startMonth === endMonth) {
      return startDay + " " + startMonth;
    }
    return startDay + " " + startMonth + " – " + endDay + " " + endMonth;
  }

  function temporalPosition(index, length) {
    return length > 1 ? (index / (length - 1)) * 100 : 0;
  }

  function renderTemporalMarkers(result, series) {
    els.temporalMarkers.replaceChildren();
    els.temporalMonths.replaceChildren();
    const markers = [
      { date: result.field && result.field.sowing_date, key: "temporal.marker.sowing" },
      { date: result.features && result.features.peak_date, key: "temporal.marker.peak" },
      { date: result.features && result.features.harvest_date, key: "temporal.marker.decline" },
    ];
    markers.forEach((marker) => {
      const index = nearestDateIndex(
        series.map((point) => point.date),
        marker.date
      );
      if (index < 0) return;
      const node = document.createElement("span");
      node.className = "temporal-marker";
      node.style.left = temporalPosition(index, series.length) + "%";
      node.textContent = t(marker.key);
      els.temporalMarkers.append(node);
    });

    let previousMonth = "";
    series.forEach((point, index) => {
      const month = point.date.slice(0, 7);
      if (month === previousMonth) return;
      previousMonth = month;
      const node = document.createElement("span");
      node.className = "temporal-month";
      node.style.left = temporalPosition(index, series.length) + "%";
      node.textContent = temporalMonth(point.date);
      els.temporalMonths.append(node);
    });
  }

  function hideTemporalLegend() {
    setHidden(els.temporalLegend, true);
    els.temporalLegend.setAttribute("aria-hidden", "true");
  }

  function renderTemporalLegend(result) {
    const palettes = result.temporal_palettes || {};
    const palette =
      state.temporalMode === "deviation" ? palettes.deviation : palettes.ndvi;
    if (!Array.isArray(palette) || !palette.length) {
      hideTemporalLegend();
      return;
    }
    els.temporalLegendColors.replaceChildren();
    palette.forEach((color) => {
      const swatch = document.createElement("span");
      swatch.className = "temporal-legend-swatch";
      swatch.style.backgroundColor = color;
      els.temporalLegendColors.append(swatch);
    });
    const deviation = state.temporalMode === "deviation";
    els.temporalLegendTitle.textContent = t(
      deviation ? "temporal.legend.deviation" : "temporal.legend.ndvi"
    );
    els.temporalLegendMin.textContent = deviation ? "−0,5" : "0";
    els.temporalLegendMax.textContent = deviation ? "+0,5" : "1";
    setHidden(els.temporalLegend, false);
    els.temporalLegend.setAttribute("aria-hidden", "false");
  }

  function renderTemporalPanel(result) {
    const series = Array.isArray(result.series) ? result.series : [];
    stopTemporalPlayback();
    if (!series.length) {
      setHidden(els.temporalPanel, true);
      hideTemporalLegend();
      resizeMapAfterTemporalLayout();
      restoreTemporalMapStyle();
      return;
    }
    state.temporalIndex = 0;
    state.temporalMode = "ndvi";
    els.temporalSlider.min = "0";
    els.temporalSlider.max = String(series.length - 1);
    els.temporalSlider.step = "1";
    els.temporalSlider.disabled = false;
    els.temporalPlay.disabled = series.length < 2;
    els.temporalPanel.setAttribute("aria-label", t("temporal.panelLabel"));
    els.temporalSlider.setAttribute("aria-label", t("temporal.sliderLabel"));
    els.temporalModeGroup.setAttribute("aria-label", t("temporal.modeLabel"));
    renderTemporalMarkers(result, series);
    els.temporalModeButtons.forEach((button) => {
      const active = button.dataset.temporalMode === state.temporalMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    setHidden(els.temporalPanel, false);
    renderTemporalLegend(result);
    updateTemporalSelection();
    resizeMapAfterTemporalLayout();
  }

  function resizeMapAfterTemporalLayout() {
    if (!state.map) return;
    window.requestAnimationFrame(() => {
      if (state.map) state.map.invalidateSize({ pan: false });
    });
  }

  function updateTemporalSelection() {
    const result = state.lastResult;
    const series = result && Array.isArray(result.series) ? result.series : [];
    if (!series.length) {
      setHidden(els.temporalPanel, true);
      hideTemporalLegend();
      return;
    }
    state.temporalIndex = clamp(state.temporalIndex, 0, series.length - 1);
    const index = state.temporalIndex;
    const point = series[index];
    const ndvi = Number(point.ndvi);
    const referencePoint =
      Array.isArray(result.reference) ? result.reference[index] : null;
    const reference =
      referencePoint == null
        ? null
        : Number(
            typeof referencePoint === "object"
              ? referencePoint.ndvi
              : referencePoint
          );
    const deviation = reference == null ? null : ndvi - reference;
    const stage = point.stage || "unknown";
    const range = temporalRangeLabel(series, index);
    const stageLabel = t("temporal.stage." + stage);
    els.temporalDate.textContent = range;
    els.temporalStage.textContent = t("temporal.stageLabel", { stage: stageLabel });
    els.temporalValue.textContent =
      state.temporalMode === "deviation"
        ? t("temporal.reading.deviation", {
            deviation: deviation == null ? t("features.none") : fmtNum(deviation, 2),
            reference: reference == null ? t("features.none") : fmtNum(reference, 2),
          })
        : t("temporal.reading.ndvi", {
            ndvi: fmtNum(ndvi, 2),
            reference: reference == null ? t("features.none") : fmtNum(reference, 2),
            deviation: deviation == null ? t("features.none") : fmtNum(deviation, 2),
          });
    els.temporalSource.textContent = t(
      result.source === "synthetic" ? "temporal.source.synthetic" : "temporal.source.satellite"
    );
    els.temporalSource.classList.toggle("synthetic", result.source === "synthetic");
    setHidden(els.temporalRestored, Boolean(point.measured));
    els.temporalSlider.value = String(index);
    els.temporalSlider.setAttribute(
      "aria-valuetext",
      t("temporal.ariaValue", {
        date: range,
        ndvi: fmtNum(ndvi, 2),
        stage: stageLabel,
      })
    );
    els.temporalSlider.style.setProperty(
      "--temporal-progress",
      temporalPosition(index, series.length) + "%"
    );
    const paused = state.temporalPlayback == null;
    els.temporalPlay.setAttribute("aria-label", t(paused ? "temporal.play" : "temporal.pause"));
    els.temporalPlay.setAttribute("aria-pressed", paused ? "false" : "true");
    els.temporalPlay.querySelector("span").textContent = paused ? "▶" : "Ⅱ";
    renderTemporalLegend(result);
    applyTemporalMapColor(point);
    if (state.chart) state.chart.draw();
  }

  function applyTemporalMapColor(point) {
    const color =
      state.temporalMode === "deviation"
        ? point.deviation_color
        : point.ndvi_color;
    const layer = state.selectedDemoId
      ? state.demoLayers.get(state.selectedDemoId)
      : state.finalLayer;
    if (layer && color) layer.setStyle({ fillColor: color, fillOpacity: 0.82 });
  }

  function restoreTemporalMapStyle() {
    if (state.selectedDemoId) {
      const layer = state.demoLayers.get(state.selectedDemoId);
      if (layer) layer.setStyle(demoStyle(true));
    } else if (state.finalLayer) {
      state.finalLayer.setStyle({ fillColor: "#ffffff", fillOpacity: 0.12 });
    }
  }

  function setTemporalMode(mode) {
    if (mode !== "ndvi" && mode !== "deviation") return;
    state.temporalMode = mode;
    els.temporalModeButtons.forEach((button) => {
      const active = button.dataset.temporalMode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    updateTemporalSelection();
  }

  function stopTemporalPlayback() {
    if (state.temporalPlayback != null) {
      window.clearInterval(state.temporalPlayback);
      state.temporalPlayback = null;
    }
  }

  function toggleTemporalPlayback() {
    const series = state.lastResult && state.lastResult.series;
    if (!Array.isArray(series) || series.length < 2) return;
    if (state.temporalPlayback != null) {
      stopTemporalPlayback();
      updateTemporalSelection();
      return;
    }
    if (state.temporalIndex >= series.length - 1) state.temporalIndex = 0;
    updateTemporalSelection();
    state.temporalPlayback = window.setInterval(() => {
      state.temporalIndex += 1;
      if (state.temporalIndex >= series.length - 1) stopTemporalPlayback();
      updateTemporalSelection();
    }, 600);
    updateTemporalSelection();
  }

  const seasonMarkersPlugin = {
    id: "seasonMarkers",
    afterDatasetsDraw(chart) {
      const markers =
        (chart.options.plugins &&
          chart.options.plugins.seasonMarkers &&
          chart.options.plugins.seasonMarkers.markers) ||
        [];
      if (!markers.length) return;
      const { ctx, chartArea, scales } = chart;
      if (!chartArea || !scales || !scales.x) return;
      ctx.save();
      ctx.font = '500 9px "Inter", sans-serif';
      ctx.fillStyle = "#7f868c";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      markers.forEach((marker) => {
        if (marker.index == null || marker.index < 0) return;
        const x = scales.x.getPixelForValue(marker.index);
        if (!Number.isFinite(x)) return;
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "#b7bdc2";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top);
        ctx.lineTo(x, chartArea.bottom);
        ctx.stroke();
        ctx.restore();
        ctx.fillText(marker.label, x, chartArea.top - 3);
      });
      ctx.restore();
    },
  };

  const temporalCursorPlugin = {
    id: "temporalCursor",
    afterDatasetsDraw(chart) {
      if (!state.lastResult || !chart.chartArea || !chart.scales || !chart.scales.x) return;
      const index = state.temporalIndex;
      const x = chart.scales.x.getPixelForValue(index);
      if (!Number.isFinite(x)) return;
      const { ctx, chartArea } = chart;
      const value = chart.data.datasets[0].data[index];
      const y = chart.scales.y.getPixelForValue(value);
      ctx.save();
      ctx.strokeStyle = "#1f6f5c";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.fillStyle = "#1f6f5c";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      chart.canvas.dataset.temporalCursorIndex = String(index);
      chart.canvas.dataset.temporalCursorX = String(x);
    },
  };

  function renderChart(result) {
    destroyChart();
    if (typeof Chart === "undefined") return;
    const canvas = els.chartCanvas;
    if (!canvas) return;
    const series = result.series || [];
    const labels = series.map((point) => point.date);
    const values = series.map((point) => Number(point.ndvi));
    const measured = series.map((point) => Boolean(point.measured));
    const reference = Array.isArray(result.reference)
      ? result.reference.map((point) =>
          Number(typeof point === "object" ? point.ndvi : point)
        )
      : [];

    Chart.defaults.font.family = '"Inter", "Segoe UI", sans-serif';
    Chart.defaults.color = "#7f868c";

    const months = t("months");
    const monthList = Array.isArray(months) ? months : [];
    const markerSpecs = [
      { date: result.field && result.field.sowing_date, label: t("marker.sowing") },
      { date: result.features && result.features.peak_date, label: t("marker.peak") },
      { date: result.features && result.features.harvest_date, label: t("marker.harvest") },
    ];
    const markers = markerSpecs
      .map((spec) => ({
        label: spec.label,
        index: nearestDateIndex(labels, spec.date),
      }))
      .filter((marker) => marker.index >= 0);

    state.chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: t("legend.measured"),
            data: values,
            borderColor: "#0f5a45",
            backgroundColor: "#0f5a45",
            borderWidth: 2,
            tension: 0.28,
            cubicInterpolationMode: "monotone",
            fill: false,
            pointRadius: measured.map((flag) => (flag ? 2.8 : 2.2)),
            pointHoverRadius: 4.5,
            pointBackgroundColor: measured.map((flag) => (flag ? "#0f5a45" : "#ffffff")),
            pointBorderColor: "#0f5a45",
            pointBorderWidth: 1.4,
            segment: {
              borderDash: (context) =>
                measured[context.p0DataIndex] && measured[context.p1DataIndex]
                  ? undefined
                  : [4, 3],
            },
          },
          {
            label: t("legend.reference"),
            data: reference,
            borderColor: "#a9b0b6",
            borderWidth: 1.4,
            borderDash: [4, 4],
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0.25,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        layout: { padding: { top: 18, right: 4, bottom: 0, left: 0 } },
        scales: {
          x: {
            grid: { display: false },
            border: { display: false },
            ticks: {
              color: "#8c9298",
              font: { size: 9.5, weight: "400" },
              autoSkip: false,
              maxRotation: 0,
              callback(value, index) {
                if (index === 0) return monthList[Number(labels[0].slice(5, 7)) - 1] || "";
                const current = String(this.getLabelForValue(value));
                const previous = String(this.getLabelForValue(value - 1));
                if (current.slice(0, 7) !== previous.slice(0, 7)) {
                  return monthList[Number(current.slice(5, 7)) - 1] || "";
                }
                return "";
              },
            },
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: "#eceef0", drawTicks: false },
            border: { display: false },
            ticks: {
              color: "#8c9298",
              font: { size: 9.5, weight: "400" },
              stepSize: 0.2,
              padding: 4,
              callback(value) {
                return Number(value).toFixed(1);
              },
            },
          },
        },
        plugins: {
          legend: { display: false },
          seasonMarkers: { markers },
          tooltip: {
            backgroundColor: "#1a1d21",
            titleFont: { size: 10.5, weight: "500" },
            bodyFont: { size: 10.5 },
            padding: 8,
            displayColors: false,
            callbacks: {
              title(items) {
                return items.length ? fmtDate(items[0].label) : "";
              },
              label(item) {
                const point = series[item.dataIndex];
                if (item.datasetIndex === 0) {
                  const stateLabel = point && point.measured
                    ? t("legend.measured")
                    : t("legend.interpolated");
                  return stateLabel + ": " + fmtNum(item.parsed.y, 2);
                }
                return t("legend.reference") + ": " + fmtNum(item.parsed.y, 2);
              },
            },
          },
        },
      },
      plugins: [seasonMarkersPlugin, temporalCursorPlugin],
    });

    const summary = (result.explanation && result.explanation.summary) || "";
    canvas.setAttribute(
      "aria-label",
      t("result.chart") + ". " + summary
    );
  }

  /* ======================================================================
     Search
     ====================================================================== */

  function parseCoordinates(text) {
    const match = String(text)
      .trim()
      .match(/^(-?\d{1,2}(?:[.,]\d+)?)\s*[,;\s]\s*(-?\d{1,3}(?:[.,]\d+)?)$/);
    if (!match) return null;
    const lat = Number(match[1].replace(",", "."));
    const lon = Number(match[2].replace(",", "."));
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
    return { lat, lon };
  }

  function onSearchInput() {
    const query = els.searchInput.value.trim();
    if (state.searchTimer) window.clearTimeout(state.searchTimer);
    if (query.length < 3) {
      hideSearchResults();
      return;
    }
    state.searchTimer = window.setTimeout(() => runSearch(query), 480);
  }

  function onSearchKeydown(event) {
    if (event.key === "Escape") {
      hideSearchResults();
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!state.searchResults.length) {
        if (els.searchInput.value.trim().length >= 3) {
          event.preventDefault();
          runSearch(els.searchInput.value.trim());
        }
        return;
      }
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      state.searchIndex =
        (state.searchIndex + direction + state.searchResults.length) %
        state.searchResults.length;
      updateSearchActive();
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (state.searchIndex >= 0 && state.searchResults[state.searchIndex]) {
        chooseSearchResult(state.searchResults[state.searchIndex]);
      } else {
        runSearch(els.searchInput.value.trim());
      }
    }
  }

  async function runSearch(query) {
    if (!query || query.length < 3) return;
    const coordinates = parseCoordinates(query);
    if (coordinates) {
      applyCoordinates(coordinates);
      return;
    }
    const seq = ++state.searchSeq;
    try {
      const payload = await fetchJSON(
        "/api/geocode?q=" +
          encodeURIComponent(query) +
          "&lang=" +
          state.lang +
          "&limit=5"
      );
      if (seq !== state.searchSeq) return;
      state.searchResults = Array.isArray(payload.results) ? payload.results : [];
      state.searchIndex = state.searchResults.length ? 0 : -1;
      renderSearchResults();
    } catch (error) {
      if (seq !== state.searchSeq) return;
      renderSearchError(t("map.searchError"));
    }
  }

  function renderSearchResults() {
    if (!state.searchResults.length) {
      renderSearchError(t("map.searchEmpty"));
      return;
    }
    els.searchResults.innerHTML = state.searchResults
      .map((result, index) => {
        const label =
          result.display_name && result.display_name.length > 96
            ? result.display_name.slice(0, 93) + "…"
            : result.display_name || "—";
        return (
          '<button type="button" role="option" class="search-result' +
          (index === state.searchIndex ? " active" : "") +
          '" id="search-result-' +
          index +
          '" data-result-index="' +
          index +
          '" aria-selected="' +
          (index === state.searchIndex ? "true" : "false") +
          '"><strong>' +
          esc(label) +
          "</strong><span>" +
          esc(fmtNum(result.lat, 5) + ", " + fmtNum(result.lon, 5)) +
          "</span></button>"
        );
      })
      .join("");
    els.searchResults.classList.remove("hidden");
    updateSearchActive();
  }

  function renderSearchError(message) {
    state.searchResults = [];
    state.searchIndex = -1;
    els.searchResults.innerHTML =
      '<p class="search-empty">' + esc(message) + "</p>";
    els.searchResults.classList.remove("hidden");
  }

  function updateSearchActive() {
    const options = $$(".search-result", els.searchResults);
    options.forEach((option, index) => {
      const active = index === state.searchIndex;
      option.classList.toggle("active", active);
      option.setAttribute("aria-selected", active ? "true" : "false");
      if (active) {
        option.scrollIntoView({ block: "nearest" });
        els.searchInput.setAttribute("aria-activedescendant", option.id);
      }
    });
  }

  function hideSearchResults() {
    state.searchResults = [];
    state.searchIndex = -1;
    els.searchResults.classList.add("hidden");
    els.searchResults.innerHTML = "";
    els.searchInput.removeAttribute("aria-activedescendant");
  }

  function chooseSearchResult(result) {
    if (!result) return;
    hideSearchResults();
    const name = result.display_name || fmtNum(result.lat, 5) + ", " + fmtNum(result.lon, 5);
    els.searchInput.value = name;
    if (!state.map) return;
    if (Array.isArray(result.boundingbox) && result.boundingbox.length === 4) {
      const south = Number(result.boundingbox[0]);
      const north = Number(result.boundingbox[1]);
      const west = Number(result.boundingbox[2]);
      const east = Number(result.boundingbox[3]);
      if ([south, north, west, east].every(Number.isFinite)) {
        state.map.fitBounds(
          [
            [south, west],
            [north, east],
          ],
          { padding: [24, 24], maxZoom: 14 }
        );
      }
    } else {
      state.map.setView([result.lat, result.lon], 13);
    }
    addSearchMarker(result.lat, result.lon, name);
    els.searchInput.blur();
  }

  function applyCoordinates(coordinates) {
    if (!state.map) return;
    const { lat, lon } = coordinates;
    hideSearchResults();
    state.map.setView([lat, lon], 13);
    const label = fmtNum(lat, 5) + ", " + fmtNum(lon, 5);
    els.searchInput.value = label;
    addSearchMarker(lat, lon, label);
  }

  function addSearchMarker(lat, lon, label) {
    if (!state.map || typeof L === "undefined") return;
    if (state.searchMarker) state.map.removeLayer(state.searchMarker);
    state.searchMarker = L.circleMarker([lat, lon], {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: "#0f5a45",
      fillOpacity: 1,
      interactive: true,
    }).addTo(state.map);
    state.searchMarker.bindTooltip(label, {
      direction: "top",
      className: "portfolio-label",
      opacity: 1,
    });
  }

  /* ======================================================================
     Portfolio
     ====================================================================== */

  async function loadSample(options = {}) {
    if (state.batchLoading) return;
    setBatchLoading(true);
    try {
      const sample = await fetchJSON("/api/sample-batch?lang=" + state.lang);
      const rows = (sample.rows || []).map((row) => ({
        id: row.id,
        crop: row.crop,
        sowing_date: row.sowing_date,
        lat: row.lat,
        lon: row.lon,
        scenario: row.scenario,
      }));
      await runBatch(rows, { silent: options.silent, sourceRows: rows });
    } catch (error) {
      setBatchLoading(false);
      showToast(t("portfolio.error") + ": " + (error.message || ""));
    }
  }

  async function runBatch(rows, options = {}) {
    setBatchLoading(true);
    try {
      const payload = await fetchJSON("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang: state.lang, rows }),
      });
      state.batchRows = Array.isArray(payload.rows) ? payload.rows : [];
      state.batchMeta = payload;
      state.portfolioLoaded = true;
      state.lastBatchAt = new Date();
      if (options.sourceFile) state.batchFile = options.sourceFile;
      else state.batchFile = null;
      if (options.sourceRows) state.batchInputRows = options.sourceRows;
      else state.batchInputRows = null;
      renderPortfolio();
    } catch (error) {
      showToast(t("portfolio.error") + ": " + (error.message || ""));
    } finally {
      setBatchLoading(false);
    }
  }

  function setBatchLoading(loading) {
    state.batchLoading = Boolean(loading);
    setHidden(els.portfolioLoading, !loading);
    els.loadSample.disabled = loading;
    if (loading) setHidden(els.portfolioEmpty, true);
  }

  async function onCsvChange(event) {
    const file = els.csvInput.files && els.csvInput.files[0];
    if (!file) return;
    if (file.size > 2_000_000) {
      showToast(t("portfolio.error"));
      els.csvInput.value = "";
      return;
    }
    await uploadBatchFile(file, { silent: false });
    els.csvInput.value = "";
  }

  async function uploadBatchFile(file, options = {}) {
    if (!file || state.batchLoading) return;
    setBatchLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("lang", state.lang);
      const response = await fetch("/api/batch", {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        let message = response.status + " " + (response.statusText || "");
        try {
          const payload = await response.json();
          if (payload && payload.detail) {
            message =
              typeof payload.detail === "string"
                ? payload.detail
                : JSON.stringify(payload.detail);
          }
        } catch (error) {
          /* keep the status message */
        }
        throw new Error(message.trim());
      }
      const payload = await response.json();
      state.batchRows = Array.isArray(payload.rows) ? payload.rows : [];
      state.batchMeta = payload;
      state.portfolioLoaded = true;
      state.lastBatchAt = new Date();
      state.batchFile = file;
      state.batchInputRows = null;
      renderPortfolio();
    } catch (error) {
      showToast(t("portfolio.error") + ": " + (error.message || ""));
    } finally {
      setBatchLoading(false);
    }
  }

  function reloadBatchForLanguage() {
    if (state.batchLoading) return;
    if (state.batchFile) {
      uploadBatchFile(state.batchFile, { silent: true });
    } else if (state.batchInputRows) {
      runBatch(state.batchInputRows, { silent: true, sourceRows: state.batchInputRows });
    }
  }

  function sortRows(rows) {
    const key = state.sortKey;
    const direction = state.sortDir === "asc" ? 1 : -1;
    return rows.slice().sort((first, second) => {
      let comparison = 0;
      if (key === "score") {
        comparison = Number(first.score || 0) - Number(second.score || 0);
      } else if (key === "verdict") {
        comparison =
          (RISK_ORDER[first.verdict] == null ? 9 : RISK_ORDER[first.verdict]) -
          (RISK_ORDER[second.verdict] == null ? 9 : RISK_ORDER[second.verdict]);
      } else if (key === "crop") {
        comparison = String(first.crop_label || first.crop || "").localeCompare(
          String(second.crop_label || second.crop || ""),
          locale()
        );
      } else {
        comparison = String(first.id || "").localeCompare(
          String(second.id || ""),
          locale(),
          { numeric: true }
        );
      }
      if (comparison === 0) {
        comparison = Number(first.score || 0) - Number(second.score || 0);
      }
      return comparison * direction;
    });
  }

  function shortReason(row) {
    const summary = row.summary || row.recommendation || row.reason_code || "—";
    const firstSentence = String(summary).split(/(?<=\.)\s/)[0] || String(summary);
    return firstSentence.length > 140
      ? firstSentence.slice(0, 137) + "…"
      : firstSentence;
  }

  function renderPortfolio() {
    const filter = els.verdictFilter.value;
    const allRows = state.batchRows || [];
    const filtered = filter
      ? allRows.filter((row) => row.verdict === filter)
      : allRows.slice();
    const rows = sortRows(filtered);
    state.portfolioView = rows;

    els.portfolioCount.textContent = allRows.length
      ? t("portfolio.shown", { shown: rows.length, total: allRows.length })
      : t("portfolio.empty");
    const source = (state.batchMeta && state.batchMeta.source) || "synthetic";
    const sourceLabel =
      (state.batchMeta && state.batchMeta.source_label) ||
      (source === "sentinel-2-l2a" ? t("field.real") : t("field.synthetic"));
    els.portfolioSource.textContent = sourceLabel;
    els.portfolioSource.className =
      "source-chip " + (source === "sentinel-2-l2a" ? "real" : "synthetic");
    if (state.batchMeta && state.batchMeta.meta && state.batchMeta.meta.note) {
      els.portfolioNote.textContent = state.batchMeta.meta.note;
    } else {
      els.portfolioNote.textContent = t("portfolio.note");
    }
    els.portfolioTimestamp.textContent = t("portfolio.columns");
    els.portfolioTimestamp.title = state.lastBatchAt
      ? state.lastBatchAt.toLocaleTimeString(locale(), {
          hour: "2-digit",
          minute: "2-digit",
        })
      : "";

    if (!allRows.length) {
      els.portfolioBody.innerHTML = "";
      setHidden(els.portfolioTableWrap, true);
      els.portfolioEmpty.querySelector("p").textContent = t("portfolio.empty");
      setHidden(els.portfolioEmpty, false);
      updateSortIndicators();
      saveAppState();
      return;
    }
    if (!rows.length) {
      els.portfolioBody.innerHTML = "";
      setHidden(els.portfolioTableWrap, true);
      els.portfolioEmpty.querySelector("p").textContent = t("portfolio.noMatch");
      setHidden(els.portfolioEmpty, false);
      updateSortIndicators();
      saveAppState();
      return;
    }

    setHidden(els.portfolioEmpty, true);
    setHidden(els.portfolioTableWrap, false);
    els.portfolioBody.innerHTML = rows
      .map(
        (row) =>
          '<tr data-row-id="' +
          esc(row.id) +
          '" tabindex="0" aria-label="' +
          esc(row.id + ", " + row.verdict_label + ", " + row.score) +
          '">' +
          '<td class="field-id">' +
          esc(row.id) +
          "</td>" +
          "<td>" +
          esc(row.crop_label || row.crop) +
          "</td>" +
          '<td class="numeric">' +
          esc(row.score) +
          "</td>" +
          '<td><span class="verdict-cell"><i class="status-dot v-' +
          esc(row.verdict) +
          '"></i>' +
          esc(row.verdict_label) +
          "</span></td>" +
          '<td class="reason" title="' +
          esc(row.summary || "") +
          '">' +
          esc(shortReason(row)) +
          "</td></tr>"
      )
      .join("");
    updateSortIndicators();
    saveAppState();
  }

  function updateSortIndicators() {
    const headers = $$("th[data-sort]", els.portfolioTable);
    headers.forEach((header) => {
      const key = header.dataset.sort;
      const active = key === state.sortKey;
      const mark = $(".sort-mark", header);
      if (mark) {
        mark.textContent = active ? (state.sortDir === "asc" ? "↑" : "↓") : "";
      }
      header.setAttribute(
        "aria-sort",
        active ? (state.sortDir === "asc" ? "ascending" : "descending") : "none"
      );
    });
  }

  function onSort(key) {
    if (!key) return;
    if (state.sortKey === key) {
      state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = key;
      state.sortDir = key === "score" ? "desc" : "asc";
    }
    renderPortfolio();
    saveAppState();
  }

  function openPortfolioRow(id) {
    const row = state.batchRows.find((item) => item.id === id);
    if (!row) return;
    switchTab("check");
    window.scrollTo({ top: 0, behavior: REDUCED_MOTION ? "auto" : "smooth" });
    abortPendingVerify();
    removeFinalLayer();
    state.selectedDemoId = null;
    state.selectedField = null;
    state.geojson = null;
    state.drawnAreaHa = null;
    state.drawnVertices = 0;
    updateDemoStyles();
    resetResult();
    els.fieldCard.classList.remove("is-empty");
    els.fieldName.textContent = row.id;
    els.fieldMeta.textContent =
      (row.crop_label || row.crop || "") +
      " · " +
      fmtDate(row.sowing_date) +
      " · " +
      fmtNum(row.lat, 4) +
      ", " +
      fmtNum(row.lon, 4);
    els.fieldMeta.title = els.fieldMeta.textContent;
    els.fieldHonesty.textContent = t("portfolio.rowNote");
    els.fieldHonesty.classList.remove("hidden");
    setFieldSource(row.source || "synthetic");
    hideFieldError();
    updateVerifyState();
    if (state.map && typeof L !== "undefined") {
      state.map.setView([row.lat, row.lon], 14);
      if (state.portfolioMarker) state.map.removeLayer(state.portfolioMarker);
      state.portfolioMarker = L.circleMarker([row.lat, row.lon], {
        radius: 8,
        color: "#ffffff",
        weight: 2,
        fillColor: "#0f5a45",
        fillOpacity: 1,
        interactive: true,
      }).addTo(state.map);
      state.portfolioMarker.bindTooltip(row.id, {
        direction: "top",
        className: "portfolio-label",
        permanent: true,
        opacity: 1,
      });
      window.setTimeout(() => state.map.invalidateSize(), 70);
    }
    showToast(t("portfolio.rowHint"));
  }

  function removePortfolioMarker() {
    if (state.portfolioMarker && state.map) {
      state.map.removeLayer(state.portfolioMarker);
    }
    state.portfolioMarker = null;
  }

  function csvCell(value) {
    const text = String(value == null ? "" : value);
    if (/[",\r\n;]/.test(text)) {
      return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
  }

  function exportPortfolioCsv() {
    const rows = state.portfolioView || [];
    if (!rows.length) {
      showToast(t("portfolio.empty"));
      return;
    }
    const header = [
      "id",
      "crop",
      "sowing_date",
      "lat",
      "lon",
      "score",
      "verdict",
      "verdict_label",
      "reason_code",
      "recommendation",
      "ml_predicted",
      "ml_probability",
      "source",
    ];
    const lines = [header.join(",")];
    rows.forEach((row) => {
      lines.push(
        [
          row.id,
          row.crop,
          row.sowing_date,
          row.lat,
          row.lon,
          row.score,
          row.verdict,
          row.verdict_label,
          row.reason_code,
          row.recommendation,
          row.ml_predicted,
          row.ml_probability,
          row.source,
        ]
          .map(csvCell)
          .join(",")
      );
    });
    const blob = new Blob(["\ufeff" + lines.join("\r\n")], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    link.href = url;
    link.download = "agro-portfolio-" + stamp + ".csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ======================================================================
     Effect calculator
     ====================================================================== */

  function updateEffect() {
    if (!els.effectSavings) return;
    const visit = Math.max(0, parseNumber(els.effectVisit.value, 0));
    const remote = Math.max(0, parseNumber(els.effectRemote.value, 0));
    const fields = Math.max(0, parseNumber(els.effectFields.value, 0));
    const perField = Math.max(0, visit - remote);
    const savings = perField * fields;
    els.effectSavings.textContent = fmtInt(Math.round(savings)) + " ₸";
    els.effectFormula.textContent =
      "(" +
      fmtInt(visit) +
      " − " +
      fmtInt(remote) +
      ") × " +
      fmtInt(fields);
  }

  /* ======================================================================
     Print report
     ====================================================================== */

  function printReport() {
    if (!state.lastResult) {
      showToast(t("result.empty"));
      return;
    }
    const result = state.lastResult;
    const field = result.field || {};
    const meta = result.meta || {};
    const features = featureRows(result);
    const synthetic = result.source === "synthetic";
    const probability = result.ml ? fmtPercent(result.ml.probability) : "—";
    const predicted = (result.ml && result.ml.predicted_label) || "—";
    const generated = new Date().toLocaleString(locale(), {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
    const reasons =
      (result.explanation && result.explanation.reasons) || [];
    const top3 = result.top3 || [];
    const chartImage =
      state.chart && typeof state.chart.toBase64Image === "function"
        ? state.chart.toBase64Image("image/png", 1)
        : "";
    const sourceNote = result.data_source_note || "";
    const disclaimer =
      meta[state.lang === "kz" ? "disclaimer_kz" : "disclaimer_ru"] ||
      t("print.disclaimer");

    const infoRows = [
      [t("print.field"), field.name || "—"],
      [t("print.crop"), field.crop_declared_label || "—"],
      [t("print.sowing"), fmtDate(field.sowing_date)],
      [
        t("print.area"),
        field.area_ha == null ? "—" : fmtNum(field.area_ha, 1) + " " + t("field.area"),
      ],
    ];

    els.printReport.innerHTML =
      '<div class="print-head">' +
      '<div><div class="print-brand">' +
      esc(t("brand")) +
      '</div><div class="print-title">' +
      esc(t("print.title")) +
      "</div></div>" +
      '<div class="print-generated">' +
      esc(t("print.generated")) +
      ": " +
      esc(generated) +
      "</div></div>" +
      '<div class="print-summary"><div><div class="print-verdict">' +
      esc(result.verdict_label || "—") +
      '</div><div class="print-source' +
      (synthetic ? " is-synthetic" : "") +
      '">' +
      esc(t("print.source")) +
      ": " +
      esc(result.source_label || "—") +
      "</div></div>" +
      '<div class="print-scores"><div><span>' +
      esc(t("print.score")) +
      "</span><strong>" +
      esc(String(result.score == null ? "—" : result.score)) +
      " / 100</strong></div><div><span>" +
      esc(t("print.confidence")) +
      "</span><strong>" +
      esc(probability) +
      "</strong><span>" +
      esc(predicted) +
      "</span></div></div></div>" +
      '<div class="print-grid">' +
      infoRows
        .map(
          (row) =>
            '<div class="print-row"><dt>' +
            esc(row[0]) +
            "</dt><dd>" +
            esc(row[1]) +
            "</dd></div>"
        )
        .join("") +
      "</div>" +
      '<div class="print-section"><h2>' +
      esc(t("print.explanation")) +
      "</h2><p>" +
      esc((result.explanation && result.explanation.summary) || "") +
      "</p></div>" +
      '<div class="print-section"><h2>' +
      esc(t("print.reasons")) +
      "</h2><ul>" +
      reasons.map((reason) => "<li>" + esc(reason) + "</li>").join("") +
      "</ul></div>" +
      '<div class="print-section"><h2>' +
      esc(t("print.features")) +
      '</h2><div class="print-grid">' +
      features
        .map(
          (item) =>
            '<div class="print-row"><dt>' +
            esc(item.label) +
            "</dt><dd>" +
            esc(item.value) +
            "</dd></div>"
        )
        .join("") +
      "</div></div>" +
      (chartImage
        ? '<div class="print-section"><h2>' +
          esc(t("print.chart")) +
          '</h2><img class="print-chart" src="' +
          chartImage +
          '" alt="' +
          esc(t("result.chart")) +
          '"></div>'
        : "") +
      '<div class="print-section"><h2>' +
      esc(t("print.recommendation")) +
      "</h2><p>" +
      esc((result.explanation && result.explanation.recommendation) || "") +
      "</p></div>" +
      '<div class="print-section"><h2>' +
      esc(t("top3.title")) +
      "</h2><p>" +
      top3
        .map((item) => {
          const labelKey = state.lang === "kz" ? "label_kz" : "label_ru";
          return (
            esc(item[labelKey] || item.crop || "") +
            " — " +
            esc(fmtPercent(item.probability))
          );
        })
        .join(" · ") +
      "</p></div>" +
      '<div class="print-note"><strong>' +
      esc(t("print.source")) +
      ":</strong> " +
      esc(result.source_label || "") +
      ". " +
      esc(sourceNote) +
      " " +
      esc(t("print.modelNote")) +
      " " +
      esc(disclaimer) +
      "</div>";

    els.printReport.setAttribute("aria-hidden", "false");
    window.print();
    window.setTimeout(() => {
      els.printReport.setAttribute("aria-hidden", "true");
    }, 500);
  }
})();
