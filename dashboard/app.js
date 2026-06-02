/**
 * DrillMind RTOC Dashboard v0.4 — Application Logic
 * ==================================================
 * Real-time drilling operations dashboard.
 *
 * Wire-up:
 *   * Live telemetry  →  WebSocket /ws/stream (configurable speed 1×–1000×)
 *   * Live alerts     →  WebSocket /ws/alerts (snapshot + create/ack/resolve)
 *   * Time-depth      →  GET /api/data/timedepth
 *   * Formation tops  →  GET /api/well/formations (derived from real LWD gamma)
 *   * Copilot         →  POST /api/copilot/query  (multi-agent or tools mode)
 */

const API_BASE = '';     // same origin
const WS_BASE = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`;
const MAX_CHART_POINTS = 500;
const SPARKLINE_SIZE = 100;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let charts = {};
let activeAlerts = [];
let allEvents = [];
let currentEventFilter = 'all';
let currentAlertFilter = 'all';
let replaySocket = null;
let alertsSocket = null;
let currentReplaySpeed = 10;
let isPaused = false;

// ---- Sparkline ring buffers ----
const SPARK_CHANNELS = ['Depth','WOB','SPP','Hookload','Torque','RPM','Flow','MSE','Anomaly'];
const sparkBuffers = {};
const sparkMinMax = {};
SPARK_CHANNELS.forEach(ch => {
    sparkBuffers[ch] = new Float32Array(SPARKLINE_SIZE);
    sparkMinMax[ch] = { min: Infinity, max: -Infinity, idx: 0 };
});

function pushSparkValue(channel, val) {
    if (val == null || !isFinite(val)) return;
    const buf = sparkBuffers[channel];
    const mm = sparkMinMax[channel];
    const idx = mm.idx % SPARKLINE_SIZE;
    buf[idx] = val;
    mm.idx++;
    if (mm.idx % SPARKLINE_SIZE === 0 || val < mm.min || val > mm.max) {
        let lo = Infinity, hi = -Infinity;
        const len = Math.min(mm.idx, SPARKLINE_SIZE);
        for (let i = 0; i < len; i++) {
            if (buf[i] < lo) lo = buf[i];
            if (buf[i] > hi) hi = buf[i];
        }
        mm.min = lo;
        mm.max = hi;
    }
}

const sparkCanvasCache = {};
function renderSparkline(canvasId, channel, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (!sparkCanvasCache[canvasId]) {
        sparkCanvasCache[canvasId] = canvas.getContext('2d', { alpha: true });
        canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
        canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
    }
    const ctx = sparkCanvasCache[canvasId];
    const w = canvas.width;
    const h = canvas.height;
    const buf = sparkBuffers[channel];
    const mm = sparkMinMax[channel];
    const len = Math.min(mm.idx, SPARKLINE_SIZE);
    if (len < 2) return;

    ctx.clearRect(0, 0, w, h);
    const range = mm.max - mm.min || 1;
    const startIdx = mm.idx >= SPARKLINE_SIZE ? mm.idx % SPARKLINE_SIZE : 0;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
    for (let i = 0; i < len; i++) {
        const dataIdx = (startIdx + i) % SPARKLINE_SIZE;
        const x = (i / (len - 1)) * w;
        const y = h - ((buf[dataIdx] - mm.min) / range) * (h * 0.85);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

function renderAllSparklines() {
    renderSparkline('sparkDepth', 'Depth', '#60a5fa');
    renderSparkline('sparkWOB', 'WOB', '#a78bfa');
    renderSparkline('sparkSPP', 'SPP', '#818cf8');
    renderSparkline('sparkHookload', 'Hookload', '#67e8f9');
    renderSparkline('sparkTorque', 'Torque', '#fcd34d');
    renderSparkline('sparkRPM', 'RPM', '#6ee7b7');
    renderSparkline('sparkFlow', 'Flow', '#22d3ee');
    renderSparkline('sparkMSE', 'MSE', '#c084fc');
    renderSparkline('sparkAnomaly', 'Anomaly', '#f87171');
}

// ---------------------------------------------------------------------------
// MSE computation (Teale, 1965)
// ---------------------------------------------------------------------------
const BIT_DIAMETER_INCHES = 12.25;
const BIT_DIAMETER_M = BIT_DIAMETER_INCHES * 0.0254;
const BIT_AREA_M2 = Math.PI / 4 * BIT_DIAMETER_M * BIT_DIAMETER_M;

function computeMSE(wob, torque_kNm, rpm, rop_mh) {
    if (!rop_mh || rop_mh <= 0 || !rpm || rpm <= 0) return null;
    const torque_Nm = torque_kNm * 1000;
    const rop_ms = rop_mh / 3600;
    const rpm_rps = rpm / 60;
    const rotary = (2 * Math.PI * torque_Nm * rpm_rps) / (BIT_AREA_M2 * rop_ms);
    const thrust = (wob || 0) / BIT_AREA_M2;
    const mse_mpa = (rotary + thrust) / 1e6;
    return mse_mpa > 0 && mse_mpa < 1e5 ? mse_mpa : null;
}

// ---------------------------------------------------------------------------
// Chart helpers
// ---------------------------------------------------------------------------
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
        legend: { display: false },
        tooltip: {
            backgroundColor: 'rgba(18, 22, 32, 0.95)',
            borderColor: 'rgba(99, 102, 241, 0.3)',
            borderWidth: 1,
            titleFont: { family: "'Inter', sans-serif", size: 11 },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            padding: 10,
            cornerRadius: 6,
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(42, 48, 66, 0.5)', lineWidth: 0.5 },
            ticks: {
                color: '#64748b',
                font: { family: "'JetBrains Mono', monospace", size: 9 },
                maxTicksLimit: 8,
                maxRotation: 0,
            },
        },
        y: {
            grid: { color: 'rgba(42, 48, 66, 0.5)', lineWidth: 0.5 },
            ticks: {
                color: '#64748b',
                font: { family: "'JetBrains Mono', monospace", size: 10 },
                maxTicksLimit: 5,
            },
        },
    },
};

function createChart(canvasId, color, label) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, color.replace(')', ', 0.25)').replace('rgb', 'rgba'));
    gradient.addColorStop(1, color.replace(')', ', 0.01)').replace('rgb', 'rgba'));
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label, data: [],
                borderColor: color, backgroundColor: gradient,
                fill: true, borderWidth: 1.4,
                pointRadius: 0, tension: 0.3,
            }],
        },
        options: { ...CHART_DEFAULTS },
    });
}

function createAnomalyChart() {
    const ctx = document.getElementById('chartAnomaly').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Anomaly Score',
                    data: [],
                    borderColor: '#f87171',
                    backgroundColor: 'rgba(248, 113, 113, 0.06)',
                    fill: true,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: 'Threshold',
                    data: [],
                    borderColor: 'rgba(251, 191, 36, 0.55)',
                    borderWidth: 1,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: { ...CHART_DEFAULTS },
    });
}

function createTimeDepthChart() {
    const ctx = document.getElementById('chartTimeDepth').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Bit Depth (m MD)', data: [], borderColor: '#67e8f9', borderWidth: 1.4, pointRadius: 0, tension: 0.05 },
            { label: 'Hole Depth (m MD)', data: [], borderColor: '#a78bfa', borderWidth: 1.2, pointRadius: 0, tension: 0.05 },
            { label: 'TVD (m)', data: [], borderColor: '#fcd34d', borderWidth: 1.0, pointRadius: 0, borderDash: [4, 3] },
        ]},
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: {
                    display: true, position: 'top', align: 'end',
                    labels: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 10 }, boxWidth: 12, padding: 12 },
                },
            },
            scales: {
                x: CHART_DEFAULTS.scales.x,
                y: { ...CHART_DEFAULTS.scales.y, reverse: true, title: { display: true, text: 'Depth (m)', color: '#94a3b8' } },
            },
        },
    });
}

function initCharts() {
    charts.spp = createChart('chartSPP', 'rgb(129, 140, 248)', 'SPP');
    charts.hookload = createChart('chartHookload', 'rgb(103, 232, 249)', 'Hookload');
    charts.torque = createChart('chartTorque', 'rgb(252, 211, 77)', 'Torque');
    charts.pit = createChart('chartPit', 'rgb(110, 231, 183)', 'Pit Volume');
    charts.anomaly = createAnomalyChart();
    charts.timedepth = createTimeDepthChart();
}

// ---------------------------------------------------------------------------
// Update functions
// ---------------------------------------------------------------------------
function updateCharts(records) {
    if (!records || !records.length) return;
    const labels = records.map(d => (d.timestamp || '').substring(11, 19));
    const set = (chart, vals) => {
        chart.data.labels = labels;
        chart.data.datasets[0].data = vals;
        chart.update('none');
    };
    set(charts.spp, records.map(d => d.spp));
    set(charts.hookload, records.map(d => d.weight_on_hook));
    set(charts.torque, records.map(d => d.torque_averaged));
    set(charts.pit, records.map(d => d.pit_volume_active));
}

function updateAnomalyChart(scores, threshold = 0.3) {
    const labels = scores.map(s => (s.timestamp || '').substring(11, 19));
    const values = scores.map(s => s.combined);
    charts.anomaly.data.labels = labels;
    charts.anomaly.data.datasets[0].data = values;
    charts.anomaly.data.datasets[1].data = labels.map(() => threshold);
    charts.anomaly.update('none');
}

function updateTimeDepth(points) {
    if (!charts.timedepth) return;
    charts.timedepth.data.labels = points.map(p => (p.timestamp || '').substring(0, 10));
    charts.timedepth.data.datasets[0].data = points.map(p => p.bit_depth);
    charts.timedepth.data.datasets[1].data = points.map(p => p.hole_depth_md);
    charts.timedepth.data.datasets[2].data = points.map(p => p.tvd);
    charts.timedepth.update('none');
}

let _liveSeeded = false;
let _lastLiveRender = 0;
const LIVE_MAX_POINTS = 240;

function pushLiveCharts(record) {
    if (!record || record.timestamp == null) return;
    const liveCharts = [charts.spp, charts.hookload, charts.torque, charts.pit, charts.anomaly];
    if (!_liveSeeded) {
        liveCharts.forEach(c => {
            if (!c) return;
            c.data.labels = [];
            c.data.datasets.forEach(ds => { ds.data = []; });
        });
        _liveSeeded = true;
    }
    const label = String(record.timestamp).substring(11, 19);
    const trim = (c) => {
        if (c.data.labels.length > LIVE_MAX_POINTS) {
            c.data.labels.shift();
            c.data.datasets.forEach(ds => ds.data.shift());
        }
    };
    const pushOne = (c, value) => {
        if (!c) return;
        c.data.labels.push(label);
        c.data.datasets[0].data.push(value == null ? null : value);
        trim(c);
    };
    pushOne(charts.spp, record.spp);
    pushOne(charts.hookload, record.weight_on_hook);
    pushOne(charts.torque, record.torque_averaged);
    pushOne(charts.pit, record.pit_volume_active);
    if (charts.anomaly) {
        charts.anomaly.data.labels.push(label);
        charts.anomaly.data.datasets[0].data.push(record.anomaly_score == null ? null : record.anomaly_score);
        if (charts.anomaly.data.datasets[1]) charts.anomaly.data.datasets[1].data.push(0.3);
        trim(charts.anomaly);
    }
    const now = Date.now();
    if (now - _lastLiveRender >= 120) {
        _lastLiveRender = now;
        liveCharts.forEach(c => { if (c) c.update('none'); });
    }
}

// ---- KPIs ----
let _hookloadBaseline = null;
const ON_BOTTOM_THRESHOLD = 5.0;
let _sparkRafPending = false;

function setValue(id, val, decimals = 1) {
    const el = document.getElementById(id);
    if (el && val != null && !isNaN(val)) el.textContent = Number(val).toFixed(decimals);
}
function setMinMax(minId, maxId, channel) {
    const mm = sparkMinMax[channel];
    if (mm && mm.idx > 0) {
        const minEl = document.getElementById(minId);
        const maxEl = document.getElementById(maxId);
        if (minEl) minEl.textContent = `▼ ${mm.min.toFixed(1)}`;
        if (maxEl) maxEl.textContent = `▲ ${mm.max.toFixed(1)}`;
    }
}

function updateKPIs(record) {
    if (!record) return;
    if (record.bit_depth != null) pushSparkValue('Depth', record.bit_depth);
    const wobVal = (record.wob_avg != null && record.wob_avg < 0) ? 0 : (record.wob_avg || 0);
    pushSparkValue('WOB', wobVal);
    if (record.spp != null) pushSparkValue('SPP', record.spp);
    if (record.weight_on_hook != null) pushSparkValue('Hookload', record.weight_on_hook);
    if (record.torque_averaged != null) pushSparkValue('Torque', record.torque_averaged);
    pushSparkValue('RPM', record.rpm_avg || 0);
    if (record.flow_pumps != null) pushSparkValue('Flow', record.flow_pumps);
    const rop = record.rop || record.rop_5ft_avg || 0;
    const mse = computeMSE(wobVal, record.torque_averaged || 0, record.rpm_avg || 0, rop);
    if (mse != null) pushSparkValue('MSE', mse);

    setValue('kpiDepthValue', record.bit_depth, 1);
    setValue('kpiROPValue', wobVal, 2);
    setValue('kpiSPPValue', record.spp, 0);
    setValue('kpiHookloadValue', record.weight_on_hook, 1);
    setValue('kpiTorqueValue', record.torque_averaged, 2);
    setValue('kpiRPMValue', record.rpm_avg, 0);
    setValue('kpiFlowValue', record.flow_pumps, 0);
    const mseEl = document.getElementById('kpiMSEValue');
    if (mseEl) mseEl.textContent = mse != null ? mse.toFixed(1) : '—';

    setMinMax('kpiDepthMin', 'kpiDepthMax', 'Depth');
    setMinMax('kpiWOBMin', 'kpiWOBMax', 'WOB');
    setMinMax('kpiSPPMin', 'kpiSPPMax', 'SPP');
    setMinMax('kpiHookMin', 'kpiHookMax', 'Hookload');
    setMinMax('kpiTorqueMin', 'kpiTorqueMax', 'Torque');
    setMinMax('kpiRPMMin', 'kpiRPMMax', 'RPM');
    setMinMax('kpiFlowMin', 'kpiFlowMax', 'Flow');
    setMinMax('kpiMSEMin', 'kpiMSEMax', 'MSE');

    const tsEl = document.getElementById('currentTimestamp');
    if (tsEl && record.timestamp) {
        tsEl.textContent = record.timestamp.substring(0, 19).replace('T', ' ');
    }

    const bitLabel = document.getElementById('bitDepthLabel');
    if (bitLabel && record.bit_depth != null) {
        bitLabel.textContent = `Bit: ${record.bit_depth.toFixed(0)} m`;
    }
    // Drive bit indicator y position 350..430 mapped to 0..4890 m
    const bitG = document.getElementById('bitIndicator');
    if (bitG && record.bit_depth != null) {
        const y = 60 + Math.min(370, Math.max(0, record.bit_depth / 4890 * 370));
        bitG.setAttribute('transform', `translate(120, ${y})`);
        const lbl = document.getElementById('bitDepthLabel');
        if (lbl) lbl.setAttribute('y', y + 4);
    }

    updateOnBottomStatus(record);

    if (!_sparkRafPending) {
        _sparkRafPending = true;
        requestAnimationFrame(() => {
            renderAllSparklines();
            _sparkRafPending = false;
        });
    }
}

function updateOnBottomStatus(record) {
    const hookload = record.weight_on_hook;
    const rpm = record.rpm_avg || 0;
    if (hookload != null && !isNaN(hookload)) {
        if (_hookloadBaseline === null) _hookloadBaseline = hookload;
        else _hookloadBaseline = Math.max(hookload, _hookloadBaseline * 0.999);
    }
    const isOnBottom = (
        _hookloadBaseline !== null && hookload != null &&
        hookload < (_hookloadBaseline - ON_BOTTOM_THRESHOLD) && rpm > 0
    );
    const card = document.getElementById('kpiOnBottom');
    const valueEl = document.getElementById('kpiOnBottomValue');
    if (valueEl) valueEl.textContent = isOnBottom ? 'ON BTM' : 'OFF BTM';
    if (card) card.classList.toggle('is-on-bottom', isOnBottom);
}

function updateAnomalyKPI(score) {
    const el = document.getElementById('kpiAnomalyScore');
    const card = document.getElementById('kpiAnomaly');
    if (el) el.textContent = score != null ? Number(score).toFixed(3) : '—';
    if (card) card.classList.toggle('alert', score > 0.3);
    if (score != null) pushSparkValue('Anomaly', score);
}

// ---------------------------------------------------------------------------
// Events list
// ---------------------------------------------------------------------------
function renderEvents(events, filter) {
    const container = document.getElementById('eventsList');
    if (!container) return;
    let filtered = events;
    if (filter && filter !== 'all') filtered = events.filter(e => e.severity === filter);
    if (filtered.length === 0) { container.innerHTML = '<div class="loading-events">No events found</div>'; return; }
    container.innerHTML = filtered.slice(0, 60).map(event => `
        <div class="event-item" data-severity="${event.severity}">
            <div class="event-severity ${event.severity}"></div>
            <div class="event-content">
                <div class="event-header">
                    <span class="event-type ${event.event_type}">${event.event_type.replace('_', ' ')}</span>
                    <span class="event-time">${event.timestamp.substring(0, 19).replace('T', ' ')}</span>
                    <span class="event-score">${Number(event.score).toFixed(3)}</span>
                </div>
                <div class="event-description">${event.description}</div>
                <div class="event-action">→ ${event.recommended_action}</div>
            </div>
        </div>
    `).join('');
}

function renderSummary(summary) {
    const container = document.getElementById('summaryContent');
    if (!container || !summary) return;
    const anomalyPct = (summary.anomaly_rate * 100).toFixed(1);
    const normalPct = (100 - summary.anomaly_rate * 100).toFixed(1);
    const typeColors = {
        kick: '#ef4444', lost_circulation: '#f97316', stuck_pipe: '#f97316',
        bit_dysfunction: '#f59e0b', washout: '#f59e0b', connection_gas: '#22c55e',
        unknown: '#64748b',
    };
    const typeBreakdown = summary.by_type ? Object.entries(summary.by_type).map(([type, count]) => `
        <div class="type-item">
            <div class="type-dot" style="background: ${typeColors[type] || '#64748b'}"></div>
            <span class="type-name">${type.replace('_', ' ')}</span>
            <span class="type-count">${count}</span>
        </div>`).join('') : '';
    container.innerHTML = `
        <div class="summary-stat"><span class="summary-stat-label">Total Samples</span>
            <span class="summary-stat-value">${(summary.total_samples || 0).toLocaleString()}</span></div>
        <div class="summary-stat"><span class="summary-stat-label">Anomalous Samples</span>
            <span class="summary-stat-value" style="color:#ef4444">${(summary.total_anomalous_samples || 0).toLocaleString()}</span></div>
        <div class="summary-stat"><span class="summary-stat-label">Anomaly Rate</span>
            <span class="summary-stat-value">${anomalyPct}%</span></div>
        <div class="summary-stat"><span class="summary-stat-label">Total Events</span>
            <span class="summary-stat-value">${summary.total_events || 0}</span></div>
        <div class="summary-bar">
            <div class="summary-bar-header"><span>Normal ${normalPct}%</span><span>Anomalous ${anomalyPct}%</span></div>
            <div class="bar-container">
                <div class="bar-segment normal" style="width:${normalPct}%"></div>
                <div class="bar-segment anomalous" style="width:${anomalyPct}%"></div>
            </div>
        </div>
        <div class="type-breakdown">${typeBreakdown}</div>`;
}

// ---------------------------------------------------------------------------
// Persistent Alerts (WebSocket + REST)
// ---------------------------------------------------------------------------
function renderActiveAlerts(filter) {
    const container = document.getElementById('alertsList');
    if (!container) return;
    let list = activeAlerts;
    if (filter && filter !== 'all') list = list.filter(a => a.severity === filter);
    if (list.length === 0) {
        container.innerHTML = '<div class="loading-events">No active alerts.</div>';
        updateAlarmCounts();
        return;
    }
    container.innerHTML = list.slice(0, 50).map(a => `
        <div class="alert-row severity-${a.severity}" data-id="${a.id}">
            <div class="event-severity"></div>
            <div class="alert-body">
                <div class="alert-head">
                    <span class="alert-type event-type ${a.event_type}">${(a.event_type || '').replace('_', ' ')}</span>
                    <span class="alert-time">${a.timestamp ? a.timestamp.substring(0, 19).replace('T', ' ') : ''}</span>
                </div>
                <div class="alert-desc">${escapeHtml(a.description || '')}</div>
                <div class="alert-action">→ ${escapeHtml(a.recommended_action || '')}</div>
            </div>
            <div class="alert-actions">
                <button class="alert-btn ack"     data-action="ack"     data-id="${a.id}">Ack</button>
                <button class="alert-btn resolve" data-action="resolve" data-id="${a.id}">Resolve</button>
            </div>
        </div>`).join('');
    updateAlarmCounts();
}

function updateAlarmCounts() {
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    activeAlerts.forEach(a => { if (counts[a.severity] != null) counts[a.severity]++; });
    document.getElementById('alarmCountCritical').textContent = counts.critical;
    document.getElementById('alarmCountHigh').textContent = counts.high;
    document.getElementById('alarmCountMedium').textContent = counts.medium;
    document.getElementById('alarmCountLow').textContent = counts.low;
    document.querySelector('.alarm-critical').classList.toggle('has-alerts', counts.critical > 0);
    document.querySelector('.alarm-high').classList.toggle('has-alerts', counts.high > 0);
}

function pushTickerItem(alert, evtType = 'created') {
    const ticker = document.getElementById('tickerContent');
    if (!ticker) return;
    const sev = alert.severity || 'info';
    const sevClass = sev === 'critical' ? 'ticker-critical'
                   : sev === 'high'     ? 'ticker-high'
                   : sev === 'medium'   ? 'ticker-medium'
                   : 'ticker-info';
    const desc = escapeHtml(alert.description || alert.event_type || 'Alert');
    const tag = evtType === 'resolved' ? 'RESOLVED'
              : evtType === 'acknowledged' ? 'ACK'
              : sev.toUpperCase();
    const item = document.createElement('span');
    item.className = `ticker-item ${sevClass}`;
    item.innerHTML = `[${tag}] ${desc}`;
    ticker.appendChild(item);
    // Cap ticker length
    while (ticker.children.length > 30) ticker.removeChild(ticker.firstChild);
}

async function loadActiveAlerts() {
    try {
        const res = await fetch(`${API_BASE}/api/alerts/active?limit=200`);
        const data = await res.json();
        activeAlerts = data.items || [];
        renderActiveAlerts(currentAlertFilter);
    } catch (e) { console.error('loadActiveAlerts failed', e); }
}

function setupAlertSocket() {
    try {
        alertsSocket = new WebSocket(`${WS_BASE}/ws/alerts`);
    } catch (e) { console.error('alerts WS error', e); return; }
    alertsSocket.addEventListener('message', (ev) => {
        let payload;
        try { payload = JSON.parse(ev.data); } catch { return; }
        if (payload.type === 'snapshot' && Array.isArray(payload.items)) {
            activeAlerts = payload.items;
            renderActiveAlerts(currentAlertFilter);
            return;
        }
        if (payload.type === 'alert' && payload.alert) {
            const a = payload.alert;
            if (payload.event === 'created') {
                activeAlerts = [a, ...activeAlerts.filter(x => x.id !== a.id)];
                pushTickerItem(a, 'created');
            } else if (payload.event === 'resolved' || payload.event === 'suppressed') {
                activeAlerts = activeAlerts.filter(x => x.id !== a.id);
                pushTickerItem(a, payload.event);
            } else if (payload.event === 'acknowledged') {
                activeAlerts = activeAlerts.map(x => x.id === a.id ? a : x);
                pushTickerItem(a, 'acknowledged');
            }
            renderActiveAlerts(currentAlertFilter);
        }
    });
    alertsSocket.addEventListener('close', () => { setTimeout(setupAlertSocket, 4000); });
}

function setupAlertActions() {
    const container = document.getElementById('alertsList');
    if (!container) return;
    container.addEventListener('click', async (ev) => {
        const btn = ev.target.closest('.alert-btn');
        if (!btn) return;
        const id = btn.dataset.id;
        const action = btn.dataset.action;
        try {
            await fetch(`${API_BASE}/api/alerts/${id}/${action === 'ack' ? 'acknowledge' : 'resolve'}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ actor: 'rtoc-user' }),
            });
        } catch (e) { console.error(e); }
    });
    document.querySelectorAll('.alerts-panel .filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.alerts-panel .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentAlertFilter = btn.dataset.filter;
            renderActiveAlerts(currentAlertFilter);
        });
    });
}

// ---------------------------------------------------------------------------
// REST loaders
// ---------------------------------------------------------------------------
async function fetchJSON(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`Fetch error for ${endpoint}:`, err);
        return null;
    }
}

async function loadWellInfo() {
    const info = await fetchJSON('/api/well/info');
    if (!info) return;
    document.getElementById('wellName').textContent = info.well || '—';
    document.getElementById('fieldName').textContent = info.field || '—';
    document.getElementById('operatorName').textContent = info.operator || '—';
    const sn = document.getElementById('schematicWellName');
    if (sn && info.well) sn.textContent = info.well;
    if (info.max_replay_speed) {
        document.getElementById('replaySpeed').max = info.max_replay_speed;
    }
    setConnected(true);
}

async function loadTimeseries(start = 0, limit = 500) {
    const data = await fetchJSON(`/api/data/timeseries?start=${start}&limit=${limit}`);
    if (data && data.data) {
        updateCharts(data.data);
        if (data.data.length > 0) updateKPIs(data.data[data.data.length - 1]);
    }
}

async function loadAnomalyScores(start = 0, limit = 500) {
    const data = await fetchJSON(`/api/anomalies/scores?start=${start}&limit=${limit}`);
    if (data && data.scores) {
        updateAnomalyChart(data.scores);
        if (data.scores.length) updateAnomalyKPI(data.scores[data.scores.length - 1].combined);
    }
}

async function loadEvents() {
    const data = await fetchJSON('/api/anomalies/events?limit=200');
    if (data && data.events) {
        allEvents = data.events;
        renderEvents(allEvents, currentEventFilter);
    }
}

async function loadSummary() {
    const data = await fetchJSON('/api/anomalies/summary');
    if (data) renderSummary(data);
}

async function loadRigState() {
    const data = await fetchJSON('/api/rig/summary');
    if (data && data.states) {
        const topState = Object.entries(data.states).sort((a, b) => b[1].count - a[1].count)[0];
        if (topState) {
            document.getElementById('kpiRigStateValue').textContent = topState[0].replace('_', ' ');
            document.getElementById('kpiRigState').setAttribute('data-state', topState[0]);
        }
    }
}

async function loadTimeDepth() {
    const data = await fetchJSON('/api/data/timedepth?start=0&limit=20000');
    if (data && data.data) updateTimeDepth(data.data);
}

async function loadFormations() {
    const data = await fetchJSON('/api/well/formations');
    if (!data || !data.tops || !data.tops.length) return;
    const g = document.getElementById('formationLabels');
    if (!g) return;
    g.innerHTML = '';
    // Map depths 0..4890 m onto y 60..430 in the SVG
    data.tops.slice(0, 5).forEach((top, i) => {
        const y = 60 + Math.min(370, Math.max(0, top.depth_md / 4890 * 370));
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', '35');
        rect.setAttribute('y', y - 8);
        rect.setAttribute('width', '52');
        rect.setAttribute('height', '14');
        rect.setAttribute('fill', 'rgba(74,222,128,0.06)');
        rect.setAttribute('stroke', 'rgba(74,222,128,0.25)');
        rect.setAttribute('stroke-width', '0.5');
        rect.setAttribute('rx', '3');
        g.appendChild(rect);
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('x', '42');
        t.setAttribute('y', y + 2);
        t.setAttribute('fill', '#4ade80');
        t.setAttribute('font-size', '7');
        t.setAttribute('font-family', 'Inter');
        t.setAttribute('font-weight', '600');
        t.textContent = `Fm ${i + 1}: ${top.depth_md} m`;
        g.appendChild(t);
    });
}

function setConnected(connected) {
    const el = document.getElementById('connectionStatus');
    if (!el) return;
    el.className = 'status-indicator ' + (connected ? 'connected' : 'error');
    el.querySelector('span').textContent = connected ? 'Connected' : 'Disconnected';
}

// ---------------------------------------------------------------------------
// Live WebSocket stream (replay)
// ---------------------------------------------------------------------------
function setupStreamSocket() {
    try {
        replaySocket = new WebSocket(`${WS_BASE}/ws/stream`);
    } catch (e) { console.error('stream WS error', e); return; }
    replaySocket.addEventListener('open', () => {
        setConnected(true);
        try {
            replaySocket.send(JSON.stringify({ action: 'set_speed', speed: currentReplaySpeed }));
        } catch {}
    });
    replaySocket.addEventListener('message', (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type !== 'data') return;
        updateKPIs(msg);
        pushLiveCharts(msg);
        if (msg.anomaly_score != null) updateAnomalyKPI(msg.anomaly_score);
    });
    replaySocket.addEventListener('close', () => {
        setConnected(false);
        setTimeout(setupStreamSocket, 4000);
    });
    replaySocket.addEventListener('error', () => setConnected(false));
}

function setReplaySpeed(speed) {
    currentReplaySpeed = Math.max(1, Math.min(1000, parseInt(speed) || 1));
    document.getElementById('replaySpeedValue').textContent = currentReplaySpeed + '×';
    if (replaySocket && replaySocket.readyState === WebSocket.OPEN) {
        try { replaySocket.send(JSON.stringify({ action: 'set_speed', speed: currentReplaySpeed })); } catch {}
    }
}

function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('replayPause');
    btn.textContent = isPaused ? '▶' : '❚❚';
    btn.classList.toggle('paused', isPaused);
    if (replaySocket && replaySocket.readyState === WebSocket.OPEN) {
        try {
            replaySocket.send(JSON.stringify({ action: isPaused ? 'pause' : 'resume' }));
        } catch {}
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------
function setupFilters() {
    document.querySelectorAll('.events-panel .filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.events-panel .filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentEventFilter = btn.dataset.eventfilter || btn.dataset.filter || 'all';
            renderEvents(allEvents, currentEventFilter);
        });
    });
}

// ---------------------------------------------------------------------------
// Theme / copilot toggle
// ---------------------------------------------------------------------------
function setupThemeToggle() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const saved = localStorage.getItem('drillmind-theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
        btn.textContent = saved === 'light' ? '☀' : '☾';
    }
    btn.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme');
        const next = cur === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('drillmind-theme', next);
        btn.textContent = next === 'light' ? '☀' : '☾';
    });
}

function setupCopilotToggle() {
    const toggleBtn = document.getElementById('copilotToggle');
    const panel = document.getElementById('copilotSection');
    if (!toggleBtn || !panel) return;
    toggleBtn.addEventListener('click', () => {
        const isOpen = !panel.classList.contains('collapsed');
        panel.classList.toggle('collapsed', isOpen);
        document.body.classList.toggle('copilot-open', !isOpen);
        toggleBtn.textContent = isOpen ? 'AI' : '✕';
    });
}

// ---------------------------------------------------------------------------
// Copilot chat
// ---------------------------------------------------------------------------
function simpleMarkdown(text) {
    return text
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/^#### (.*)$/gm, '<h4>$1</h4>')
        .replace(/^### (.*)$/gm, '<h3>$1</h3>')
        .replace(/^## (.*)$/gm, '<h2>$1</h2>')
        .replace(/^- (.*)$/gm, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
        .replace(/✅/g, '<span style="color:#22c55e">✅</span>')
        .replace(/⚠/g,  '<span style="color:#fbbf24">⚠</span>')
        .replace(/🚨/g, '<span style="color:#ef4444">🚨</span>')
        .replace(/\n/g, '<br>');
}

function addMessage(role, content, meta = null) {
    const container = document.getElementById('copilotMessages');
    const welcome = container.querySelector('.copilot-welcome');
    if (welcome) welcome.remove();
    const msgDiv = document.createElement('div');
    msgDiv.className = `copilot-msg ${role}`;
    let html = `<div class="msg-bubble">${role === 'assistant' ? simpleMarkdown(content) : escapeHtml(content)}</div>`;
    if (meta) html += `<div class="msg-meta">${meta}</div>`;
    msgDiv.innerHTML = html;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const container = document.getElementById('copilotMessages');
    const typing = document.createElement('div');
    typing.className = 'copilot-msg assistant';
    typing.id = 'typingIndicator';
    typing.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    container.appendChild(typing);
    container.scrollTop = container.scrollHeight;
}
function removeTyping() { const el = document.getElementById('typingIndicator'); if (el) el.remove(); }

async function sendCopilotQuery(question) {
    if (!question.trim()) return;
    const input = document.getElementById('copilotInput');
    const sendBtn = document.getElementById('copilotSend');
    const status = document.getElementById('copilotStatus');
    const mode = (document.querySelector('input[name="copilotMode"]:checked') || {}).value || 'multi';

    addMessage('user', question);
    input.value = '';
    sendBtn.disabled = true;
    status.textContent = 'Thinking…';
    status.className = 'copilot-status thinking';
    showTyping();

    try {
        const res = await fetch(`${API_BASE}/api/copilot/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, mode }),
        });
        removeTyping();
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const ctx = data.context_summary || {};
        let meta = `Intent: ${ctx.intent || 'general'}`;
        if (ctx.agents_run) meta += ` · Agents: ${ctx.agents_run.join(' → ')}`;
        if (ctx.tools_called && ctx.tools_called.length) {
            const unique = Array.from(new Set(ctx.tools_called));
            meta += ` · Tools: ${unique.join(', ')}`;
        }
        if (ctx.total_time_ms != null) meta += ` · ${ctx.total_time_ms} ms`;
        if (ctx.confidence != null) meta += ` · conf ${ctx.confidence}`;
        addMessage('assistant', data.answer || '(no answer)', meta);
    } catch (err) {
        removeTyping();
        addMessage('assistant', `Error: ${err.message}. Check that the API server is running.`);
    }

    sendBtn.disabled = false;
    status.textContent = 'Ready';
    status.className = 'copilot-status';
}

function setupCopilot() {
    const input = document.getElementById('copilotInput');
    const sendBtn = document.getElementById('copilotSend');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendCopilotQuery(input.value); }
        });
    }
    if (sendBtn) sendBtn.addEventListener('click', () => sendCopilotQuery(input.value));
    document.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const q = btn.dataset.query; if (q) sendCopilotQuery(q);
        });
    });
}

// ---------------------------------------------------------------------------
// Offset Wells
// ---------------------------------------------------------------------------
let offsetChart = null;
const WELL_COLORS = ['#818cf8', '#67e8f9', '#6ee7b7', '#fcd34d', '#f87171', '#c084fc', '#f472b6', '#2dd4bf'];

async function loadOffsetWells() {
    const data = await fetchJSON('/api/data/production?limit=5000');
    if (!data || !data.data || data.data.length === 0) return;
    const byWell = {};
    data.data.forEach(row => {
        const rawName = row.wellbore_code || row.wellbore || 'Unknown';
        const well = rawName.replace(/^NO 15\/9-/, '');
        if (!byWell[well]) byWell[well] = [];
        byWell[well].push(row);
    });
    const wellNames = Object.keys(byWell);
    const datasets = wellNames.map((well, i) => {
        const rows = byWell[well].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        let cumOil = 0;
        const points = rows.map((r, day) => {
            cumOil += (r.oil_vol || 0);
            return { x: day, y: cumOil };
        });
        return {
            label: well, data: points,
            borderColor: WELL_COLORS[i % WELL_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 1.4, pointRadius: 0, tension: 0.3,
        };
    });
    const ctx = document.getElementById('chartOffset');
    if (!ctx) return;
    if (offsetChart) offsetChart.destroy();
    offsetChart = new Chart(ctx, {
        type: 'line', data: { datasets },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                legend: { display: true, position: 'top',
                    labels: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 10 }, boxWidth: 10, padding: 8 } },
            },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Days', color: '#64748b' },
                    grid: { color: 'rgba(42, 48, 66, 0.3)' }, ticks: { color: '#64748b' } },
                y: { title: { display: true, text: 'Cum Oil (Sm³)', color: '#64748b' },
                    grid: { color: 'rgba(42, 48, 66, 0.3)' }, ticks: { color: '#64748b' } },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(str) {
    if (str == null) return '';
    return String(str).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

// ---------------------------------------------------------------------------
// Nav tabs (smooth-scroll to section)
// ---------------------------------------------------------------------------
function setupNavTabs() {
    const map = {
        overview:    '.row-tridash',
        performance: '.row-params',
        anomalies:   '.events-panel',
        history:     '.offset-panel',
        reports:     '.copilot-panel',
    };
    document.querySelectorAll('.nav-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const sel = map[btn.dataset.view];
            if (!sel) return;
            const el = document.querySelector(sel);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

// ---------------------------------------------------------------------------
// Replay control
// ---------------------------------------------------------------------------
function setupReplayControls() {
    const slider = document.getElementById('replaySpeed');
    const pauseBtn = document.getElementById('replayPause');
    if (slider) {
        slider.addEventListener('input', (e) => setReplaySpeed(e.target.value));
    }
    if (pauseBtn) pauseBtn.addEventListener('click', togglePause);
}

// ---------------------------------------------------------------------------
// Initial sparkline warm-up
// ---------------------------------------------------------------------------
async function initializeSparklineBuffers() {
    const data = await fetchJSON('/api/data/timeseries?start=0&limit=100');
    if (!data || !data.data) return;
    data.data.forEach(record => {
        if (record.bit_depth != null) pushSparkValue('Depth', record.bit_depth);
        const wob = (record.wob_avg != null && record.wob_avg < 0) ? 0 : (record.wob_avg || 0);
        pushSparkValue('WOB', wob);
        if (record.spp != null) pushSparkValue('SPP', record.spp);
        if (record.weight_on_hook != null) pushSparkValue('Hookload', record.weight_on_hook);
        if (record.torque_averaged != null) pushSparkValue('Torque', record.torque_averaged);
        pushSparkValue('RPM', record.rpm_avg || 0);
        if (record.flow_pumps != null) pushSparkValue('Flow', record.flow_pumps);
        const rop = record.rop || record.rop_5ft_avg || 0;
        const mse = computeMSE(wob, record.torque_averaged || 0, record.rpm_avg || 0, rop);
        if (mse != null) pushSparkValue('MSE', mse);
    });
    renderAllSparklines();
    if (data.data.length > 0) updateKPIs(data.data[data.data.length - 1]);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function setupChartZoom() {
    const chartFor = {
        chartSPP: () => charts.spp,
        chartHookload: () => charts.hookload,
        chartTorque: () => charts.torque,
        chartPit: () => charts.pit,
        chartAnomaly: () => charts.anomaly,
        chartTimeDepth: () => charts.timedepth,
        chartOffset: () => offsetChart,
    };

    let backdrop = document.getElementById('chartBackdrop');
    if (!backdrop) {
        backdrop = document.createElement('div');
        backdrop.id = 'chartBackdrop';
        backdrop.className = 'chart-backdrop';
        document.body.appendChild(backdrop);
    }

    let activePanel = null;
    let activeChart = null;

    function closeMax() {
        if (!activePanel) return;
        const c = activeChart;
        activePanel.classList.remove('maximized');
        backdrop.classList.remove('show');
        activePanel = null;
        activeChart = null;
        setTimeout(() => { if (c) { try { c.resize(); } catch (e) {} } }, 60);
    }

    function openMax(panel, getChart) {
        if (activePanel) closeMax();
        activePanel = panel;
        activeChart = getChart();
        panel.classList.add('maximized');
        backdrop.classList.add('show');
        setTimeout(() => { if (activeChart) { try { activeChart.resize(); } catch (e) {} } }, 60);
    }

    backdrop.addEventListener('click', closeMax);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMax(); });

    Object.keys(chartFor).forEach(canvasId => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const panel = canvas.closest('.chart-panel');
        if (!panel) return;
        panel.classList.add('zoomable');
        const btn = document.createElement('button');
        btn.className = 'chart-expand';
        btn.type = 'button';
        btn.title = 'Expand / collapse';
        btn.textContent = '⤢';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (panel.classList.contains('maximized')) closeMax();
            else openMax(panel, chartFor[canvasId]);
        });
        panel.appendChild(btn);
    });
}

async function init() {
    initCharts();
    setupChartZoom();
    setupFilters();
    setupCopilot();
    setupThemeToggle();
    setupCopilotToggle();
    setupReplayControls();
    setupNavTabs();
    setupAlertActions();

    await initializeSparklineBuffers();

    await Promise.all([
        loadWellInfo(),
        loadTimeseries(0, 500),
        loadAnomalyScores(0, 500),
        loadEvents(),
        loadSummary(),
        loadRigState(),
        loadOffsetWells(),
        loadActiveAlerts(),
        loadTimeDepth(),
        loadFormations(),
    ]);

    // Live websockets — start AFTER initial fetch so charts are pre-populated
    setupStreamSocket();
    setupAlertSocket();

    // Apply slider initial value
    setReplaySpeed(document.getElementById('replaySpeed').value);
}

document.addEventListener('DOMContentLoaded', init);
