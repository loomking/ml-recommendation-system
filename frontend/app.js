/**
 * RecoAI v2 — Frontend Application
 * Movie browsing, real-time recommendations, ratings, metrics dashboard, and scoring explanations.
 */

const API_BASE = '';
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_BASE = `${WS_PROTOCOL}//${window.location.host}`;

// ─── State ──────────────────────────────────────────────────────────────────

const state = {
    currentUser: null,
    users: [],
    recommendations: [],
    browsePage: 1,
    browseGenre: null,
    ws: null,
    wsReconnectTimer: null,
    modalMovie: null,
    lastTiming: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── API ────────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`API Error [${path}]:`, err);
        return null;
    }
}

// ─── Toast ──────────────────────────────────────────────────────────────────

function showToast(message, type = 'info') {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `${type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️'} ${message}`;
    container.appendChild(toast);
    setTimeout(() => { toast.classList.add('fade-out'); setTimeout(() => toast.remove(), 300); }, 3000);
}

// ─── Movie Cards ────────────────────────────────────────────────────────────

function createMovieCard(movie, showScore = false) {
    const card = document.createElement('div');
    card.className = 'movie-card';
    card.onclick = () => openMovieModal(movie.id);

    const ratingBadge = movie.avg_rating > 0
        ? `<div class="movie-rating-badge"><span class="star">★</span> ${movie.avg_rating.toFixed(1)}</div>`
        : '';

    let scoreBadge = '';
    if (showScore && movie.rec_score !== undefined) {
        const pct = Math.round(movie.rec_score * 100);
        scoreBadge = `<div class="movie-score-badge" onclick="event.stopPropagation(); openExplanation(${movie.id})" title="Click for scoring breakdown">${pct}% match</div>`;
    }

    card.innerHTML = `
        <div class="movie-poster" style="background: linear-gradient(135deg, ${movie.gradient_start}, ${movie.gradient_end})">
            ${ratingBadge}${scoreBadge}
            <div class="movie-poster-title">${movie.title}</div>
            <div class="movie-poster-year">${movie.year}</div>
        </div>
        <div class="movie-card-info">
            <div class="movie-card-title" title="${movie.title}">${movie.title}</div>
            <div class="movie-card-genres">${movie.genres.join(' · ')}</div>
        </div>`;
    return card;
}

function renderMovieRow(containerId, movies, showScore = false) {
    const container = $(`#${containerId}`);
    container.innerHTML = '';
    if (!movies || movies.length === 0) {
        container.innerHTML = '<div class="loading-container"><p>No movies found</p></div>';
        return;
    }
    movies.forEach(m => container.appendChild(createMovieCard(m, showScore)));
}

// ─── Sections ───────────────────────────────────────────────────────────────

async function loadTrending() {
    const movies = await api('/api/movies/trending?limit=20');
    if (movies) renderMovieRow('trending-row', movies);
}

async function loadTopRated() {
    const movies = await api('/api/movies/top-rated?limit=20');
    if (movies) renderMovieRow('top-rated-row', movies);
}

async function loadRecommendations() {
    if (!state.currentUser) {
        $('#recommendations-row').innerHTML = '<div class="loading-container"><p>Select a user to see personalized recommendations</p></div>';
        $('#rec-subtitle').textContent = 'Select a user from the top-right corner to get started';
        return;
    }
    $('#recommendations-row').innerHTML = '<div class="loading-container"><div class="loading-spinner"></div><p>Generating recommendations...</p></div>';

    const result = await api(`/api/recommendations/${state.currentUser.id}?n=20`);
    if (result && result.movies) {
        state.recommendations = result.movies;
        state.lastTiming = result.timing;
        renderMovieRow('recommendations-row', result.movies, true);
        updateTimingDisplay(result.timing);
        $('#rec-subtitle').textContent = `${result.movies.length} personalized picks for ${state.currentUser.name} — hybrid ML scoring`;
    }
}

function updateTimingDisplay(timing) {
    if (!timing) return;
    $('#inference-time').textContent = timing.total_ms;
    const badge = $('#inference-badge');
    badge.title = `Content: ${timing.content_ms}ms | Collab: ${timing.collab_ms}ms | Weights: ${timing.weights.content}/${timing.weights.collaborative} | Profile: ${timing.user_profile}`;
}

async function loadBrowse() {
    const genreParam = state.browseGenre ? `&genre=${encodeURIComponent(state.browseGenre)}` : '';
    const result = await api(`/api/movies?page=${state.browsePage}&limit=24${genreParam}`);
    if (result) {
        const grid = $('#browse-grid');
        grid.innerHTML = '';
        result.movies.forEach(m => grid.appendChild(createMovieCard(m)));
        renderPagination(result.page, result.pages);
    }
}

function renderPagination(current, total) {
    const container = $('#pagination');
    container.innerHTML = '';
    if (total <= 1) return;

    const prevBtn = document.createElement('button');
    prevBtn.className = 'page-btn'; prevBtn.textContent = '← Prev';
    prevBtn.disabled = current <= 1;
    prevBtn.onclick = () => { state.browsePage = current - 1; loadBrowse(); };
    container.appendChild(prevBtn);

    const maxShow = 5;
    let start = Math.max(1, current - Math.floor(maxShow / 2));
    let end = Math.min(total, start + maxShow - 1);
    start = Math.max(1, end - maxShow + 1);
    for (let i = start; i <= end; i++) {
        const btn = document.createElement('button');
        btn.className = `page-btn ${i === current ? 'active' : ''}`;
        btn.textContent = i;
        btn.onclick = () => { state.browsePage = i; loadBrowse(); };
        container.appendChild(btn);
    }

    const nextBtn = document.createElement('button');
    nextBtn.className = 'page-btn'; nextBtn.textContent = 'Next →';
    nextBtn.disabled = current >= total;
    nextBtn.onclick = () => { state.browsePage = current + 1; loadBrowse(); };
    container.appendChild(nextBtn);
}

// ─── Genre Tabs ─────────────────────────────────────────────────────────────

async function loadGenreTabs() {
    const genres = await api('/api/genres');
    if (!genres) return;
    const container = $('#genre-tabs');
    container.innerHTML = '';
    const allTab = document.createElement('button');
    allTab.className = 'genre-tab active'; allTab.textContent = 'All';
    allTab.onclick = () => selectGenre(null, allTab);
    container.appendChild(allTab);
    genres.forEach(genre => {
        const tab = document.createElement('button');
        tab.className = 'genre-tab'; tab.textContent = genre;
        tab.onclick = () => selectGenre(genre, tab);
        container.appendChild(tab);
    });
}

function selectGenre(genre, tabEl) {
    state.browseGenre = genre; state.browsePage = 1;
    $$('.genre-tab').forEach(t => t.classList.remove('active'));
    tabEl.classList.add('active');
    loadBrowse();
}

// ─── Users ──────────────────────────────────────────────────────────────────

async function loadUsers() {
    state.users = await api('/api/users') || [];
    renderUserDropdown();
    if (state.users.length > 0 && !state.currentUser) selectUser(state.users[0]);
}

function renderUserDropdown() {
    const dropdown = $('#user-dropdown');
    dropdown.innerHTML = '';
    state.users.forEach(user => {
        const option = document.createElement('div');
        option.className = `user-option ${state.currentUser?.id === user.id ? 'active' : ''}`;
        // Use dynamic genres if available, otherwise fall back to static
        const displayGenres = (user.dynamic_genres && user.dynamic_genres.length > 0)
            ? user.dynamic_genres.map(g => g.name).join(', ')
            : user.preferred_genres.join(', ');
        option.innerHTML = `
            <div class="user-avatar" style="background: ${user.avatar_color}">${user.name.charAt(0)}</div>
            <div><div class="user-option-name">${user.name}</div><div class="user-option-genres">♥ ${displayGenres}</div></div>`;
        option.onclick = (e) => { e.stopPropagation(); selectUser(user); dropdown.classList.remove('active'); $('#user-switcher').classList.remove('open'); };
        dropdown.appendChild(option);
    });
}

async function selectUser(user) {
    state.currentUser = user;
    $('#current-user-avatar').style.background = user.avatar_color;
    $('#current-user-avatar').textContent = user.name.charAt(0);
    $('#current-user-name').textContent = user.name;
    renderUserDropdown();
    connectWebSocket();
    loadRecommendations();
    loadTasteProfile();
    showToast(`Switched to ${user.name}`, 'success');
}

// ─── WebSocket ──────────────────────────────────────────────────────────────

function setWSStatus(status) {
    const dot = $('#ws-dot');
    const label = $('#ws-label');
    dot.className = `ws-dot ${status}`;
    label.textContent = status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected';
}

function connectWebSocket() {
    if (state.ws) { state.ws.close(); state.ws = null; }
    if (!state.currentUser) return;

    setWSStatus('connecting');
    const url = `${WS_BASE}/ws/${state.currentUser.id}`;

    try {
        state.ws = new WebSocket(url);

        state.ws.onopen = () => {
            setWSStatus('connected');
            if (state.wsReconnectTimer) { clearTimeout(state.wsReconnectTimer); state.wsReconnectTimer = null; }
        };

        state.ws.onmessage = (event) => {
            try { handleWSMessage(JSON.parse(event.data)); } catch (e) { console.error('WS parse error:', e); }
        };

        state.ws.onclose = () => {
            setWSStatus('disconnected');
            state.wsReconnectTimer = setTimeout(() => connectWebSocket(), 3000);
        };

        state.ws.onerror = () => setWSStatus('disconnected');

        const pingInterval = setInterval(() => {
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.ws.send(JSON.stringify({ type: 'ping' }));
            } else { clearInterval(pingInterval); }
        }, 30000);
    } catch (e) { setWSStatus('disconnected'); }
}

function handleWSMessage(msg) {
    switch (msg.type) {
        case 'recommendations':
            if (msg.data && msg.data.length > 0) {
                state.recommendations = msg.data;
                state.lastTiming = msg.timing;
                renderMovieRow('recommendations-row', msg.data, true);
                if (msg.timing) updateTimingDisplay(msg.timing);
                showToast('Recommendations updated in real-time!', 'info');
            }
            break;
        case 'profile_updated':
            handleProfileUpdate(msg.data);
            break;
        case 'activity': addActivityItem(msg.data); break;
        case 'pong': break;
    }
}

// ─── Activity Feed ──────────────────────────────────────────────────────────

function addActivityItem(data) {
    const list = $('#activity-list');
    const empty = list.querySelector('.activity-empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'activity-item';
    let icon = '👁️', action = 'viewed';
    if (data.event_type === 'rated') { icon = '⭐'; action = `rated ${data.rating}/5`; }
    item.innerHTML = `${icon} <span class="activity-user">${data.user_name}</span> ${action} <span class="activity-movie">${data.movie_title}</span> <span class="activity-time">just now</span>`;
    list.insertBefore(item, list.firstChild);
    while (list.children.length > 20) list.removeChild(list.lastChild);
}

// ─── Movie Modal ────────────────────────────────────────────────────────────

async function openMovieModal(movieId) {
    const movie = await api(`/api/movies/${movieId}`);
    if (!movie) return;
    state.modalMovie = movie;

    if (state.currentUser) {
        api('/api/events', { method: 'POST', body: JSON.stringify({ user_id: state.currentUser.id, movie_id: movieId, event_type: 'view' }) });
    }

    $('#modal-poster').style.background = `linear-gradient(135deg, ${movie.gradient_start}, ${movie.gradient_end})`;
    $('#modal-title').textContent = movie.title;
    $('#modal-meta').innerHTML = `<span>📅 ${movie.year}</span><span>⏱️ ${movie.runtime} min</span><span>⭐ ${movie.avg_rating.toFixed(1)} (${movie.rating_count} ratings)</span><span>🎬 ${movie.director}</span>`;
    $('#modal-genres').innerHTML = movie.genres.map(g => `<span class="genre-pill">${g}</span>`).join('');
    $('#modal-desc').textContent = movie.description;
    $('#modal-cast').innerHTML = `<h3>Cast</h3><p>${movie.cast.join(', ')}</p>`;

    await loadUserRating(movieId);
    loadSimilarMovies(movieId);

    $('#movie-modal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal(id) {
    $(`#${id}`).classList.remove('active');
    if (!$('#movie-modal').classList.contains('active') && !$('#explain-modal').classList.contains('active') && !$('#metrics-modal').classList.contains('active')) {
        document.body.style.overflow = '';
    }
}

async function loadUserRating(movieId) {
    const stars = $$('#modal-stars .star');
    let currentRating = 0;

    if (state.currentUser) {
        const userData = await api(`/api/users/${state.currentUser.id}`);
        if (userData?.ratings) {
            const existing = userData.ratings.find(r => r.movie_id === movieId);
            if (existing) currentRating = existing.rating;
        }
    }

    updateStarDisplay(currentRating);

    stars.forEach((star, index) => {
        const value = index + 1;
        star.onmouseenter = () => { stars.forEach((s, i) => s.classList.toggle('hover', i <= index)); $('#rating-label').textContent = ['', 'Poor', 'Fair', 'Good', 'Great', 'Excellent'][value]; };
        star.onmouseleave = () => { stars.forEach(s => s.classList.remove('hover')); $('#rating-label').textContent = currentRating > 0 ? `Your rating: ${currentRating}/5` : 'Click to rate'; };
        star.onclick = async () => {
            if (!state.currentUser) { showToast('Select a user first!', 'warning'); return; }
            currentRating = value;
            updateStarDisplay(currentRating);
            $('#rating-label').textContent = `Your rating: ${value}/5`;
            const result = await api('/api/ratings', { method: 'POST', body: JSON.stringify({ user_id: state.currentUser.id, movie_id: movieId, rating: value }) });
            if (result) showToast(`Rated "${state.modalMovie.title}" ${value}/5 ⭐`, 'success');
        };
    });
}

function updateStarDisplay(rating) {
    $$('#modal-stars .star').forEach((star, i) => star.classList.toggle('active', i < rating));
    $('#rating-label').textContent = rating > 0 ? `Your rating: ${rating}/5` : 'Click to rate';
}

async function loadSimilarMovies(movieId) {
    const container = $('#similar-row');
    container.innerHTML = '<div class="loading-container"><div class="loading-spinner"></div></div>';
    const similar = await api(`/api/movies/${movieId}/similar?n=8`);
    if (similar && similar.length > 0) {
        container.innerHTML = '';
        similar.forEach(m => container.appendChild(createMovieCard(m)));
    } else {
        container.innerHTML = '<p style="color: var(--text-tertiary); font-size: 13px;">No similar movies found</p>';
    }
}

// ─── Explanation Modal ──────────────────────────────────────────────────────

async function openExplanation(movieId) {
    if (!state.currentUser) { showToast('Select a user to see explanations', 'warning'); return; }

    const modal = $('#explain-modal');
    const content = $('#explain-content');
    content.innerHTML = '<div class="loading-container"><div class="loading-spinner"></div><p>Analyzing scoring...</p></div>';
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    const data = await api(`/api/recommendations/${state.currentUser.id}/explain/${movieId}`);
    if (!data) { content.innerHTML = '<p>Failed to load explanation</p>'; return; }

    const cw = data.scoring.content_weight;
    const fw = data.scoring.collab_weight;
    const cwPct = Math.round(cw * 100);
    const fwPct = Math.round(fw * 100);

    let similarHtml = '';
    if (data.content_based.similar_to_liked && data.content_based.similar_to_liked.length > 0) {
        similarHtml = data.content_based.similar_to_liked.map(s =>
            `<div class="similar-item"><span>${s.movie}</span><span class="sim-score">${(s.similarity * 100).toFixed(1)}% similar</span></div>`
        ).join('');
    } else {
        similarHtml = '<p style="color: var(--text-tertiary); font-size: 13px;">Using genre preferences (cold start)</p>';
    }

    const predRating = data.collaborative.predicted_rating;

    content.innerHTML = `
        <h2>Scoring Breakdown</h2>
        <p class="explain-subtitle">${data.movie.title} — recommended to ${data.user.name}</p>

        <div class="explain-section">
            <h3>⚖️ Hybrid Weights (${data.scoring.user_profile})</h3>
            <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;">${data.scoring.explanation}</p>
            <div class="weight-bar">
                <div class="content-fill" style="width: ${cwPct}%"></div>
                <div class="collab-fill" style="width: ${fwPct}%"></div>
            </div>
            <div class="weight-labels">
                <span>🟡 Content-Based: ${cwPct}%</span>
                <span>🟣 Collaborative: ${fwPct}%</span>
            </div>
        </div>

        <div class="explain-section">
            <h3>🟡 Content-Based Score</h3>
            <p style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 8px;">${data.content_based.description}</p>
            <div class="similar-list">${similarHtml}</div>
        </div>

        <div class="explain-section">
            <h3>🟣 Collaborative Score</h3>
            <p style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 8px;">${data.collaborative.description}</p>
            ${predRating ? `<div style="margin-top: 8px;">Predicted Rating: <span class="predicted-rating">${predRating.toFixed(1)}</span> <span style="font-size: 14px; color: var(--text-tertiary);">/ 5.0</span></div>` : '<p style="color: var(--text-tertiary); font-size: 13px;">No collaborative data for this user-movie pair</p>'}
        </div>
    `;
}

// ─── Metrics Dashboard ──────────────────────────────────────────────────────

async function openMetrics() {
    $('#metrics-modal').classList.add('active');
    document.body.style.overflow = 'hidden';
    await refreshMetrics();
}

async function refreshMetrics() {
    const [metricsData, evalData, healthData] = await Promise.all([
        api('/api/metrics'),
        api('/api/evaluation'),
        api('/api/health'),
    ]);

    // ── Performance Tab ──
    if (metricsData) {
        $('#m-total-requests').textContent = metricsData.total_requests.toLocaleString();
        $('#m-total-recs').textContent = metricsData.total_recommendations.toLocaleString();
        $('#m-total-ratings').textContent = metricsData.total_ratings.toLocaleString();
        const mins = Math.floor(metricsData.uptime_seconds / 60);
        const secs = Math.round(metricsData.uptime_seconds % 60);
        $('#m-uptime').textContent = `${mins}m ${secs}s`;

        // Latency table
        const tbody = $('#latency-tbody');
        tbody.innerHTML = '';
        for (const [ep, data] of Object.entries(metricsData.endpoints)) {
            const lat = data.latency_ms;
            const p95class = lat.p95 > 100 ? 'warn' : '';
            tbody.innerHTML += `<tr>
                <td class="endpoint-name">${ep}</td>
                <td>${data.count}</td>
                <td>${lat.avg}</td>
                <td>${lat.p50}</td>
                <td class="${p95class}">${lat.p95}</td>
                <td>${lat.p99}</td>
            </tr>`;
        }

        // Inference breakdown
        const inf = metricsData.model_inference;
        $('#inference-breakdown').innerHTML = `
            <div class="inference-bar"><div class="bar-label">Total</div><div class="bar-value">${inf.latency_ms.avg}<span class="bar-unit">ms avg</span></div></div>
            <div class="inference-bar"><div class="bar-label">Content-Based</div><div class="bar-value">${inf.content_based_ms.avg}<span class="bar-unit">ms avg</span></div></div>
            <div class="inference-bar"><div class="bar-label">Collaborative</div><div class="bar-value">${inf.collaborative_ms.avg}<span class="bar-unit">ms avg</span></div></div>
            <div class="inference-bar"><div class="bar-label">P95 Total</div><div class="bar-value">${inf.latency_ms.p95}<span class="bar-unit">ms</span></div></div>
        `;
    }

    // ── ML Evaluation Tab ──
    if (evalData && evalData.k_metrics) {
        const etbody = $('#eval-tbody');
        etbody.innerHTML = '';
        for (const [k, m] of Object.entries(evalData.k_metrics)) {
            etbody.innerHTML += `<tr>
                <td class="k-label">${k}</td>
                <td>${m.precision.toFixed(4)}</td>
                <td>${m.recall.toFixed(4)}</td>
                <td>${m.ndcg.toFixed(4)}</td>
                <td>${m.hit_rate.toFixed(4)}</td>
            </tr>`;
        }

        const sm = evalData.system_metrics;
        $('#eval-system-metrics').innerHTML = `
            <div class="metric-card"><div class="metric-label">Catalog Coverage</div><div class="metric-value good">${(sm.catalog_coverage * 100).toFixed(1)}%</div><div class="metric-sub">${sm.catalog_items_recommended} / ${sm.total_catalog_size} items</div></div>
            <div class="metric-card"><div class="metric-label">Genre Diversity</div><div class="metric-value">${sm.avg_genre_diversity}</div><div class="metric-sub">avg genres per rec set / ${sm.total_genres} total</div></div>
        `;

        const meta = evalData.meta;
        $('#eval-meta').innerHTML = `
            <strong>Methodology:</strong> ${meta.methodology}<br>
            <strong>Users Evaluated:</strong> ${meta.users_evaluated} / ${meta.total_users}<br>
            <strong>Total Ratings:</strong> ${meta.total_ratings.toLocaleString()}<br>
            <strong>Test Split:</strong> ${(meta.test_ratio * 100)}%<br>
            <strong>Evaluation Time:</strong> ${meta.evaluation_time_seconds}s<br>
            <strong>Last Run:</strong> ${meta.evaluated_at}
        `;
    }

    // ── System Health Tab ──
    if (metricsData) {
        const cache = metricsData.cache;
        const hitRatePct = (cache.hit_rate * 100).toFixed(1);
        const hitClass = cache.hit_rate > 0.5 ? 'good' : cache.hit_rate > 0.2 ? 'warn' : '';
        $('#cache-metrics').innerHTML = `
            <div class="metric-card"><div class="metric-label">Hit Rate</div><div class="metric-value ${hitClass}">${hitRatePct}%</div></div>
            <div class="metric-card"><div class="metric-label">Hits</div><div class="metric-value">${cache.hits}</div></div>
            <div class="metric-card"><div class="metric-label">Misses</div><div class="metric-value">${cache.misses}</div></div>
            <div class="metric-card"><div class="metric-label">Total Lookups</div><div class="metric-value">${cache.total_lookups}</div></div>
        `;

        const tr = metricsData.training;
        $('#training-metrics').innerHTML = `
            <div class="metric-card"><div class="metric-label">Training Cycles</div><div class="metric-value">${tr.total_cycles}</div></div>
            <div class="metric-card"><div class="metric-label">Last Trained</div><div class="metric-value" style="font-size:12px">${tr.last_trained || '--'}</div></div>
            <div class="metric-card"><div class="metric-label">Avg Duration</div><div class="metric-value">${tr.avg_duration_s}s</div></div>
            <div class="metric-card"><div class="metric-label">Last Duration</div><div class="metric-value">${tr.last_duration_s}s</div></div>
        `;

        const ws = metricsData.websocket;
        $('#ws-metrics').innerHTML = `
            <div class="metric-card"><div class="metric-label">Active Connections</div><div class="metric-value good">${ws.active_connections}</div></div>
            <div class="metric-card"><div class="metric-label">Total Connections</div><div class="metric-value">${ws.total_connections}</div></div>
            <div class="metric-card"><div class="metric-label">Messages Sent</div><div class="metric-value">${ws.messages_sent}</div></div>
        `;
    }

    if (healthData) {
        const hs = $('#health-status');
        hs.innerHTML = '';
        for (const [comp, status] of Object.entries(healthData.components)) {
            const isUp = status === 'up' || status.includes('active');
            hs.innerHTML += `<div class="health-row"><span>${comp}</span><span class="health-status-badge ${isUp ? 'up' : 'down'}">${status}</span></div>`;
        }
    }
}

// ─── Search ─────────────────────────────────────────────────────────────────

let searchDebounce = null;

function initSearch() {
    const input = $('#search-input');
    const results = $('#search-results');

    input.addEventListener('input', () => {
        clearTimeout(searchDebounce);
        const query = input.value.trim();
        if (query.length < 2) { results.classList.remove('active'); return; }

        searchDebounce = setTimeout(async () => {
            const data = await api(`/api/movies?search=${encodeURIComponent(query)}&limit=8`);
            if (data && data.movies.length > 0) {
                results.innerHTML = data.movies.map(movie => `
                    <div class="search-result-item" onclick="openMovieModal(${movie.id}); document.getElementById('search-results').classList.remove('active');">
                        <div class="search-result-poster" style="background: linear-gradient(135deg, ${movie.gradient_start}, ${movie.gradient_end})"></div>
                        <div class="search-result-info">
                            <div class="search-result-title">${movie.title}</div>
                            <div class="search-result-meta">${movie.year} · ${movie.genres.join(', ')} · ⭐ ${movie.avg_rating.toFixed(1)}</div>
                        </div>
                    </div>`).join('');
                results.classList.add('active');
            } else {
                results.innerHTML = '<div class="search-result-item"><div class="search-result-info"><div class="search-result-title">No results found</div></div></div>';
                results.classList.add('active');
            }
        }, 300);
    });

    document.addEventListener('click', (e) => { if (!e.target.closest('.search-box')) results.classList.remove('active'); });
}

// ─── Event Handlers ─────────────────────────────────────────────────────────

function initEventHandlers() {
    // User switcher
    const switcher = $('#user-switcher');
    const dropdown = $('#user-dropdown');
    switcher.addEventListener('click', (e) => { e.stopPropagation(); dropdown.classList.toggle('active'); switcher.classList.toggle('open'); });
    document.addEventListener('click', () => { dropdown.classList.remove('active'); switcher.classList.remove('open'); });

    // Movie modal
    $('#modal-close').addEventListener('click', () => closeModal('movie-modal'));
    $('#movie-modal').addEventListener('click', (e) => { if (e.target === $('#movie-modal')) closeModal('movie-modal'); });

    // Explain modal
    $('#explain-close').addEventListener('click', () => closeModal('explain-modal'));
    $('#explain-modal').addEventListener('click', (e) => { if (e.target === $('#explain-modal')) closeModal('explain-modal'); });

    // Metrics modal
    $('#metrics-btn').addEventListener('click', openMetrics);
    $('#metrics-close').addEventListener('click', () => closeModal('metrics-modal'));
    $('#metrics-modal').addEventListener('click', (e) => { if (e.target === $('#metrics-modal')) closeModal('metrics-modal'); });

    // Metrics tabs
    $$('.metrics-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.metrics-tab').forEach(t => t.classList.remove('active'));
            $$('.metrics-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            $(`#panel-${tab.dataset.tab}`).classList.add('active');
        });
    });

    // Eval re-run
    $('#eval-rerun-btn').addEventListener('click', async () => {
        $('#eval-rerun-btn').textContent = '⏳ Running...';
        $('#eval-rerun-btn').disabled = true;
        await api('/api/evaluation/run', { method: 'POST' });
        await refreshMetrics();
        $('#eval-rerun-btn').textContent = '🔄 Re-run Evaluation';
        $('#eval-rerun-btn').disabled = false;
        showToast('Evaluation complete!', 'success');
    });

    // Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal('movie-modal'); closeModal('explain-modal'); closeModal('metrics-modal');
        }
    });

    // Activity toggle
    const activityFeed = $('#activity-feed');
    $('#activity-toggle').addEventListener('click', () => {
        activityFeed.classList.toggle('collapsed');
        $('#activity-toggle').textContent = activityFeed.classList.contains('collapsed') ? '+' : '−';
    });
    // Logo
    $('#logo').addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}

// ─── Taste Profile ──────────────────────────────────────────────────────

async function loadTasteProfile() {
    const profileEl = $('#taste-profile');
    if (!state.currentUser) {
        profileEl.style.display = 'none';
        return;
    }

    const data = await api(`/api/users/${state.currentUser.id}/profile`);
    if (!data) {
        profileEl.style.display = 'none';
        return;
    }

    renderTasteProfile(data.genres, data.total_ratings, data.evolution);
}

function renderTasteProfile(genres, totalRatings, evolution) {
    const profileEl = $('#taste-profile');
    const barsEl = $('#taste-bars');
    const badgesEl = $('#taste-shift-badges');
    const labelEl = $('#taste-label');
    const evoEl = $('#taste-evolution');

    if (!genres || genres.length === 0) {
        profileEl.style.display = 'block';
        barsEl.innerHTML = `
            <div class="taste-profile-empty">
                <p>🎬 Rate some movies to build your taste profile</p>
                <p class="taste-empty-hint">Your genre preferences will evolve as you rate movies</p>
            </div>`;
        badgesEl.innerHTML = '';
        evoEl.style.display = 'none';
        labelEl.textContent = 'No ratings yet';
        return;
    }

    profileEl.style.display = 'block';
    labelEl.textContent = `Based on ${totalRatings} rating${totalRatings !== 1 ? 's' : ''}`;

    // Render genre bars with animation
    barsEl.innerHTML = '';
    genres.forEach((genre, i) => {
        const pct = Math.round(genre.score * 100);
        const rank = i + 1;
        const row = document.createElement('div');
        row.className = 'taste-bar-row';
        row.innerHTML = `
            <span class="taste-bar-rank rank-${rank}">#${rank}</span>
            <span class="taste-bar-label">${genre.name}</span>
            <div class="taste-bar-track">
                <div class="taste-bar-fill genre-${i}" style="width: 0%">
                    <span class="taste-bar-score">${pct}%</span>
                </div>
            </div>`;
        barsEl.appendChild(row);

        // Animate the bar width after a delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                row.querySelector('.taste-bar-fill').style.width = `${pct}%`;
            }, 50 + i * 80);
        });
    });

    // Render evolution (rising/falling trends)
    if (evolution && (evolution.rising.length > 0 || evolution.falling.length > 0)) {
        evoEl.style.display = 'flex';
        const risingEl = $('#evolution-rising');
        const fallingEl = $('#evolution-falling');

        if (evolution.rising.length > 0) {
            risingEl.innerHTML = '📈 Rising: ' + evolution.rising.map(r =>
                `<span class="evo-genre">${r.genre}</span>`
            ).join(', ');
        } else {
            risingEl.innerHTML = '';
        }

        if (evolution.falling.length > 0) {
            fallingEl.innerHTML = '📉 Cooling: ' + evolution.falling.map(r =>
                `<span class="evo-genre">${r.genre}</span>`
            ).join(', ');
        } else {
            fallingEl.innerHTML = '';
        }
    } else {
        evoEl.style.display = 'none';
    }
}

function handleProfileUpdate(data) {
    // Render updated taste profile
    renderTasteProfile(
        data.genres,
        data.total_ratings,
        null  // Evolution will be fetched on next full load
    );

    // Show taste shift badges
    const badgesEl = $('#taste-shift-badges');
    badgesEl.innerHTML = '';

    if (data.gained && data.gained.length > 0) {
        data.gained.forEach(g => {
            badgesEl.innerHTML += `<span class="taste-shift-badge gained">+ ${g}</span>`;
        });
    }
    if (data.lost && data.lost.length > 0) {
        data.lost.forEach(g => {
            badgesEl.innerHTML += `<span class="taste-shift-badge lost">− ${g}</span>`;
        });
    }

    // Show toast for taste shifts
    if (data.gained.length > 0 || data.lost.length > 0) {
        let msg = 'Taste profile updated';
        if (data.gained.length > 0) msg += ` — +${data.gained.join(', ')}`;
        if (data.lost.length > 0) msg += ` — -${data.lost.join(', ')}`;
        showToast(msg, 'info');
    }

    // Update current user's genres in state & dropdown
    if (state.currentUser && data.genres.length > 0) {
        state.currentUser.dynamic_genres = data.genres;
        state.currentUser.preferred_genres = data.genres.map(g => g.name);
        renderUserDropdown();
    }

    // Clear shift badges after 5 seconds
    setTimeout(() => {
        badgesEl.innerHTML = '';
    }, 5000);
}

// ─── Init ───────────────────────────────────────────────────────────────────

async function init() {
    initEventHandlers();
    initSearch();
    await Promise.all([loadUsers(), loadTrending(), loadTopRated(), loadGenreTabs()]);
    await loadBrowse();
}

document.addEventListener('DOMContentLoaded', init);
