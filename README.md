# 🎬 RecoAI — Real-Time ML-Powered Recommendation System

A production-grade movie recommendation system built end-to-end with **hybrid ML** (content-based + collaborative filtering), **real-time WebSocket updates**, **full observability**, and **ML evaluation metrics**.

> **Not a tutorial project.** This system demonstrates real ML engineering: offline evaluation with Precision@K/NDCG@K, inference-time instrumentation, cache-aware request flow, and dynamic model weight tuning based on user profile maturity.

---

## 🏗️ Architecture

```
User Request → FastAPI → [TTL Cache Check] → Hybrid ML Engine → [Cache Store] → Response
                  ↓                              ↓                    ↓
           Metrics Middleware           Content-Based + SVD      WebSocket Push
                  ↓                     (TF-IDF)   (Matrix       (real-time)
           Latency Tracking              Cosine     Factorization)
                  ↓                        ↓            ↓
           /api/metrics              Dynamic Weight Combiner
                                    (cold-start aware)
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/CSS/JS)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────────┐ │
│  │ Movie    │ │ Star     │ │ Search & │ │ Observability          │ │
│  │ Browser  │ │ Ratings  │ │ Filters  │ │ Dashboard (metrics,    │ │
│  │          │ │          │ │          │ │ eval, health)          │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬───────────────┘ │
│       │             │            │                 │                 │
│       └─────────────┴────────────┴─────────────────┘                │
│                           │ WebSocket │ REST API                    │
└───────────────────────────┴──────┬────┴─────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────┐
│                      Backend (FastAPI + Python)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────────┐  │
│  │ REST API │ │ WebSocket│ │ Metrics      │ │ Background        │  │
│  │ 18+      │ │ Server   │ │ Middleware   │ │ Trainer (5-min    │  │
│  │ endpoints│ │ (per-user│ │ (latency,    │ │ retraining cycle) │  │
│  │          │ │ channels)│ │ inference)   │ │                   │  │
│  └────┬─────┘ └────┬─────┘ └──────┬───────┘ └─────────┬─────────┘  │
│       │             │              │                    │            │
│  ┌────┴─────────────┴──────────────┴────────────────────┴────────┐  │
│  │                    Hybrid ML Engine                            │  │
│  │  ┌─────────────────┐    ┌──────────────────┐                  │  │
│  │  │ Content-Based   │    │ Collaborative    │                  │  │
│  │  │ TF-IDF + Cosine │    │ TruncatedSVD     │                  │  │
│  │  │ Similarity      │    │ 49 latent factors│                  │  │
│  │  └────────┬────────┘    └────────┬─────────┘                  │  │
│  │           └──────────┬───────────┘                            │  │
│  │              Dynamic Weight Combiner                          │  │
│  │         (cold: 85/15 → warm: transition → established: 35/65)│  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────┐  ┌───────┴───────┐  ┌─────────────────────────┐ │
│  │ TTL Cache     │  │ SQLite DB     │  │ Model Artifacts         │ │
│  │ (LRU, 60s)   │  │ (WAL mode)    │  │ (joblib serialized)     │ │
│  │ Redis-swappable│ │               │  │                         │ │
│  └───────────────┘  └───────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Methodology

### Content-Based Filtering
- **Approach**: TF-IDF vectorization on concatenated movie metadata (genres × 3 + description + director)
- **Similarity**: Cosine similarity on 500×500 matrix (5,000 max TF-IDF features, bigrams)
- **Cold-start**: Genre profile vectors (averaged TF-IDF per genre) for new users

### Collaborative Filtering
- **Approach**: TruncatedSVD matrix factorization on 50×500 user-item rating matrix
- **Latent Factors**: 49 components capturing user/item interaction patterns
- **Prediction**: Reconstructed rating = user_factors @ item_factors

### Hybrid Combination
Dynamic weighting based on user profile maturity:

| Profile | Ratings | Content Weight | Collab Weight | Rationale |
|---------|---------|---------------|---------------|-----------|
| Cold Start | < 5 | 85% | 15% | Not enough interaction data for CF to be reliable |
| Warm | 5-20 | Linear transition | Linear transition | Gradually trust CF as data accumulates |
| Established | 20+ | 35% | 65% | CF captures nuanced taste patterns better |

### Offline Evaluation
Hold-out temporal split (80/20) with relevance threshold ≥ 3.5:

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| **Precision** | Measured | Measured | Measured |
| **Recall** | Measured | Measured | Measured |
| **NDCG** | Measured | Measured | Measured |
| **Hit Rate** | Measured | Measured | Measured |

Additional metrics: **Catalog Coverage** (% of items recommended) and **Genre Diversity** (avg distinct genres per recommendation set).

> Run live evaluation: `POST /api/evaluation/run`  
> View results: `GET /api/evaluation`

---

## ⚡ System Design

### Request Flow
```
1. Client sends GET /api/recommendations/{user_id}
2. Metrics middleware starts latency timer
3. TTL Cache lookup (60s TTL, LRU eviction)
   → Cache HIT: return immediately (~0.5ms)
   → Cache MISS: continue to step 4
4. Load user's rated movies from SQLite (WAL mode for concurrent reads)
5. Determine user profile (cold/warm/established) → select weights
6. Content-based inference: cosine similarity aggregation (~2ms)
7. Collaborative inference: SVD factor dot product (~1ms)
8. Hybrid score normalization and combination
9. Fetch movie details for top-N recommendations
10. Store in cache, return response
11. Metrics middleware records endpoint latency + inference breakdown
```

### Why This Cache Design?
- **TTL Cache (not Redis)**: Single-instance deployment, no external dependencies
- **Same interface as Redis**: `get(key)`, `set(key, value, ttl)`, `invalidate_user(id)`
- **Swap for production**: Replace `TTLCache` with `redis.Redis` — API is identical
- **Invalidation**: Rating submission invalidates that user's cache + clears trending cache

### Real-Time Updates (Not Fake)
- **WebSocket per user**: `ws://host/ws/{user_id}` — persistent bidirectional connection
- **Event flow**: User rates movie → API saves → cache invalidated → ML re-scores → WebSocket pushes new recommendations
- **Activity broadcast**: All connected users see live activity feed (who rated what)
- **Keepalive**: 30-second ping/pong to detect stale connections

---

## 📊 Observability

### Tracked Metrics (via `/api/metrics`)

| Category | Metrics |
|----------|---------|
| **API Latency** | Per-endpoint: count, avg, P50, P95, P99, min, max (ms) |
| **Model Inference** | Total, content-based, collaborative breakdown (ms) |
| **Cache** | Hits, misses, hit rate, total lookups |
| **Training** | Cycle count, avg duration, last trained timestamp |
| **WebSocket** | Active connections, total connections, messages sent |
| **System** | Uptime, total requests, recommendations generated, ratings submitted |

### Production-Ready Design
Current implementation uses in-process metrics (bounded deques). For production:
- Swap `MetricsCollector` → Prometheus client library
- Export via `/metrics` in OpenMetrics format
- Scrape with Prometheus → Grafana dashboards
- The metric names and semantics are already aligned with Prometheus conventions

---

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run everything (generates data → seeds DB → trains models → starts server)
python run.py

# Open http://localhost:8000
```

### Docker
```bash
docker-compose up --build
# Models are trained at build time → instant startup
```

### API Documentation
```
http://localhost:8000/docs    # Interactive Swagger UI
```

---

## 📁 Project Structure

```
ml-recommendation-system/
├── run.py                    # Entry point (orchestrates everything)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # One-command deployment
│
├── data/
│   ├── generate_dataset.py   # Synthetic data generator (500 movies, 50 users, 10K ratings)
│   ├── seed_db.py            # SQLite database seeder
│   └── generated/            # JSON data files (movies, users, ratings)
│
├── ml/
│   ├── content_based.py      # TF-IDF + cosine similarity engine
│   ├── collaborative.py      # SVD matrix factorization engine
│   ├── hybrid.py             # Dynamic-weight hybrid combiner
│   ├── evaluation.py         # Precision@K, Recall@K, NDCG@K, Coverage, Diversity
│   └── trainer.py            # Training pipeline + background retrainer
│
├── server/
│   ├── app.py                # FastAPI (18+ endpoints, WebSocket, middleware)
│   ├── database.py           # SQLite CRUD operations
│   ├── models.py             # Pydantic request/response schemas
│   ├── metrics.py            # Observability collector (latency, inference, cache)
│   └── cache.py              # TTL LRU cache (Redis-swappable)
│
├── frontend/
│   ├── index.html            # SPA with metrics dashboard + explanation panels
│   ├── styles.css            # Dark glassmorphism design system
│   └── app.js                # Interactive frontend (WS, ratings, metrics, explanations)
│
├── models/                   # Serialized model artifacts (joblib)
└── app.db                    # SQLite database
```

---

## 🔑 Key API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/recommendations/{user_id}` | Personalized hybrid recommendations with scoring metadata |
| `GET` | `/api/recommendations/{user_id}/explain/{movie_id}` | Full scoring breakdown (weights, similarities, predicted rating) |
| `POST` | `/api/ratings` | Submit rating → triggers real-time recommendation refresh |
| `GET` | `/api/metrics` | Full observability data (latency, inference, cache, training) |
| `GET` | `/api/evaluation` | Offline ML metrics (P@K, R@K, NDCG@K, Coverage, Diversity) |
| `POST` | `/api/evaluation/run` | Re-run offline evaluation |
| `GET` | `/api/health` | Component health check |
| `WS` | `/ws/{user_id}` | Real-time recommendation push + activity streaming |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI + Uvicorn | Async, high-performance, native WebSocket support |
| ML | scikit-learn (TF-IDF, TruncatedSVD, cosine_similarity) | Industry-standard, efficient for this scale |
| Data | SQLite (WAL mode) | Zero-config, concurrent reads, sufficient for single-instance |
| Cache | In-memory LRU + TTL | Same API as Redis, zero external deps for dev |
| Frontend | Vanilla HTML/CSS/JS | No framework overhead, full control |
| Deployment | Docker + Compose | Reproducible builds, one-command deploy |

---

## 💡 Interview Talking Points

1. **"How does the 65% match work?"** → Hybrid scoring: normalized content-based (cosine sim on TF-IDF vectors) + collaborative (SVD predicted ratings), dynamically weighted by user maturity. Click any match badge for full breakdown.

2. **"How do you handle cold start?"** → Dynamic weight shifting. New users (< 5 ratings) get 85% content-based using genre preference profiles. As ratings accumulate, collaborative filtering weight increases linearly to 65%.

3. **"Is the real-time actually real?"** → Yes. WebSocket per user. When a rating is submitted: cache is invalidated, ML engine re-scores, new recommendations are pushed over the WebSocket within the same request cycle. Activity is broadcast to all connected users.

4. **"How do you know the recommendations are good?"** → Offline evaluation with hold-out temporal split. Precision@K measures relevance, NDCG@K measures ranking quality, Hit Rate@K measures coverage. All metrics available via API and UI dashboard.

5. **"What about latency?"** → Full request pipeline instrumented. Content-based inference ~2ms, collaborative ~1ms, total hybrid ~5ms. P95 endpoint latency tracked per endpoint. Cache reduces repeated queries to ~0.5ms.

6. **"How would you scale this?"** → Cache layer is Redis-swappable (same interface). Models can be served via separate inference service. Training pipeline already decoupled (background thread → separate worker). SQLite → PostgreSQL for multi-instance.
