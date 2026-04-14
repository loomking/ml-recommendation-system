# RecoAI — Movie Recommendation System

A real-time movie recommendation system that uses a hybrid approach combining content-based filtering and collaborative filtering to suggest movies. Built with FastAPI for the backend, vanilla JS frontend, and scikit-learn for the ML pipeline.

**Live demo**: [ml-recommendation-system.onrender.com](https://ml-recommendation-system.onrender.com)

## What it does

- Recommends movies based on what you've liked before and what similar users enjoy
- Updates recommendations in real-time when you rate a movie (via WebSocket)
- Shows a match percentage for each recommendation with a full scoring breakdown
- Tracks system metrics like API latency, cache performance, and model inference times

## How the recommendation works

The system uses two approaches and combines them:

**Content-based filtering** — looks at movie metadata (genres, description, director) and finds similar movies using TF-IDF vectors and cosine similarity. Good for new users since it doesn't need much data.

**Collaborative filtering** — uses SVD matrix factorization on the user-item rating matrix to find patterns. Works better when there's enough rating data from the user.

These are combined with dynamic weights that change based on how many ratings a user has:
- New users (< 5 ratings): 85% content-based, 15% collaborative
- Established users (20+ ratings): 35% content-based, 65% collaborative
- In between: weights transition linearly

## Tech stack

- **Backend**: Python, FastAPI, Uvicorn
- **ML**: scikit-learn (TF-IDF, TruncatedSVD, cosine similarity)
- **Database**: SQLite with WAL mode
- **Frontend**: HTML, CSS, JavaScript (no frameworks)
- **Deployment**: Docker, Render
- **Caching**: In-memory TTL cache (designed to be swappable with Redis)

## Project structure

```
├── run.py                  # entry point - runs everything
├── data/
│   ├── generate_dataset.py # creates synthetic movie/user/rating data
│   └── seed_db.py          # loads data into sqlite
├── ml/
│   ├── content_based.py    # TF-IDF + cosine similarity
│   ├── collaborative.py    # SVD matrix factorization
│   ├── hybrid.py           # combines both with dynamic weights
│   ├── evaluation.py       # precision@k, ndcg, recall metrics
│   └── trainer.py          # training pipeline + background retrainer
├── server/
│   ├── app.py              # FastAPI app (REST + WebSocket)
│   ├── database.py         # sqlite operations
│   ├── models.py           # pydantic schemas
│   ├── cache.py            # TTL cache layer
│   └── metrics.py          # latency and performance tracking
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── Dockerfile
└── docker-compose.yml
```

## Running locally

```bash
pip install -r requirements.txt
python run.py
# open http://localhost:8000
```

First run takes ~30s because it generates data and trains the models. After that it starts in a couple seconds.

## Running with Docker

```bash
docker-compose up --build
```

## API endpoints

Some of the main ones:

| Endpoint | What it does |
|----------|-------------|
| `GET /api/recommendations/{user_id}` | Get personalized recommendations |
| `GET /api/recommendations/{user_id}/explain/{movie_id}` | See why a movie was recommended |
| `POST /api/ratings` | Submit a rating (triggers real-time update) |
| `GET /api/metrics` | System performance metrics |
| `GET /api/evaluation` | ML evaluation results |
| `GET /api/health` | Health check |
| `WS /ws/{user_id}` | WebSocket for live updates |

Full docs at `/docs` (Swagger UI).

## Evaluation

The system runs offline evaluation using a hold-out split (80% train, 20% test). It measures:

- **Precision@K** — how many of the top-K recommendations are actually relevant
- **Recall@K** — how many relevant items appear in the top-K
- **NDCG@K** — how well the ranking order matches ideal ranking
- **Catalog coverage** — what percentage of movies get recommended
- **Genre diversity** — how diverse the recommendations are

You can re-run evaluation from the dashboard or via `POST /api/evaluation/run`.

## Things I'd change for production

- Replace the in-memory cache with Redis
- Move the metrics collector to Prometheus + Grafana
- Use PostgreSQL instead of SQLite for multi-instance support
- Separate the training pipeline into its own worker/service
- Add proper A/B testing for comparing model versions
