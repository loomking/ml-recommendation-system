"""
FastAPI application — REST API + WebSocket + Observability.

Request Flow:
    User → FastAPI → [Cache Check] → Hybrid ML Engine → [Cache Store] → Response
                   ↓                                   ↓
              Metrics Middleware              WebSocket Push (real-time)
                   ↓
           Observability Collector (latency, inference time, cache stats)
"""

import asyncio
import json
import time
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import RatingCreate, EventCreate
from server import database as db
from server.metrics import MetricsCollector
from server.cache import TTLCache
from ml.trainer import load_trained_models, train_models, BackgroundTrainer
from ml.trainer import load_movies_from_db, load_ratings_from_db
from ml.evaluation import ModelEvaluator


# ─── Globals ──────────────────────────────────────────────────────────────────

trainer: BackgroundTrainer | None = None
metrics = MetricsCollector()
rec_cache = TTLCache(max_size=500, ttl_seconds=60)      # Recommendations cache
trending_cache = TTLCache(max_size=10, ttl_seconds=120)  # Trending/top-rated cache
evaluator: ModelEvaluator | None = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load ML models, run initial evaluation. Shutdown: stop trainer."""
    global trainer, evaluator

    print("🚀 Starting recommendation server...")

    # Load trained models
    engine = load_trained_models()

    # Start background trainer (retrain every 5 minutes)
    trainer = BackgroundTrainer(interval_seconds=300)
    trainer.start(engine)

    # Run initial offline evaluation
    evaluator = ModelEvaluator(engine)
    try:
        movies = load_movies_from_db()
        ratings = load_ratings_from_db()
        users_data = db.get_all_users()
        eval_results = evaluator.evaluate(ratings, users_data, movies)
        print(f"📊 Evaluation complete: P@10={eval_results['k_metrics']['@10']['precision']:.4f}, "
              f"NDCG@10={eval_results['k_metrics']['@10']['ndcg']:.4f}")
    except Exception as e:
        print(f"⚠️ Evaluation skipped: {e}")

    yield

    if trainer:
        trainer.stop()
    print("👋 Server stopped")


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecoAI — ML Movie Recommendations",
    description="Real-time ML-powered movie recommendation system with hybrid filtering, observability, and evaluation metrics.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Metrics Middleware ───────────────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request latency for every API call."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    # Record by path pattern (e.g., /api/movies, /api/recommendations)
    path = request.url.path
    if path.startswith("/api/"):
        # Normalize dynamic segments for grouping
        parts = path.split("/")
        if len(parts) >= 4 and parts[2] in ("movies", "recommendations", "users"):
            if len(parts) > 3 and parts[3].isdigit():
                parts[3] = "{id}"
        endpoint = "/".join(parts)
        is_error = response.status_code >= 400
        metrics.record_request(endpoint, latency_ms, error=is_error)

    return response


# ─── WebSocket Connection Manager ────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections per user with metrics tracking."""

    def __init__(self):
        self.active: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active:
            self.active[user_id] = []
        self.active[user_id].append(websocket)
        metrics.ws_connect()

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active:
            self.active[user_id] = [ws for ws in self.active[user_id] if ws != websocket]
            if not self.active[user_id]:
                del self.active[user_id]
        metrics.ws_disconnect()

    async def send_to_user(self, user_id: int, message: dict):
        if user_id in self.active:
            dead = []
            for ws in self.active[user_id]:
                try:
                    await ws.send_json(message)
                    metrics.ws_message()
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active[user_id].remove(ws)

    async def broadcast(self, message: dict, exclude_user: int = None):
        for user_id, connections in list(self.active.items()):
            if user_id != exclude_user:
                for ws in connections:
                    try:
                        await ws.send_json(message)
                        metrics.ws_message()
                    except Exception:
                        pass

    @property
    def connection_count(self):
        return sum(len(conns) for conns in self.active.values())


ws_manager = ConnectionManager()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_engine():
    """Get the current hybrid recommendation engine."""
    if trainer:
        return trainer.get_engine()
    return None


def generate_recommendations(user_id: int, n: int = 20):
    """Generate recommendations with full inference timing.
    
    Returns: (rec_movies, inference_timing_dict)
    """
    engine = get_engine()
    if not engine:
        return [], {}

    user = db.get_user(user_id)
    if not user:
        return [], {}

    liked_ids = db.get_user_rated_movie_ids(user_id, min_rating=3.5)

    # ── Time the full inference pipeline ────────────────────────
    t_start = time.perf_counter()

    # Content-based scoring
    t_content_start = time.perf_counter()
    content_recs = engine.content.get_recommendations_for_profile(
        liked_ids, n=n * 3
    ) if liked_ids else engine.content.get_genre_recommendations(
        user.get("preferred_genres", []), n=n * 3
    )
    t_content = (time.perf_counter() - t_content_start) * 1000

    # Collaborative scoring
    t_collab_start = time.perf_counter()
    collab_recs = engine.collab.get_recommendations(user_id, n=n * 3)
    t_collab = (time.perf_counter() - t_collab_start) * 1000

    # Hybrid combination
    recs = engine.get_recommendations(
        user_id=user_id,
        rated_movie_ids=liked_ids,
        preferred_genres=user.get("preferred_genres", []),
        n=n,
    )

    t_total = (time.perf_counter() - t_start) * 1000

    # Record metrics
    metrics.record_inference(t_total, t_content, t_collab)

    # Fetch movie details
    rec_ids = [mid for mid, _ in recs]
    rec_movies = db.get_movies_by_ids(rec_ids)

    score_map = {mid: score for mid, score in recs}
    n_ratings = len(liked_ids)
    content_w, collab_w = engine._get_weights(n_ratings)

    for movie in rec_movies:
        movie["rec_score"] = round(score_map.get(movie["id"], 0), 3)
        movie["scoring"] = {
            "content_weight": round(content_w, 2),
            "collab_weight": round(collab_w, 2),
            "user_rating_count": n_ratings,
        }

    timing = {
        "total_ms": round(t_total, 2),
        "content_ms": round(t_content, 2),
        "collab_ms": round(t_collab, 2),
        "weights": {"content": round(content_w, 2), "collaborative": round(collab_w, 2)},
        "user_profile": "cold_start" if n_ratings < 5 else ("warm" if n_ratings < 20 else "established"),
    }

    return rec_movies, timing


async def push_recommendations(user_id: int):
    """Generate and push updated recommendations via WebSocket."""
    rec_movies, timing = generate_recommendations(user_id)
    if rec_movies:
        # Invalidate cache for this user
        rec_cache.invalidate_user(user_id)

        await ws_manager.send_to_user(user_id, {
            "type": "recommendations",
            "data": rec_movies,
            "timing": timing,
        })


# ─── REST API Endpoints ──────────────────────────────────────────────────────

# ── Health & System ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """System health check with component status."""
    engine = get_engine()
    return {
        "status": "healthy",
        "components": {
            "database": "up",
            "ml_engine": "up" if engine else "down",
            "background_trainer": "up" if trainer else "down",
            "websocket": f"{ws_manager.connection_count} active",
        },
        "version": "2.0.0",
    }


@app.get("/api/metrics")
async def get_metrics():
    """Full observability metrics — latency, inference, cache, WebSocket."""
    data = metrics.get_metrics()
    data["cache_details"] = rec_cache.stats()
    return data


@app.get("/api/evaluation")
async def get_evaluation():
    """Offline ML evaluation metrics — Precision@K, Recall@K, NDCG@K, Coverage."""
    if evaluator and evaluator.results:
        return evaluator.results
    raise HTTPException(status_code=503, detail="Evaluation not yet complete")


@app.post("/api/evaluation/run")
async def run_evaluation():
    """Trigger a fresh evaluation run."""
    engine = get_engine()
    if not engine or not evaluator:
        raise HTTPException(status_code=503, detail="Engine not ready")

    movies = load_movies_from_db()
    ratings = load_ratings_from_db()
    users_data = db.get_all_users()
    results = evaluator.evaluate(ratings, users_data, movies)
    return results


# ── Movies ────────────────────────────────────────────────────────────────

@app.get("/api/movies")
async def get_movies(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    genre: str = Query(None),
    search: str = Query(None),
):
    """Get paginated movies with optional genre/search filters."""
    return db.get_all_movies(page=page, limit=limit, genre=genre, search=search)


@app.get("/api/movies/trending")
async def get_trending(limit: int = Query(20, ge=1, le=50)):
    """Get trending movies (cached 2 min)."""
    cache_key = f"trending:{limit}"
    cached = trending_cache.get(cache_key)
    if cached:
        metrics.record_cache_hit()
        return cached

    metrics.record_cache_miss()
    result = db.get_trending_movies(limit=limit)
    trending_cache.set(cache_key, result)
    return result


@app.get("/api/movies/top-rated")
async def get_top_rated(limit: int = Query(20, ge=1, le=50)):
    """Get top rated movies (cached 2 min)."""
    cache_key = f"toprated:{limit}"
    cached = trending_cache.get(cache_key)
    if cached:
        metrics.record_cache_hit()
        return cached

    metrics.record_cache_miss()
    result = db.get_top_rated_movies(limit=limit)
    trending_cache.set(cache_key, result)
    return result


@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get movie details."""
    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@app.get("/api/movies/{movie_id}/similar")
async def get_similar_movies(movie_id: int, n: int = Query(10, ge=1, le=30)):
    """Get movies similar to the given movie (content-based similarity)."""
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Models not loaded")

    similar = engine.get_similar_movies(movie_id, n=n)
    similar_ids = [mid for mid, _ in similar]
    movies = db.get_movies_by_ids(similar_ids)

    score_map = {mid: score for mid, score in similar}
    for movie in movies:
        movie["similarity"] = round(score_map.get(movie["id"], 0), 3)

    return movies


# ── Recommendations ───────────────────────────────────────────────────────

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, n: int = Query(20, ge=1, le=100)):
    """Get personalized hybrid recommendations with scoring metadata.
    
    Response includes:
        - rec_score: hybrid match score (0-1)
        - scoring.content_weight / collab_weight: dynamic weights used
        - scoring.user_rating_count: basis for weight selection
    """
    # Check cache first
    cache_key = f"recs:{user_id}:{n}"
    cached = rec_cache.get(cache_key)
    if cached:
        metrics.record_cache_hit()
        return cached

    metrics.record_cache_miss()

    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rec_movies, timing = generate_recommendations(user_id, n)

    result = {"movies": rec_movies, "timing": timing}
    rec_cache.set(cache_key, result)
    return result


@app.get("/api/recommendations/{user_id}/explain/{movie_id}")
async def explain_recommendation(user_id: int, movie_id: int):
    """Explain WHY a movie was recommended to a user.
    
    Returns:
        - Hybrid scoring weights (content vs collaborative)
        - Content-based: similarity to user's top-rated movies
        - Collaborative: predicted rating from SVD factorization
        - User profile classification (cold_start / warm / established)
    """
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Models not loaded")

    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    movie = db.get_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    liked_ids = db.get_user_rated_movie_ids(user_id, min_rating=3.5)
    all_rated = db.get_user_rated_movie_ids(user_id)

    # Get explanation from hybrid engine
    explanation = engine.explain_recommendation(user_id, movie_id, liked_ids)
    n_ratings = len(liked_ids)
    content_w, collab_w = engine._get_weights(n_ratings)

    # Enhance with user profile info
    profile = "cold_start" if n_ratings < 5 else ("warm" if n_ratings < 20 else "established")

    return {
        "movie": {"id": movie["id"], "title": movie["title"], "genres": movie["genres"]},
        "user": {"id": user["id"], "name": user["name"], "total_ratings": len(all_rated)},
        "scoring": {
            "content_weight": round(content_w, 2),
            "collab_weight": round(collab_w, 2),
            "user_profile": profile,
            "explanation": (
                f"With {n_ratings} liked movies, using {round(content_w*100)}% content-based "
                f"+ {round(collab_w*100)}% collaborative scoring."
            ),
        },
        "content_based": {
            "similar_to_liked": explanation.get("similar_to", []),
            "description": "Cosine similarity between TF-IDF vectors (genres + description)",
        },
        "collaborative": {
            "predicted_rating": explanation.get("predicted_rating", None),
            "description": "SVD matrix factorization (49 latent factors) predicted rating",
        },
    }


# ── Ratings ───────────────────────────────────────────────────────────────

@app.post("/api/ratings")
async def submit_rating(rating: RatingCreate):
    """Submit or update a movie rating. Triggers real-time recommendation refresh."""
    movie = db.get_movie(rating.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    result = db.add_rating(rating.user_id, rating.movie_id, rating.rating)
    db.add_event(rating.user_id, rating.movie_id, "rate")
    metrics.record_rating()

    # Invalidate cache and push updated recommendations
    rec_cache.invalidate_user(rating.user_id)
    trending_cache.clear()
    asyncio.create_task(push_recommendations(rating.user_id))

    # Broadcast activity
    user = db.get_user(rating.user_id)
    await ws_manager.broadcast({
        "type": "activity",
        "data": {
            "user_name": user["name"] if user else "Unknown",
            "movie_title": movie["title"],
            "event_type": "rated",
            "rating": rating.rating,
        }
    })

    return result


# ── Events ────────────────────────────────────────────────────────────────

@app.post("/api/events")
async def track_event(event: EventCreate):
    """Track user behavior event (view, click, search)."""
    result = db.add_event(event.user_id, event.movie_id, event.event_type)

    if event.event_type == "view" and event.movie_id:
        user = db.get_user(event.user_id)
        movie = db.get_movie(event.movie_id)
        if user and movie:
            await ws_manager.broadcast({
                "type": "activity",
                "data": {
                    "user_name": user["name"],
                    "movie_title": movie["title"],
                    "event_type": "viewed",
                }
            })

    return result


# ── Users ─────────────────────────────────────────────────────────────────

@app.get("/api/users")
async def get_users():
    """Get all demo users."""
    return db.get_all_users()


@app.get("/api/users/{user_id}")
async def get_user_detail(user_id: int):
    """Get user details including their ratings."""
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    ratings = db.get_user_ratings(user_id)
    user["ratings"] = ratings
    user["rating_count"] = len(ratings)
    return user


@app.get("/api/users/{user_id}/ratings")
async def get_user_ratings_api(user_id: int):
    return db.get_user_ratings(user_id)


@app.get("/api/stats")
async def get_stats():
    return db.get_stats()


@app.get("/api/genres")
async def get_genres():
    return [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket for real-time recommendation pushes and activity streaming."""
    await ws_manager.connect(websocket, user_id)

    try:
        # Send initial recommendations
        rec_movies, timing = generate_recommendations(user_id)
        if rec_movies:
            await websocket.send_json({
                "type": "recommendations",
                "data": rec_movies,
                "timing": timing,
            })

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "request_recommendations":
                    rec_movies, timing = generate_recommendations(user_id)
                    if rec_movies:
                        await websocket.send_json({
                            "type": "recommendations",
                            "data": rec_movies,
                            "timing": timing,
                        })
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, user_id)
    except Exception:
        ws_manager.disconnect(websocket, user_id)


# ─── Serve Frontend ──────────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(frontend_dir / "index.html")


app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
