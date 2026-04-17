"""
Database operations for the recommendation system.
SQLite-based CRUD for movies, users, ratings, and events.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = Path(__file__).parent.parent / "app.db"


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_movie(row) -> dict:
    """Convert a database row to a movie dict."""
    return {
        "id": row["id"],
        "title": row["title"],
        "year": row["year"],
        "genres": json.loads(row["genres"]),
        "description": row["description"],
        "runtime": row["runtime"],
        "director": row["director"],
        "cast": json.loads(row["cast_members"]) if row["cast_members"] else [],
        "gradient_start": row["gradient_start"],
        "gradient_end": row["gradient_end"],
        "avg_rating": round(row["avg_rating"], 1) if row["avg_rating"] else 0,
        "rating_count": row["rating_count"] or 0,
        "view_count": row["view_count"] or 0,
    }


def row_to_user(row) -> dict:
    """Convert a database row to a user dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "preferred_genres": json.loads(row["preferred_genres"]) if row["preferred_genres"] else [],
        "avatar_color": row["avatar_color"],
    }


# ─── Movie Operations ─────────────────────────────────────────────────────────

def get_all_movies(page: int = 1, limit: int = 20, genre: str = None, search: str = None) -> dict:
    """Get paginated movies with optional filters."""
    with get_db() as conn:
        query = "SELECT * FROM movies WHERE 1=1"
        params = []
        
        if genre:
            query += " AND genres LIKE ?"
            params.append(f'%"{genre}"%')
        
        if search:
            query += " AND (title LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        total = conn.execute(count_query, params).fetchone()[0]
        
        # Paginate
        query += " ORDER BY avg_rating DESC, rating_count DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, (page - 1) * limit])
        
        rows = conn.execute(query, params).fetchall()
        movies = [row_to_movie(r) for r in rows]
        
        return {
            "movies": movies,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit,
        }


def get_movie(movie_id: int) -> dict | None:
    """Get a single movie by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
        return row_to_movie(row) if row else None


def get_movies_by_ids(movie_ids: list) -> list:
    """Get multiple movies by their IDs, preserving order."""
    if not movie_ids:
        return []
    
    with get_db() as conn:
        placeholders = ",".join("?" * len(movie_ids))
        rows = conn.execute(
            f"SELECT * FROM movies WHERE id IN ({placeholders})", movie_ids
        ).fetchall()
        
        movie_map = {row_to_movie(r)["id"]: row_to_movie(r) for r in rows}
        return [movie_map[mid] for mid in movie_ids if mid in movie_map]


def get_trending_movies(limit: int = 20) -> list:
    """Get trending movies (highest recent activity)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT m.* FROM movies m
               LEFT JOIN ratings r ON m.id = r.movie_id
               GROUP BY m.id
               ORDER BY COUNT(r.id) DESC, m.avg_rating DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [row_to_movie(r) for r in rows]


def get_top_rated_movies(limit: int = 20) -> list:
    """Get top rated movies (minimum rating count threshold)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM movies 
               WHERE rating_count >= 5
               ORDER BY avg_rating DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [row_to_movie(r) for r in rows]


def increment_view_count(movie_id: int):
    """Increment a movie's view count."""
    with get_db() as conn:
        conn.execute(
            "UPDATE movies SET view_count = view_count + 1 WHERE id = ?",
            (movie_id,)
        )


# ─── User Operations ──────────────────────────────────────────────────────────

def get_all_users() -> list:
    """Get all users."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        return [row_to_user(r) for r in rows]


def get_user(user_id: int) -> dict | None:
    """Get a single user by ID."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return row_to_user(row) if row else None


# ─── Rating Operations ────────────────────────────────────────────────────────

def get_user_ratings(user_id: int) -> list:
    """Get all ratings for a user."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT r.*, m.title, m.genres, m.gradient_start, m.gradient_end
               FROM ratings r
               JOIN movies m ON r.movie_id = m.id
               WHERE r.user_id = ?
               ORDER BY r.created_at DESC""",
            (user_id,)
        ).fetchall()
        
        return [{
            "movie_id": r["movie_id"],
            "rating": r["rating"],
            "created_at": r["created_at"],
            "movie_title": r["title"],
            "movie_genres": json.loads(r["genres"]),
            "gradient_start": r["gradient_start"],
            "gradient_end": r["gradient_end"],
        } for r in rows]


def get_user_rated_movie_ids(user_id: int, min_rating: float = 0) -> list:
    """Get movie IDs rated by a user (optionally above a threshold)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT movie_id FROM ratings WHERE user_id = ? AND rating >= ?",
            (user_id, min_rating)
        ).fetchall()
        return [r["movie_id"] for r in rows]


def add_rating(user_id: int, movie_id: int, rating: float) -> dict:
    """Add or update a rating."""
    with get_db() as conn:
        now = datetime.now().isoformat()
        
        # Upsert rating
        conn.execute(
            """INSERT INTO ratings (user_id, movie_id, rating, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, movie_id) 
               DO UPDATE SET rating = ?, created_at = ?""",
            (user_id, movie_id, rating, now, rating, now)
        )
        
        # Update movie stats
        conn.execute(
            """UPDATE movies SET 
                avg_rating = (SELECT AVG(rating) FROM ratings WHERE movie_id = ?),
                rating_count = (SELECT COUNT(*) FROM ratings WHERE movie_id = ?)
               WHERE id = ?""",
            (movie_id, movie_id, movie_id)
        )
        
        return {"user_id": user_id, "movie_id": movie_id, "rating": rating, "created_at": now}


def get_user_rating(user_id: int, movie_id: int) -> float | None:
    """Get a specific user's rating for a movie."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT rating FROM ratings WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id)
        ).fetchone()
        return row["rating"] if row else None


# ─── Dynamic Genre Preference Computation ─────────────────────────────────────

def compute_dynamic_genres(user_id: int, top_n: int = 5) -> dict:
    """Compute a user's current genre preferences from their rating history.
    
    Uses recency-weighted scoring:
      - Exponential time decay: recent ratings matter more (half-life ~14 days)
      - Rating-value weighting: 5★ counts much more than 2★
      - Normalizes to 0-1 scale
    
    Returns:
        dict with keys:
            "genres": list of (genre, score) tuples, sorted desc, top_n
            "genre_scores": full dict of all genre → score
            "total_ratings": int
            "has_enough_data": bool (True if >= 3 ratings)
    """
    with get_db() as conn:
        rows = conn.execute(
            """SELECT r.rating, r.created_at, m.genres
               FROM ratings r
               JOIN movies m ON r.movie_id = m.id
               WHERE r.user_id = ?
               ORDER BY r.created_at DESC""",
            (user_id,)
        ).fetchall()
    
    if not rows:
        # Fall back to static preferred_genres
        user = get_user(user_id)
        if user and user.get("preferred_genres"):
            static = user["preferred_genres"]
            return {
                "genres": [(g, 1.0) for g in static[:top_n]],
                "genre_scores": {g: 1.0 for g in static},
                "total_ratings": 0,
                "has_enough_data": False,
            }
        return {"genres": [], "genre_scores": {}, "total_ratings": 0, "has_enough_data": False}
    
    import math
    
    now = datetime.now()
    genre_weights = {}   # genre → accumulated weighted score
    genre_counts = {}    # genre → count of ratings touching this genre
    
    for row in rows:
        rating = row["rating"]
        genres = json.loads(row["genres"])
        created_str = row["created_at"]
        
        # Parse timestamp
        try:
            created = datetime.fromisoformat(created_str)
        except (ValueError, TypeError):
            created = now  # fallback
        
        # ── Recency decay (half-life = 14 days) ──
        days_ago = max(0, (now - created).total_seconds() / 86400)
        recency_weight = math.exp(-0.0495 * days_ago)  # ln(2)/14 ≈ 0.0495
        
        # ── Rating-value weight ──
        # Map 1-5 star to a preference signal:
        #   5★ → 1.0 (strong positive)
        #   4★ → 0.7
        #   3★ → 0.3 (neutral)
        #   2★ → -0.2 (mild negative)
        #   1★ → -0.5 (strong negative)
        rating_signal = {5: 1.0, 4: 0.7, 3: 0.3, 2: -0.2, 1: -0.5}.get(
            int(round(rating)), 0.3
        )
        
        combined_weight = recency_weight * rating_signal
        
        for genre in genres:
            genre_weights[genre] = genre_weights.get(genre, 0) + combined_weight
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    if not genre_weights:
        return {"genres": [], "genre_scores": {}, "total_ratings": len(rows), "has_enough_data": False}
    
    # Normalize scores to [0, 1]
    max_score = max(abs(v) for v in genre_weights.values()) or 1
    normalized = {g: max(0, v / max_score) for g, v in genre_weights.items()}
    
    # Sort and pick top N
    sorted_genres = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    top_genres = [(g, round(s, 3)) for g, s in sorted_genres if s > 0.05][:top_n]
    
    return {
        "genres": top_genres,
        "genre_scores": {g: round(s, 3) for g, s in normalized.items()},
        "total_ratings": len(rows),
        "has_enough_data": len(rows) >= 3,
    }


def update_user_preferred_genres(user_id: int, genres: list):
    """Update a user's preferred_genres in the database.
    
    Args:
        user_id: The user ID.
        genres: List of genre name strings (e.g. ["Thriller", "Action", "Crime"]).
    """
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET preferred_genres = ? WHERE id = ?",
            (json.dumps(genres), user_id)
        )


def get_genre_evolution(user_id: int, window_days: int = 7) -> dict:
    """Compute how a user's genre preferences have shifted recently.
    
    Compares "recent window" preferences vs "all-time" to detect taste shifts.
    
    Returns:
        dict with "rising" and "falling" genres.
    """
    with get_db() as conn:
        now = datetime.now()
        cutoff = (now - timedelta(days=window_days)).isoformat()
        
        # Recent ratings
        recent_rows = conn.execute(
            """SELECT r.rating, m.genres
               FROM ratings r JOIN movies m ON r.movie_id = m.id
               WHERE r.user_id = ? AND r.created_at >= ?""",
            (user_id, cutoff)
        ).fetchall()
        
        # All ratings
        all_rows = conn.execute(
            """SELECT r.rating, m.genres
               FROM ratings r JOIN movies m ON r.movie_id = m.id
               WHERE r.user_id = ?""",
            (user_id,)
        ).fetchall()
    
    def genre_dist(rows):
        dist = {}
        total = 0
        for row in rows:
            rating = row["rating"]
            genres = json.loads(row["genres"])
            weight = max(0, (rating - 2.5) / 2.5)  # 0 for <=2.5, up to 1.0 for 5★
            for g in genres:
                dist[g] = dist.get(g, 0) + weight
                total += weight
        if total > 0:
            dist = {g: v / total for g, v in dist.items()}
        return dist
    
    recent_dist = genre_dist(recent_rows)
    all_dist = genre_dist(all_rows)
    
    rising = []
    falling = []
    
    all_genres = set(list(recent_dist.keys()) + list(all_dist.keys()))
    for g in all_genres:
        recent_share = recent_dist.get(g, 0)
        all_share = all_dist.get(g, 0)
        diff = recent_share - all_share
        if diff > 0.05:
            rising.append((g, round(diff, 3)))
        elif diff < -0.05:
            falling.append((g, round(abs(diff), 3)))
    
    rising.sort(key=lambda x: x[1], reverse=True)
    falling.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "rising": [{"genre": g, "shift": s} for g, s in rising[:3]],
        "falling": [{"genre": g, "shift": s} for g, s in falling[:3]],
        "recent_count": len(recent_rows),
    }


# ─── Event Operations ─────────────────────────────────────────────────────────

def add_event(user_id: int, movie_id: int, event_type: str) -> dict:
    """Track a user behavior event."""
    with get_db() as conn:
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO user_events (user_id, movie_id, event_type, created_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, movie_id, event_type, now)
        )
        
        # Auto-increment view count for view events
        if event_type == "view" and movie_id:
            conn.execute(
                "UPDATE movies SET view_count = view_count + 1 WHERE id = ?",
                (movie_id,)
            )
        
        return {"user_id": user_id, "movie_id": movie_id, "event_type": event_type}


def get_recent_events(limit: int = 20) -> list:
    """Get recent events for activity feed."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT e.*, u.name as user_name, m.title as movie_title
               FROM user_events e
               LEFT JOIN users u ON e.user_id = u.id
               LEFT JOIN movies m ON e.movie_id = m.id
               ORDER BY e.created_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        
        return [{
            "user_name": r["user_name"],
            "movie_title": r["movie_title"],
            "event_type": r["event_type"],
            "created_at": r["created_at"],
        } for r in rows]


# ─── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Get system statistics."""
    with get_db() as conn:
        movies = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM user_events").fetchone()[0]
        
        return {
            "total_movies": movies,
            "total_users": users,
            "total_ratings": ratings,
            "total_events": events,
        }
