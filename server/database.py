"""
Database operations for the recommendation system.
SQLite-based CRUD for movies, users, ratings, and events.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
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
