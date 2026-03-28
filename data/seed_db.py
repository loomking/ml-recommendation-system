"""
Seed SQLite database with generated JSON data.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta


DB_PATH = Path(__file__).parent.parent / "app.db"
DATA_DIR = Path(__file__).parent / "generated"


def create_tables(conn: sqlite3.Connection):
    """Create all database tables."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            year INTEGER,
            genres TEXT NOT NULL,
            description TEXT,
            runtime INTEGER,
            director TEXT,
            cast_members TEXT,
            gradient_start TEXT,
            gradient_end TEXT,
            quality_score REAL,
            view_count INTEGER DEFAULT 0,
            avg_rating REAL DEFAULT 0,
            rating_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            preferred_genres TEXT,
            avatar_color TEXT
        );

        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (movie_id) REFERENCES movies(id),
            UNIQUE(user_id, movie_id)
        );

        CREATE TABLE IF NOT EXISTS user_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER,
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        );

        CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
        CREATE INDEX IF NOT EXISTS idx_events_user ON user_events(user_id);
    """)


def seed_movies(conn: sqlite3.Connection, movies: list):
    """Insert movies into the database."""
    for m in movies:
        conn.execute(
            """INSERT OR REPLACE INTO movies 
               (id, title, year, genres, description, runtime, director, 
                cast_members, gradient_start, gradient_end, quality_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                m["id"], m["title"], m["year"],
                json.dumps(m["genres"]),
                m["description"], m["runtime"], m["director"],
                json.dumps(m["cast"]),
                m["gradient_start"], m["gradient_end"],
                m["quality_score"],
            )
        )


def seed_users(conn: sqlite3.Connection, users: list):
    """Insert users into the database."""
    for u in users:
        conn.execute(
            """INSERT OR REPLACE INTO users 
               (id, name, preferred_genres, avatar_color)
               VALUES (?, ?, ?, ?)""",
            (
                u["id"], u["name"],
                json.dumps(u["preferred_genres"]),
                u["avatar_color"],
            )
        )


def seed_ratings(conn: sqlite3.Connection, ratings: list):
    """Insert ratings and update movie statistics."""
    now = datetime.now()
    
    for r in ratings:
        created = now - timedelta(days=r["days_ago"])
        conn.execute(
            """INSERT OR REPLACE INTO ratings 
               (user_id, movie_id, rating, created_at)
               VALUES (?, ?, ?, ?)""",
            (r["user_id"], r["movie_id"], r["rating"], created.isoformat())
        )
    
    # Update movie aggregate stats
    conn.execute("""
        UPDATE movies SET 
            avg_rating = (SELECT COALESCE(AVG(rating), 0) FROM ratings WHERE ratings.movie_id = movies.id),
            rating_count = (SELECT COUNT(*) FROM ratings WHERE ratings.movie_id = movies.id)
    """)
    
    # Set view counts based on rating counts with some bonus
    conn.execute("""
        UPDATE movies SET view_count = rating_count * 3 + ABS(RANDOM()) % 100
    """)


def seed_database():
    """Main function to seed the database from generated JSON data."""
    if not DATA_DIR.exists():
        print("❌ No generated data found. Run generate_dataset.py first.")
        return False
    
    print(f"📦 Seeding database at {DB_PATH}...")
    
    # Remove existing DB for clean start
    if DB_PATH.exists():
        DB_PATH.unlink()
    
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        create_tables(conn)
        
        # Load JSON data
        with open(DATA_DIR / "movies.json", "r", encoding="utf-8") as f:
            movies = json.load(f)
        with open(DATA_DIR / "users.json", "r", encoding="utf-8") as f:
            users = json.load(f)
        with open(DATA_DIR / "ratings.json", "r", encoding="utf-8") as f:
            ratings = json.load(f)
        
        print(f"   Loading {len(movies)} movies...")
        seed_movies(conn, movies)
        
        print(f"   Loading {len(users)} users...")
        seed_users(conn, users)
        
        print(f"   Loading {len(ratings)} ratings...")
        seed_ratings(conn, ratings)
        
        conn.commit()
        print("✅ Database seeded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    seed_database()
