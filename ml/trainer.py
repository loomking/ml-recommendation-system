"""
Model Trainer — handles training pipeline and background retraining.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

from ml.content_based import ContentBasedEngine
from ml.collaborative import CollaborativeEngine
from ml.hybrid import HybridEngine


DB_PATH = Path(__file__).parent.parent / "app.db"
MODEL_DIR = Path(__file__).parent.parent / "models"


def load_movies_from_db(db_path: str = None) -> list:
    """Load all movies from the database."""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("SELECT * FROM movies").fetchall()
    movies = []
    for r in rows:
        movies.append({
            "id": r["id"],
            "title": r["title"],
            "year": r["year"],
            "genres": json.loads(r["genres"]),
            "description": r["description"],
            "runtime": r["runtime"],
            "director": r["director"],
            "cast": json.loads(r["cast_members"]) if r["cast_members"] else [],
            "gradient_start": r["gradient_start"],
            "gradient_end": r["gradient_end"],
            "quality_score": r["quality_score"],
        })
    
    conn.close()
    return movies


def load_ratings_from_db(db_path: str = None) -> list:
    """Load all ratings from the database as (user_id, movie_id, rating) tuples."""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    
    rows = conn.execute("SELECT user_id, movie_id, rating FROM ratings").fetchall()
    ratings = [(r[0], r[1], r[2]) for r in rows]
    
    conn.close()
    return ratings


def train_models(db_path: str = None) -> HybridEngine:
    """Train all models and return the hybrid engine."""
    model_dir = str(MODEL_DIR)
    
    print("🧠 Training recommendation models...")
    
    # Load data
    movies = load_movies_from_db(db_path)
    ratings = load_ratings_from_db(db_path)
    
    print(f"   📊 Data: {len(movies)} movies, {len(ratings)} ratings")
    
    # Train content-based
    content_engine = ContentBasedEngine(model_dir=model_dir)
    content_engine.train(movies)
    
    # Train collaborative
    collab_engine = CollaborativeEngine(model_dir=model_dir, n_factors=50)
    collab_engine.train(ratings)
    
    # Create hybrid
    hybrid = HybridEngine(content_engine, collab_engine)
    
    # Save models
    hybrid.save()
    print("💾 Models saved to disk")
    
    return hybrid


def load_trained_models() -> HybridEngine:
    """Load pre-trained models from disk."""
    model_dir = str(MODEL_DIR)
    
    content_engine = ContentBasedEngine(model_dir=model_dir)
    collab_engine = CollaborativeEngine(model_dir=model_dir)
    hybrid = HybridEngine(content_engine, collab_engine)
    
    if hybrid.load():
        print("✅ Models loaded from disk")
        return hybrid
    else:
        print("⚠️ No saved models found, training from scratch...")
        return train_models()


class BackgroundTrainer:
    """Runs periodic model retraining in a background thread."""
    
    def __init__(self, interval_seconds: int = 300):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None
        self.hybrid_engine = None
        self._lock = threading.Lock()
    
    def start(self, initial_engine: HybridEngine):
        """Start background training loop."""
        self.hybrid_engine = initial_engine
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"🔄 Background trainer started (interval: {self.interval}s)")
    
    def _run(self):
        """Background training loop."""
        while not self._stop_event.wait(self.interval):
            try:
                print("🔄 Background retraining started...")
                new_engine = train_models()
                
                with self._lock:
                    self.hybrid_engine = new_engine
                
                print("✅ Background retraining completed")
            except Exception as e:
                print(f"❌ Background training error: {e}")
    
    def get_engine(self) -> HybridEngine:
        """Thread-safe access to the current hybrid engine."""
        with self._lock:
            return self.hybrid_engine
    
    def stop(self):
        """Stop the background trainer."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)


if __name__ == "__main__":
    # Direct execution: train models
    engine = train_models()
    print("\n🎯 Testing recommendations for user 1...")
    recs = engine.get_recommendations(user_id=1, n=5)
    for mid, score in recs:
        movie = engine.content.movie_data.get(mid, {})
        print(f"   {movie.get('title', 'Unknown')} (score: {score:.3f})")
