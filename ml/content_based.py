"""
Content-Based Filtering Engine using TF-IDF and Cosine Similarity.
Recommends movies based on genre and description similarity.
"""

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


class ContentBasedEngine:
    """Content-based recommendation engine using TF-IDF on movie metadata."""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_ids = []
        self.movie_index = {}     # movie_id -> matrix index
        self.movie_data = {}      # movie_id -> movie dict
        self.genre_profiles = {}  # genre -> average TF-IDF vector
    
    def _build_content_string(self, movie: dict) -> str:
        """Build a rich content string from movie metadata for TF-IDF."""
        parts = []
        
        # Add genres (repeated for emphasis)
        genres = movie.get("genres", [])
        parts.append(" ".join(genres) * 3)
        
        # Add description
        if movie.get("description"):
            parts.append(movie["description"])
        
        # Add director
        if movie.get("director"):
            parts.append(movie["director"])
        
        return " ".join(parts)
    
    def train(self, movies: list):
        """Train the content-based model on movie data.
        
        Args:
            movies: List of movie dicts with id, genres, description, etc.
        """
        self.movie_ids = [m["id"] for m in movies]
        self.movie_index = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.movie_data = {m["id"]: m for m in movies}
        
        # Build content strings
        content_strings = [self._build_content_string(m) for m in movies]
        
        # Fit TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(content_strings)
        
        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Build genre profiles (average TF-IDF vector per genre)
        all_genres = set()
        for m in movies:
            all_genres.update(m.get("genres", []))
        
        for genre in all_genres:
            genre_indices = [
                self.movie_index[m["id"]]
                for m in movies
                if genre in m.get("genres", [])
            ]
            if genre_indices:
                genre_vectors = self.tfidf_matrix[genre_indices]
                self.genre_profiles[genre] = np.asarray(genre_vectors.mean(axis=0)).flatten()
        
        print(f"   ✅ Content-based model trained on {len(movies)} movies")
    
    def get_similar(self, movie_id: int, n: int = 10, exclude: set = None) -> list:
        """Get n most similar movies to a given movie.
        
        Returns: List of (movie_id, similarity_score) tuples.
        """
        if movie_id not in self.movie_index:
            return []
        
        idx = self.movie_index[movie_id]
        similarities = self.similarity_matrix[idx]
        
        # Get top indices (excluding self and any exclusions)
        exclude = exclude or set()
        exclude.add(movie_id)
        
        scored = []
        for i, score in enumerate(similarities):
            mid = self.movie_ids[i]
            if mid not in exclude:
                scored.append((mid, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
    
    def get_recommendations_for_profile(
        self, liked_movie_ids: list, n: int = 10, exclude: set = None
    ) -> list:
        """Get recommendations based on a list of liked movies.
        
        Aggregates similarity scores across all liked movies.
        
        Returns: List of (movie_id, score) tuples.
        """
        if not liked_movie_ids:
            return []
        
        exclude = exclude or set()
        exclude.update(liked_movie_ids)
        
        # Aggregate similarity scores
        aggregate_scores = np.zeros(len(self.movie_ids))
        valid_count = 0
        
        for mid in liked_movie_ids:
            if mid in self.movie_index:
                idx = self.movie_index[mid]
                aggregate_scores += self.similarity_matrix[idx]
                valid_count += 1
        
        if valid_count == 0:
            return []
        
        # Average the scores
        aggregate_scores /= valid_count
        
        # Build result
        scored = []
        for i, score in enumerate(aggregate_scores):
            mid = self.movie_ids[i]
            if mid not in exclude:
                scored.append((mid, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
    
    def get_genre_recommendations(self, preferred_genres: list, n: int = 10) -> list:
        """Get recommendations based on genre preferences (for cold start).
        
        Returns: List of (movie_id, score) tuples.
        """
        if not preferred_genres or not self.genre_profiles:
            return []
        
        # Average the genre profile vectors
        vectors = [
            self.genre_profiles[g]
            for g in preferred_genres
            if g in self.genre_profiles
        ]
        
        if not vectors:
            return []
        
        avg_profile = np.mean(vectors, axis=0).reshape(1, -1)
        
        # Compute similarity to all movies
        similarities = cosine_similarity(avg_profile, self.tfidf_matrix).flatten()
        
        scored = [(self.movie_ids[i], float(s)) for i, s in enumerate(similarities)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
    
    def save(self):
        """Save model artifacts to disk."""
        joblib.dump({
            "tfidf": self.tfidf,
            "tfidf_matrix": self.tfidf_matrix,
            "similarity_matrix": self.similarity_matrix,
            "movie_ids": self.movie_ids,
            "movie_index": self.movie_index,
            "movie_data": self.movie_data,
            "genre_profiles": self.genre_profiles,
        }, self.model_dir / "content_based.joblib")
    
    def load(self) -> bool:
        """Load model artifacts from disk. Returns True if successful."""
        path = self.model_dir / "content_based.joblib"
        if not path.exists():
            return False
        
        data = joblib.load(path)
        self.tfidf = data["tfidf"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.similarity_matrix = data["similarity_matrix"]
        self.movie_ids = data["movie_ids"]
        self.movie_index = data["movie_index"]
        self.movie_data = data["movie_data"]
        self.genre_profiles = data["genre_profiles"]
        return True
