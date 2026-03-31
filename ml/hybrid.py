"""
Hybrid Recommendation Engine.
Combines content-based and collaborative filtering with dynamic weighting.
"""

from ml.content_based import ContentBasedEngine
from ml.collaborative import CollaborativeEngine


class HybridEngine:
    """Combines content-based and collaborative filtering recommendations.
    
    Uses dynamic weighting:
    - New users (few ratings): heavier on content-based
    - Established users (many ratings): heavier on collaborative
    """
    
    COLD_START_THRESHOLD = 5   # Below this many ratings, user is "cold"
    WARM_THRESHOLD = 20        # Above this, full collaborative weight
    
    def __init__(self, content_engine: ContentBasedEngine, collab_engine: CollaborativeEngine):
        self.content = content_engine
        self.collab = collab_engine
    
    def _get_weights(self, n_ratings: int) -> tuple:
        """Compute dynamic weights (content_weight, collab_weight) based on user's rating count."""
        if n_ratings < self.COLD_START_THRESHOLD:
            # Cold start: rely heavily on content
            return (0.85, 0.15)
        elif n_ratings < self.WARM_THRESHOLD:
            # Transition: blend
            progress = (n_ratings - self.COLD_START_THRESHOLD) / (self.WARM_THRESHOLD - self.COLD_START_THRESHOLD)
            content_w = 0.85 - (progress * 0.45)   # 0.85 -> 0.40
            collab_w = 0.15 + (progress * 0.45)     # 0.15 -> 0.60
            return (content_w, collab_w)
        else:
            # Warm user: rely more on collaborative
            return (0.35, 0.65)
    
    def _normalize_scores(self, scored: list) -> dict:
        """Normalize (movie_id, score) list to [0, 1] range. Returns dict."""
        if not scored:
            return {}
        
        scores = [s for _, s in scored]
        min_s = min(scores)
        max_s = max(scores)
        range_s = max_s - min_s
        
        if range_s == 0:
            return {mid: 0.5 for mid, _ in scored}
        
        return {mid: (s - min_s) / range_s for mid, s in scored}
    
    def get_recommendations(
        self,
        user_id: int,
        rated_movie_ids: list = None,
        preferred_genres: list = None,
        n: int = 20,
    ) -> list:
        """Get hybrid recommendations for a user.
        
        Args:
            user_id: The user's ID.
            rated_movie_ids: List of movie IDs the user has rated highly (>=3.5).
            preferred_genres: User's preferred genres (for cold start).
            n: Number of recommendations to return.
        
        Returns:
            List of (movie_id, hybrid_score) tuples, sorted by score desc.
        """
        rated_movie_ids = rated_movie_ids or []
        preferred_genres = preferred_genres or []
        n_ratings = len(rated_movie_ids)
        
        content_w, collab_w = self._get_weights(n_ratings)
        
        exclude = set(rated_movie_ids)
        
        # ── Content-Based Scores ──────────────────────────────────────────
        content_scored = {}
        
        if rated_movie_ids:
            # Use liked movies to find similar
            recs = self.content.get_recommendations_for_profile(
                rated_movie_ids, n=n * 3, exclude=exclude
            )
            content_scored = self._normalize_scores(recs)
        elif preferred_genres:
            # Cold start: use genre preferences
            recs = self.content.get_genre_recommendations(
                preferred_genres, n=n * 3
            )
            content_scored = self._normalize_scores(recs)
        
        # ── Collaborative Filtering Scores ────────────────────────────────
        collab_scored = {}
        
        if user_id in self.collab.user_index:
            recs = self.collab.get_recommendations(
                user_id, n=n * 3, exclude=exclude
            )
            collab_scored = self._normalize_scores(recs)
        
        # ── Combine Scores ────────────────────────────────────────────────
        all_movie_ids = set(content_scored.keys()) | set(collab_scored.keys())
        
        hybrid_scores = []
        for mid in all_movie_ids:
            c_score = content_scored.get(mid, 0.0)
            f_score = collab_scored.get(mid, 0.0)
            
            hybrid = (content_w * c_score) + (collab_w * f_score)
            hybrid_scores.append((mid, hybrid))
        
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:n]
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> list:
        """Get similar movies using content-based similarity.
        
        Returns: List of (movie_id, score) tuples.
        """
        return self.content.get_similar(movie_id, n=n)
    
    def explain_recommendation(self, user_id: int, movie_id: int, rated_movie_ids: list) -> dict:
        """Explain why a movie was recommended.
        
        Returns a dict with content_score, collab_score, and explanation text.
        """
        n_ratings = len(rated_movie_ids)
        content_w, collab_w = self._get_weights(n_ratings)
        
        explanation = {
            "content_weight": round(content_w, 2),
            "collab_weight": round(collab_w, 2),
            "user_rating_count": n_ratings,
        }
        
        # Content-based: find similarity to user's liked movies
        if rated_movie_ids and movie_id in self.content.movie_index:
            similarities = []
            movie_data = self.content.movie_data
            for rated_id in rated_movie_ids[:5]:  # Top 5 most recent
                if rated_id in self.content.movie_index:
                    idx1 = self.content.movie_index[movie_id]
                    idx2 = self.content.movie_index[rated_id]
                    sim = float(self.content.similarity_matrix[idx1, idx2])
                    similarities.append({
                        "movie": movie_data.get(rated_id, {}).get("title", f"Movie {rated_id}"),
                        "similarity": round(sim, 3),
                    })
            
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            explanation["similar_to"] = similarities[:3]
        
        # Collaborative: predicted rating
        if user_id in self.collab.user_index:
            explanation["predicted_rating"] = round(
                self.collab.predict_rating(user_id, movie_id), 2
            )
        
        return explanation
    
    def save(self):
        """Save both engines."""
        self.content.save()
        self.collab.save()
    
    def load(self) -> bool:
        """Load both engines. Returns True if both loaded successfully."""
        return self.content.load() and self.collab.load()
