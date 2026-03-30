"""
Collaborative Filtering Engine using SVD Matrix Factorization.
Recommends movies based on user-item interaction patterns.
"""

import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from pathlib import Path


class CollaborativeEngine:
    """Collaborative filtering using Truncated SVD on the user-item matrix."""
    
    def __init__(self, model_dir="models", n_factors=50):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.n_factors = n_factors
        self.svd = None
        self.user_factors = None     # (n_users, n_factors)
        self.item_factors = None     # (n_factors, n_items)
        self.predicted = None        # Reconstructed prediction matrix
        self.user_index = {}         # user_id -> matrix row
        self.item_index = {}         # movie_id -> matrix col
        self.reverse_user = {}       # matrix row -> user_id
        self.reverse_item = {}       # matrix col -> movie_id
        self.global_mean = 0.0
        self.user_means = {}         # user_id -> mean rating
        self.user_rated = {}         # user_id -> set of rated movie_ids
    
    def train(self, ratings_data: list):
        """Train collaborative filtering model.
        
        Args:
            ratings_data: List of (user_id, movie_id, rating) tuples.
        """
        if not ratings_data:
            print("   ⚠️ No ratings data for collaborative filtering")
            return
        
        # Build index mappings
        users = sorted(set(r[0] for r in ratings_data))
        items = sorted(set(r[1] for r in ratings_data))
        
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.item_index = {mid: i for i, mid in enumerate(items)}
        self.reverse_user = {i: uid for uid, i in self.user_index.items()}
        self.reverse_item = {i: mid for mid, i in self.item_index.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        # Build sparse user-item matrix
        rows, cols, vals = [], [], []
        for uid, mid, rating in ratings_data:
            rows.append(self.user_index[uid])
            cols.append(self.item_index[mid])
            vals.append(float(rating))
        
        matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_items)
        )
        
        # Compute means
        self.global_mean = np.mean(vals)
        
        for uid in users:
            user_ratings = [r[2] for r in ratings_data if r[0] == uid]
            self.user_means[uid] = np.mean(user_ratings)
        
        # Track rated items per user
        self.user_rated = {}
        for uid, mid, _ in ratings_data:
            if uid not in self.user_rated:
                self.user_rated[uid] = set()
            self.user_rated[uid].add(mid)
        
        # Apply SVD
        n_components = min(self.n_factors, n_users - 1, n_items - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        dense = matrix.toarray().astype(np.float64)
        
        self.user_factors = self.svd.fit_transform(dense)
        self.item_factors = self.svd.components_
        
        # Reconstruct predicted ratings
        self.predicted = self.user_factors @ self.item_factors
        
        print(f"   ✅ Collaborative model trained: {n_users} users × {n_items} items, {n_components} factors")
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict a user's rating for a specific movie."""
        if user_id not in self.user_index or movie_id not in self.item_index:
            return self.global_mean
        
        u_idx = self.user_index[user_id]
        i_idx = self.item_index[movie_id]
        
        pred = self.predicted[u_idx, i_idx]
        return float(max(1.0, min(5.0, pred)))
    
    def get_recommendations(
        self, user_id: int, n: int = 10, exclude: set = None
    ) -> list:
        """Get top-n recommendations for a user.
        
        Returns: List of (movie_id, predicted_score) tuples.
        """
        if user_id not in self.user_index:
            return []
        
        u_idx = self.user_index[user_id]
        predictions = self.predicted[u_idx]
        
        # Items to exclude (already rated + any additional)
        rated = self.user_rated.get(user_id, set())
        if exclude:
            rated = rated | set(exclude)
        
        scored = []
        for i_idx, score in enumerate(predictions):
            mid = self.reverse_item[i_idx]
            if mid not in rated:
                scored.append((mid, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
    
    def get_similar_users(self, user_id: int, n: int = 5) -> list:
        """Find users with similar taste profiles.
        
        Returns: List of (user_id, similarity_score) tuples.
        """
        if user_id not in self.user_index:
            return []
        
        u_idx = self.user_index[user_id]
        user_vec = self.user_factors[u_idx].reshape(1, -1)
        
        # Compute cosine similarity with all other users
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_vec, self.user_factors).flatten()
        
        scored = []
        for i, sim in enumerate(similarities):
            uid = self.reverse_user[i]
            if uid != user_id:
                scored.append((uid, float(sim)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
    
    def save(self):
        """Save model artifacts to disk."""
        joblib.dump({
            "svd": self.svd,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "predicted": self.predicted,
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user": self.reverse_user,
            "reverse_item": self.reverse_item,
            "global_mean": self.global_mean,
            "user_means": self.user_means,
            "user_rated": self.user_rated,
        }, self.model_dir / "collaborative.joblib")
    
    def load(self) -> bool:
        """Load model artifacts from disk. Returns True if successful."""
        path = self.model_dir / "collaborative.joblib"
        if not path.exists():
            return False
        
        data = joblib.load(path)
        self.svd = data["svd"]
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.predicted = data["predicted"]
        self.user_index = data["user_index"]
        self.item_index = data["item_index"]
        self.reverse_user = data["reverse_user"]
        self.reverse_item = data["reverse_item"]
        self.global_mean = data["global_mean"]
        self.user_means = data["user_means"]
        self.user_rated = data["user_rated"]
        return True
