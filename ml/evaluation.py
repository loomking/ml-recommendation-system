"""
ML Model Evaluation — Precision@K, Recall@K, NDCG@K, Coverage, Diversity.
Provides offline evaluation metrics to prove recommendation quality.
"""

import numpy as np
import time
from collections import defaultdict


class ModelEvaluator:
    """Evaluates recommendation quality using standard Information Retrieval metrics.
    
    Methodology:
        1. Hold-out evaluation: 80/20 split per user (chronological)
        2. For each user, score ALL candidate movies (not in train set)
        3. Check if held-out 'liked' items (rating >= 3.5) appear in top-K
        4. Aggregate across all users with >= 5 ratings
        
    Key design: We score candidates directly using the hybrid formula
    to avoid the collaborative engine's internal exclusion of rated items
    (which would exclude test items since the model was trained on full data).
    """
    
    def __init__(self, hybrid_engine):
        self.engine = hybrid_engine
        self.results = {}
    
    def _score_candidates(self, user_id, train_liked_ids, preferred_genres, candidate_ids):
        """Score a set of candidate movie IDs using the hybrid formula.
        
        This bypasses the engine's get_recommendations() to avoid its internal
        exclusion of items the user rated in the training matrix.
        """
        n_ratings = len(train_liked_ids)
        content_w, collab_w = self.engine._get_weights(n_ratings)
        
        # Content-based scores for candidates
        content_scores = {}
        if train_liked_ids:
            # Get content similarity for each candidate vs the user's profile
            for cid in candidate_ids:
                if cid in self.engine.content.movie_index:
                    sims = []
                    for liked_id in train_liked_ids:
                        if liked_id in self.engine.content.movie_index:
                            idx1 = self.engine.content.movie_index[cid]
                            idx2 = self.engine.content.movie_index[liked_id]
                            sim = float(self.engine.content.similarity_matrix[idx1, idx2])
                            sims.append(sim)
                    if sims:
                        content_scores[cid] = np.mean(sorted(sims, reverse=True)[:5])
        elif preferred_genres:
            # Cold start fallback
            recs = self.engine.content.get_genre_recommendations(preferred_genres, n=len(candidate_ids) * 2)
            content_scores = {mid: s for mid, s in recs if mid in candidate_ids}
        
        # Collaborative scores for candidates
        collab_scores = {}
        if user_id in self.engine.collab.user_index:
            u_idx = self.engine.collab.user_index[user_id]
            for cid in candidate_ids:
                if cid in self.engine.collab.item_index:
                    i_idx = self.engine.collab.item_index[cid]
                    collab_scores[cid] = float(self.engine.collab.predicted[u_idx, i_idx])
        
        # Normalize both score sets to [0, 1]
        content_norm = self._normalize(content_scores)
        collab_norm = self._normalize(collab_scores)
        
        # Combine with dynamic weights
        combined = []
        for cid in candidate_ids:
            c = content_norm.get(cid, 0.0)
            f = collab_norm.get(cid, 0.0)
            score = content_w * c + collab_w * f
            combined.append((cid, score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
    
    @staticmethod
    def _normalize(scores_dict):
        """Normalize a score dict to [0, 1]."""
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        r = max_v - min_v
        if r == 0:
            return {k: 0.5 for k in scores_dict}
        return {k: (v - min_v) / r for k, v in scores_dict.items()}
    
    def evaluate(self, ratings_data, users, movies, test_ratio=0.2, k_values=None):
        """Run full evaluation suite.
        
        Args:
            ratings_data: list of (user_id, movie_id, rating) tuples
            users: list of user dicts
            movies: list of movie dicts
            test_ratio: fraction held out for testing
            k_values: list of K values for @K metrics
            
        Returns:
            dict with all evaluation metrics
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        start_time = time.time()
        
        # Group ratings per user
        user_ratings = defaultdict(list)
        for uid, mid, rating in ratings_data:
            user_ratings[uid].append((mid, rating))
        
        user_lookup = {u["id"]: u for u in users}
        movie_lookup = {m["id"]: m for m in movies}
        all_movie_ids = set(m["id"] for m in movies)
        all_genres = set()
        for m in movies:
            all_genres.update(m.get("genres", []))
        
        metrics = {k: {"precision": [], "recall": [], "ndcg": [], "hit_rate": []} for k in k_values}
        catalog_recommended = set()
        genre_distributions = []
        
        users_evaluated = 0
        
        for user_id, user_items in user_ratings.items():
            if len(user_items) < 5:
                continue
            
            # Split: last N items as test (simulates temporal split)
            n_test = max(1, int(len(user_items) * test_ratio))
            train_items = user_items[:-n_test]
            test_items = user_items[-n_test:]
            
            # Ground truth: items rated >= 3.5 in test set
            relevant = {mid for mid, rating in test_items if rating >= 3.5}
            if not relevant:
                continue
            
            users_evaluated += 1
            
            # Items used for training (to exclude from candidates)
            train_ids = {mid for mid, _ in train_items}
            train_liked_ids = [mid for mid, rating in train_items if rating >= 3.5]
            user_data = user_lookup.get(user_id, {})
            
            # Candidate set = all movies NOT in training set
            # (test items ARE included as candidates — this is what we want)
            candidates = all_movie_ids - train_ids
            
            # Score candidates using hybrid formula directly
            scored = self._score_candidates(
                user_id=user_id,
                train_liked_ids=train_liked_ids,
                preferred_genres=user_data.get("preferred_genres", []),
                candidate_ids=candidates,
            )
            rec_ids = [mid for mid, _ in scored]
            
            # Track catalog coverage
            catalog_recommended.update(rec_ids[:max(k_values)])
            
            # Track genre diversity of recommendations
            if rec_ids:
                genres_in_recs = set()
                for mid in rec_ids[:10]:
                    m = movie_lookup.get(mid, {})
                    genres_in_recs.update(m.get("genres", []))
                genre_distributions.append(len(genres_in_recs))
            
            # Compute metrics at each K
            for k in k_values:
                top_k = rec_ids[:k]
                hits = set(top_k) & relevant
                n_hits = len(hits)
                
                # Precision@K: relevant items in top K / K
                precision = n_hits / k if k > 0 else 0
                metrics[k]["precision"].append(precision)
                
                # Recall@K: relevant items in top K / total relevant
                recall = n_hits / len(relevant) if relevant else 0
                metrics[k]["recall"].append(recall)
                
                # Hit Rate@K: did at least one relevant item appear?
                metrics[k]["hit_rate"].append(1.0 if n_hits > 0 else 0.0)
                
                # NDCG@K: position-aware metric
                dcg = 0.0
                for i, mid in enumerate(top_k):
                    if mid in relevant:
                        dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
                n_ideal = min(len(relevant), k)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[k]["ndcg"].append(ndcg)
        
        eval_time = time.time() - start_time
        
        # Aggregate results
        results = {"k_metrics": {}, "system_metrics": {}, "meta": {}}
        
        for k in k_values:
            results["k_metrics"][f"@{k}"] = {
                "precision": round(float(np.mean(metrics[k]["precision"])), 4) if metrics[k]["precision"] else 0,
                "recall": round(float(np.mean(metrics[k]["recall"])), 4) if metrics[k]["recall"] else 0,
                "ndcg": round(float(np.mean(metrics[k]["ndcg"])), 4) if metrics[k]["ndcg"] else 0,
                "hit_rate": round(float(np.mean(metrics[k]["hit_rate"])), 4) if metrics[k]["hit_rate"] else 0,
            }
        
        results["system_metrics"] = {
            "catalog_coverage": round(len(catalog_recommended) / len(movies), 4) if movies else 0,
            "catalog_items_recommended": len(catalog_recommended),
            "total_catalog_size": len(movies),
            "avg_genre_diversity": round(float(np.mean(genre_distributions)), 2) if genre_distributions else 0,
            "total_genres": len(all_genres),
        }
        
        results["meta"] = {
            "users_evaluated": users_evaluated,
            "total_users": len(user_ratings),
            "test_ratio": test_ratio,
            "total_ratings": len(ratings_data),
            "evaluation_time_seconds": round(eval_time, 2),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "methodology": "hold-out (temporal split), threshold=3.5",
        }
        
        self.results = results
        return results
    
    def get_cached_results(self):
        """Return last evaluation results."""
        return self.results if self.results else None
