"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class RatingCreate(BaseModel):
    """Request model for submitting a rating."""
    user_id: int = Field(..., ge=1, description="User ID")
    movie_id: int = Field(..., ge=1, description="Movie ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating (1-5)")


class EventCreate(BaseModel):
    """Request model for tracking user behavior events."""
    user_id: int = Field(..., ge=1, description="User ID")
    movie_id: Optional[int] = Field(None, ge=1, description="Movie ID (optional)")
    event_type: str = Field(..., description="Event type (view, click, search, etc.)")


class RecommendationRequest(BaseModel):
    """Request model for custom recommendation parameters."""
    n: int = Field(20, ge=1, le=100, description="Number of recommendations")


class MovieResponse(BaseModel):
    """Response model for a movie."""
    id: int
    title: str
    year: int
    genres: list[str]
    description: str
    runtime: int
    director: str
    cast: list[str]
    gradient_start: str
    gradient_end: str
    avg_rating: float
    rating_count: int
    view_count: int


class UserResponse(BaseModel):
    """Response model for a user."""
    id: int
    name: str
    preferred_genres: list[str]
    avatar_color: str
