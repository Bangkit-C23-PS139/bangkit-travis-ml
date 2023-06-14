from pydantic import BaseModel
from typing import Optional, List

class DestinationRecommendationAttribute(BaseModel):
    city: str
    user_preferences: List[str]