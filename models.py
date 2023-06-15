from pydantic import BaseModel
from typing import Optional, List

class DestinationRecommendationAttribute(BaseModel):
    city: str
    user_destination_preferences: List[str]
    user_restaurant_preferences: List[str]