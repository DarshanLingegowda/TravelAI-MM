from pydantic import BaseModel
from typing import Dict, List

class PlanResponse(BaseModel):
    itinerary: str
    citations: List[Dict]
    num_sources: int
    modality_breakdown: Dict[str, int]

