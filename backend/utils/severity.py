"""Utility to estimate disease severity stage for AgroVeyra."""


def estimate_stage(confidence: float, disease_name: str) -> dict:
    """Estimate disease severity stage based on confidence score and disease name.

    Args:
        confidence: Confidence score from 0.0 to 1.0.
        disease_name: PlantVillage class name string (e.g., "Tomato___Early_blight").

    Returns:
        dict with keys:
            - stage (str): Human readable stage description.
            - stage_number (int): Numeric stage (0 = healthy, 1-3 = severity).
            - estimated_percentage (int): Estimated disease progression percentage.
            - urgency (str): Recommended action urgency.
    """
    try:
        if "healthy" in disease_name.lower():
            return {
                "stage": "Healthy",
                "stage_number": 0,
                "estimated_percentage": 0,
                "urgency": "Monitor",
            }

        if confidence < 0.60:
            return {
                "stage": "Stage 1 - Early",
                "stage_number": 1,
                "estimated_percentage": 15,
                "urgency": "Monitor",
            }
        elif confidence <= 0.85:
            return {
                "stage": "Stage 2 - Moderate",
                "stage_number": 2,
                "estimated_percentage": 40,
                "urgency": "Treat within 5 days",
            }
        else:
            return {
                "stage": "Stage 3 - Severe",
                "stage_number": 3,
                "estimated_percentage": 75,
                "urgency": "Treat immediately",
            }
    except Exception as e:
        print(f"[severity.py] Error estimating stage: {e}")
        return {
            "stage": "Unknown",
            "stage_number": 0,
            "estimated_percentage": 0,
            "urgency": "Monitor",
        }


