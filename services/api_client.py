# FastAPI calls
import requests

BACKEND_URL = "http://127.0.0.1:8000"


def create_ticket(description: str):
    response = requests.post(
        f"{BACKEND_URL}/generate-ticket",
        json={"description": description}
    )
    return response
