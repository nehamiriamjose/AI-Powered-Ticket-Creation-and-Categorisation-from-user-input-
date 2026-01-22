from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid

# DB
from backend.db import create_tables, insert_ticket, fetch_all_tickets

# Notifications
from services.notification_service import add_notification

# ML & NER
from scripts.milestone3_ticket_generation_engine import (
    preprocess_text,
    extract_entities,
    refine_priority,
    CATEGORY_MAP,
    PRIORITY_MAP,
)
from scripts.milestone3_model_inference import predict_category_priority

app = FastAPI(title="AI Ticket Management API")
create_tables()


# ---------------- MODELS ----------------
class TicketRequest(BaseModel):
    description: str


class TicketResponse(BaseModel):
    ticket_id: str
    title: str
    description: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    entities: dict
    status: str
    created_at: str


# ---------------- API ----------------
@app.post("/generate-ticket", response_model=TicketResponse)
def generate_ticket(request: TicketRequest):

    text = request.description.strip()
    if not text:
        raise HTTPException(400, "Description cannot be empty")

    clean_text = preprocess_text(text)

    # ✅ ALWAYS returns 4 values
    category_id, priority_id, cat_conf, pr_conf = predict_category_priority(clean_text)

    category = CATEGORY_MAP.get(category_id, "Unknown")
    ml_priority = PRIORITY_MAP.get(priority_id, "Medium")
    final_priority = refine_priority(text, ml_priority)

    entities = extract_entities(clean_text)

    ticket_data = {
        "ticket_id": f"INC-{uuid.uuid4().hex[:6].upper()}",
        "title": f"{category} Issue",
        "description": text,
        "category": category,
        "category_confidence": round(float(cat_conf), 4),
        "priority": final_priority,
        "priority_confidence": round(float(pr_conf), 4),
        "entities": entities,
        "status": "Open",
        "created_at": datetime.now().isoformat(),
    }

    # ✅ DB insert WITHOUT entities (schema-safe)
    insert_ticket(
        title=ticket_data["title"],
        description=ticket_data["description"],
        category=ticket_data["category"],
        priority=ticket_data["priority"],
    )

    if final_priority == "High":
        add_notification(f"🚨 High priority ticket: {ticket_data['title']}", "alert")

    return ticket_data


@app.get("/tickets")
def get_tickets():
    rows = fetch_all_tickets()
    return [
        {
            "id": r[0],
            "title": r[1],
            "description": r[2],
            "category": r[3],
            "priority": r[4],
            "status": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


@app.get("/health")
def health():
    return {"status": "OK"}
