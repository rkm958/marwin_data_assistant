# chat_memory.py – Simple JSON‑file conversation memory for Marwin
"""
Purpose (layman‑friendly)
=========================
This module lets Marwin **remember past questions and answers** between a user and the RAG assistant.
Instead of a database we start with a lightweight **JSON file** that stores each turn like this:

```jsonc
[
  {
    "id": "8e2a6e2e-37c1-4d3f-b51e-c8e8dbe74f1f",
    "user_id": "843920",
    "timestamp": "2025-07-02T10:15:00Z",
    "query": "Where can I find application approvals?",
    "answer": "Use fact_applications.outcome_id…"
  },
  {
    "id": "bb2e2b2b-74f4-4a3e-a857-b6a8f00cd05e",
    "user_id": "843920",
    "timestamp": "2025-07-02T10:16:30Z",
    "query": "What about rejections?",
    "answer": "Use the same table, outcome_id = 'D'…"
  }
]
```

Core Functions
--------------
1. **load_memory(file_path)**
   • Loads the JSON list into Python (returns `list[dict]`).
   • Returns an empty list if file doesn’t exist.

2. **append_to_memory(query, answer, file_path)**
   • Adds a new `{id, user_id, timestamp, query, answer}` dict to memory and saves JSON.

3. **get_recent_memory(n, file_path)**
   • Returns the *n* most recent turns (default 5) for prompt context.

4. **clear_memory(file_path)**
   • Wipes the JSON file (useful for a "Reset conversation" button).
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict

MEMORY_FILE = "chat_memory.json"  # default path; override per‑user in future
DEFAULT_USER_ID = "843920"         # temp user id placeholder

# -----------------------------------------------------------------------------
# Helper: safe file read/write
# -----------------------------------------------------------------------------

def read_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf‑8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []  # corrupt or empty file → start fresh

def write_json(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf‑8") as f:
        json.dump(data, f, indent=2)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_memory(file_path: str = MEMORY_FILE) -> List[Dict]:
    """Load entire chat history from JSON. Empty list if none."""
    return read_json(file_path)


def append_to_memory(query: str, answer: str, top_matches: List[Dict], file_path: str = MEMORY_FILE):
    """
    Add a new memory entry including query, response, top vector matches, and metadata.
    """
    memory = read_json(file_path)
    memory.append({
        "id": str(uuid.uuid4()),
        "user_id": "823471",  # replace with real user ID logic later
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "matches": top_matches,  # top 5 matches with distance + metadata
        "feedback_type": None,  # 🛠️ Default placeholders
        "comment": None         # 🛠️ Default placeholders        
    })
    write_json(file_path, memory)
    
#-----------------------------------------------

def get_recent_memory(n: int = 10, file_path: str = MEMORY_FILE) -> List[Dict]:
    """Return the *n* most recent turns for prompt context."""
    history = read_json(file_path)
    return history[-n:]
    
#-----------------------------------------

def clear_memory(file_path: str = MEMORY_FILE):
    """Delete all stored conversation turns (reset)."""
    write_json(file_path, [])
    print("🗑️  Chat memory cleared.")
    
#----------------------------------------------------------------------------------

def update_feedback_to_memory(
    query: str,
    answer: str,
    feedback_type: str,
    comment: str = None,
    file_path: str = MEMORY_FILE
):
    """
    Locate the memory record using query and answer, and update it with feedback type and optional comment.

    Args:
        query (str): The user's original question
        answer (str): The LLM-generated answer
        feedback_type (str): 'like' or 'dislike'
        comment (str, optional): Optional feedback comment from user
        file_path (str): Path to the chat memory JSON
    """
    memory = read_json(file_path)
    updated = False

    for item in memory:
        if item.get("query") == query and item.get("answer") == answer:
            item["feedback_type"] = feedback_type
            item["comment"] = comment
            updated = True
            break

    if updated:
        write_json(file_path, memory)
        print(f"✅ Feedback updated for: {query}")
    else:
        print("⚠️ No matching record found to update feedback.")


# -----------------------------------------------------------------------------
# Self‑test (only runs if you execute python chat_memory.py)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    clear_memory()  # start clean
    append_to_memory("Where can I find approvals?", "Use fact_applications…")
    append_to_memory("What about rejections?", "Same table, outcome_id = 'D'…")
    print("Recent memory:\n", get_recent_memory())
