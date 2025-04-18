from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import shutil
import os
import json
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

app = FastAPI(title="SkillSwap API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ------------------ In-Memory Storage ------------------
users: Dict[str, dict] = {}
transactions: Dict[str, dict] = {}

# ------------------ Password Hashing ------------------
ph = PasswordHasher()

# ------------------ Models ------------------
class TimeSlot(BaseModel):
    day: str
    start: str
    end: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    gender: str = None
    age: Optional[int] = None
    pronouns: Optional[str] = None
    bio: Optional[str] = None
    can_teach_list: Optional[List[int]] = None
    wants_to_learn_list: Optional[List[int]] = None
    teaching_style_list: Optional[List[int]] = None
    learning_preference_list: Optional[List[int]] = None
    can_teach: Optional[str] = None
    wants_to_learn: Optional[str] = None
    teaching_style: Optional[str] = None
    learning_preference: Optional[str] = None
    timezone: Optional[str] = None
    languages: Optional[List[int]] = None
    availability: Optional[List[TimeSlot]] = None

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    gender: str = None
    age: Optional[int] = None
    pronouns: Optional[str] = None
    bio: Optional[str] = None
    can_teach_list: Optional[List[int]] = None
    wants_to_learn_list: Optional[List[int]] = None
    teaching_style_list: Optional[List[int]] = None
    learning_preference_list: Optional[List[int]] = None
    can_teach: Optional[str] = None
    wants_to_learn: Optional[str] = None
    teaching_style: Optional[str] = None
    learning_preference: Optional[str] = None
    timezone: Optional[str] = None
    languages: Optional[List[int]] = None
    availability: Optional[List[TimeSlot]] = None
    score: float = 5.0
    image_path: Optional[str] = None
    num_transactions: int = 0
    active_transactions: int = 0

class LoginResponse(BaseModel):
    user_id: str
    username: str
    success: bool

class TransactionCreate(BaseModel):
    teacher_id: str
    learner_id: str
    skill_taught: str
    skill_learned: str

class TransactionResponse(BaseModel):
    transaction_id: str
    teacher_id: str
    learner_id: str
    skill_taught: str
    skill_learned: str
    status: str
    teacher_confirmed: bool
    learner_confirmed: bool
    teacher_rating: Optional[float] = None
    learner_rating: Optional[float] = None
    created_at: datetime

class MatchResponse(BaseModel):
    user_id: str
    username: str
    score: float
    image_path: Optional[str] = None
    matching_skills: List[int]
    can_teach_list: Optional[List[int]] = None
    wants_to_learn_list: Optional[List[int]] = None

# ------------------ Helper Functions ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using Sentence Transformer
    """

    res = model.encode(text).tolist() 

    return res if res != None else [1, 1, 1, 0, 0, 0 , 1]

def calculate_profile_embedding(user_data: dict) -> List[float]:
    """
    Calculate a comprehensive embedding for a user by combining various profile fields
    """
    combined_text = " ".join([
    str(user_data.get("can_teach") or ""),
    str(user_data.get("wants_to_learn") or ""),
    str(user_data.get("teaching_style") or ""),
    str(user_data.get("learning_preference") or ""),
    str(user_data.get("bio") or "")
    ]).strip()
    return get_embedding(combined_text)

def calculate_availability_overlap(user1_avail, user2_avail):
    """
    Calculate overlapping availability between two users
    """
    overlap = []
    for slot1 in user1_avail or []:
        for slot2 in user2_avail or []:
            if (slot1["day"] == slot2["day"] and
                slot1["start"] < slot2["end"] and
                slot1["end"] > slot2["start"]):
                overlap.append({
                    "day": slot1["day"],
                    "start": max(slot1["start"], slot2["start"]),
                    "end": min(slot1["end"], slot2["end"])
                })
    return overlap

def calculate_match_score(user, other, availability_overlap, skill_overlap):
    """
    Calculate match score between two users based on skill overlap, embedding similarity, and availability
    """
    teach_to_learn_sim = cosine_similarity(
        [user.get("teach_embedding", [])],
        [other.get("learn_embedding", [])]
    )[0][0]

    learn_to_teach_sim = cosine_similarity(
        [user.get("learn_embedding", [])],
        [other.get("teach_embedding", [])]
    )[0][0]

    cross_score = (teach_to_learn_sim + learn_to_teach_sim) / 2
    availability_score = min(len(availability_overlap) / 5, 1.0)
    return round(0.5 * cross_score + 0.3 * (skill_overlap/5) + 0.2 * availability_score, 2)

def get_user_by_username(username: str):
    """
    Find a user by username
    """
    for user in users.values():
        if user["username"] == username:
            return user
    return None

def verify_password(plain_password, hashed_password):
    """
    Verify a password against its hash
    """
    try:
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False

def get_model_data(model):
    """
    Get data from a Pydantic model, compatible with both v1 and v2
    """
    if hasattr(model, 'model_dump'):
        return model.model_dump()
    return model.dict()

# ------------------ API Endpoints ------------------
@app.post("/login", response_model=LoginResponse)
def login_user(user_data: UserLogin):
    """
    Authenticate a user and return their user ID
    """
    user = get_user_by_username(user_data.username)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "success": True
    }

@app.post("/register", response_model=UserProfile, status_code=201)
def register_user(
    user_data: str = Form(...),
    file: UploadFile = File(None)
):
    """
    Register a new user
    """
    try:
        # Parse the JSON string into a dict
        user_dict = json.loads(user_data)
        
        # Convert to Pydantic model
        user_create = UserCreate(**user_dict)
        
        # Check existing users
        if any(u["username"] == user_create.username for u in users.values()):
            raise HTTPException(400, "Username exists")

        user_id = str(uuid4())
        image_path = None

        # Handle file upload
        if file and file.content_type in ["image/jpeg", "image/png"]:
            file_ext = os.path.splitext(file.filename)[1]
            image_path = f"static/images/{user_id}{file_ext}"
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Get model data in a way that works with both Pydantic v1 and v2
        user_data_dict = get_model_data(user_create)
        
        # Create user
        user = {
            "user_id": user_id,
            **user_data_dict,
            "password_hash": ph.hash(user_create.password),
            "score": 5.0,
            "image_path": image_path,
            "teach_embedding": get_embedding(user_create.can_teach),
            "learn_embedding": get_embedding(user_create.wants_to_learn),
            "profile_embedding": calculate_profile_embedding(user_data_dict),
            "num_transactions": 0,
            "active_transactions": 0,
            "created_at": datetime.utcnow()
        }

        users[user_id] = user
        return user
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in user_data field")
    except Exception as e:
        print(str(e))
        raise HTTPException(500, f"Registration failed: {str(e)}")


@app.get("/users/{user_id}", response_model=UserProfile)
def get_user(user_id: str):
    """
    Get a user's profile by user ID
    """
    user = users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    # Calculate active transactions
    user["active_transactions"] = sum(
        1 for t in transactions.values()
        if (t["teacher_id"] == user_id or t["learner_id"] == user_id)
        and t["status"] in ["pending", "ongoing"]
    )

    return user

@app.put("/users/{user_id}", response_model=UserProfile)
def update_user(user_id: str, user_data: dict):
    """
    Update a user's profile
    """
    user = users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    # Update user fields
    for key, value in user_data.items():
        if key not in ["user_id", "password_hash", "score", "num_transactions"]:
            user[key] = value
    
    # Recalculate embeddings if relevant fields changed
    if any(key in user_data for key in ["can_teach", "wants_to_learn", "teaching_style", "learning_preference", "bio"]):
        user["teach_embedding"] = get_embedding(user.get("can_teach", ""))
        user["learn_embedding"] = get_embedding(user.get("wants_to_learn", ""))
        user["profile_embedding"] = calculate_profile_embedding(user)
    
    return user

@app.post("/transactions", status_code=201, response_model=TransactionResponse)
def create_transaction(transaction_data: TransactionCreate):
    """
    Create a new skill exchange transaction
    """
    teacher = users.get(transaction_data.teacher_id)
    learner = users.get(transaction_data.learner_id)

    if not teacher or not learner:
        raise HTTPException(404, "User not found")

    transaction_id = str(uuid4())
    
    # Get model data in a way that works with both Pydantic v1 and v2
    transaction_dict = get_model_data(transaction_data)
    
    transaction = {
        "transaction_id": transaction_id,
        **transaction_dict,
        "status": "pending",
        "teacher_confirmed": False,
        "learner_confirmed": False,
        "created_at": datetime.utcnow()
    }
    
    transactions[transaction_id] = transaction
    return transaction

@app.get("/transactions", response_model=List[TransactionResponse])
def get_user_transactions(user_id: str):
    """
    Get all transactions for a user
    """
    user_transactions = [
        t for t in transactions.values()
        if t["teacher_id"] == user_id or t["learner_id"] == user_id
    ]
    
    return user_transactions

@app.post("/transactions/{transaction_id}/confirm")
def confirm_transaction(transaction_id: str, user_id: str, rating: float = Form(...)):
    """
    Confirm completion of a transaction and provide rating
    """
    transaction = transactions.get(transaction_id)
    if not transaction:
        raise HTTPException(404, "Transaction not found")

    # Update confirmation status
    if user_id == transaction["teacher_id"]:
        transaction["teacher_confirmed"] = True
        transaction["learner_rating"] = rating
    elif user_id == transaction["learner_id"]:
        transaction["learner_confirmed"] = True
        transaction["teacher_rating"] = rating
    else:
        raise HTTPException(403, "Not part of transaction")

    # Complete transaction if both confirmed
    if transaction["teacher_confirmed"] and transaction["learner_confirmed"]:
        transaction["status"] = "completed"
        transaction["completed_at"] = datetime.utcnow()

        # Update user scores
        teacher = users[transaction["teacher_id"]]
        learner = users[transaction["learner_id"]]

        teacher["score"] = round(
            ((teacher["score"] * teacher["num_transactions"]) + transaction["teacher_rating"]) /
            (teacher["num_transactions"] + 1), 2
        )
        teacher["num_transactions"] += 1

        learner["score"] = round(
            ((learner["score"] * learner["num_transactions"]) + transaction["learner_rating"]) /
            (learner["num_transactions"] + 1), 2
        )
        learner["num_transactions"] += 1

    return {"status": transaction["status"]}

@app.get("/matches/{user_id}", response_model=List[MatchResponse])
def get_matches(user_id: str, limit: int = 10):
    """
    Find potential skill exchange matches for a user
    """
    user = users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    matches = []
    for other in users.values():
        if other["user_id"] == user_id:
            continue
            
        # Calculate skill overlap
        skill_overlap = len(set(user.get("wants_to_learn_list", [])) & set(other.get("can_teach_list", [])))
        
        # Skip if no skill overlap
        if skill_overlap == 0:
            continue
            
        # Calculate availability overlap
        availability_overlap = calculate_availability_overlap(
            user.get("availability", []),
            other.get("availability", [])
        )

        # Calculate match score
        score = calculate_match_score(
            user, other, availability_overlap, skill_overlap
        )

        matches.append({
            "user_id": other["user_id"],
            "username": other["username"],
            "score": score,
            "image_path": other.get("image_path"),
            "matching_skills": list(set(user.get("wants_to_learn_list", [])) & set(other.get("can_teach_list", []))),
            "can_teach_list": other.get("can_teach_list", []),
            "wants_to_learn_list": other.get("wants_to_learn_list", [])
        })

    # Sort by match score (highest first) and limit results
    return sorted(matches, key=lambda x: x["score"], reverse=True)[:limit]

@app.get("/feed/{user_id}")
def get_feed(user_id: str, limit: int = 10):
    """
    Get a personalized feed of activity for a user
    """
    user = users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    # Get recent completed transactions
    recent_transactions = [
        t for t in transactions.values()
        if t["status"] == "completed"
    ]
    recent_transactions.sort(key=lambda x: x.get("completed_at", x["created_at"]), reverse=True)
    recent_transactions = recent_transactions[:5]
    
    # Get potential teachers
    potential_teachers = []
    for other in users.values():
        if other["user_id"] == user_id:
            continue
            
        # Check if they can teach what user wants to learn
        matching_skills = set(user.get("wants_to_learn_list", [])) & set(other.get("can_teach_list", []))
        if matching_skills:
            potential_teachers.append({
                "type": "potential_teacher",
                "user_id": other["user_id"],
                "username": other["username"],
                "image_path": other.get("image_path"),
                "matching_skills": list(matching_skills),
                "rating": other.get("score", 5.0)
            })
    
    # Sort potential teachers by rating
    potential_teachers.sort(key=lambda x: x["rating"], reverse=True)
    potential_teachers = potential_teachers[:5]
    
    # Combine feed items
    feed_items = []
    
    # Add transactions to feed
    for t in recent_transactions:
        teacher = users.get(t["teacher_id"], {})
        learner = users.get(t["learner_id"], {})
        
        feed_items.append({
            "type": "transaction",
            "transaction_id": t["transaction_id"],
            "teacher": {
                "user_id": teacher.get("user_id"),
                "username": teacher.get("username"),
                "image_path": teacher.get("image_path")
            },
            "learner": {
                "user_id": learner.get("user_id"),
                "username": learner.get("username"),
                "image_path": learner.get("image_path")
            },
            "skill_taught": t["skill_taught"],
            "skill_learned": t["skill_learned"],
            "timestamp": t.get("completed_at", t["created_at"])
        })
    
    # Add potential teachers to feed
    feed_items.extend(potential_teachers)
    
    # Sort by recency for transactions and rating for recommendations
    feed_items = sorted(feed_items, 
                        key=lambda x: x.get("timestamp", datetime.utcnow()) if x["type"] == "transaction" else datetime.utcnow(),
                        reverse=True)
    
    return feed_items[:limit] if feed_items else []

@app.get("/search/{user_id}")
def search_skills(user_id: str, query: str, limit: int = 10):
    """
    Search for users by skill
    """
    user = users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    
    if not query.strip():
        raise HTTPException(400, "Search query cannot be empty")
    
    # Generate embedding for search query
    query_embedding = get_embedding(query)
    
    results = []
    for other in users.values():
        if other["user_id"] == user_id:
            continue
        
        # Calculate similarity with teach embedding
        teach_similarity = 0
        if "teach_embedding" in other:
            teach_similarity = cosine_similarity(
                [query_embedding],
                [other["teach_embedding"]]
            )[0][0]
        
        # Only include if similarity is above threshold
        if teach_similarity > 0.3:
            results.append({
                "user_id": other["user_id"],
                "username": other["username"],
                "image_path": other.get("image_path"),
                "relevance": round(teach_similarity, 2),
                "can_teach": other.get("can_teach", ""),
                "can_teach_list": other.get("can_teach_list", [])
            })
    
    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:limit]

# ------------------ Persistence Functions ------------------
def save_data():
    """
    Save in-memory data to JSON files
    """
    # Convert datetime objects to strings for JSON serialization
    users_json = {}
    for user_id, user in users.items():
        user_copy = user.copy()
        if "created_at" in user_copy:
            user_copy["created_at"] = user_copy["created_at"].isoformat()
        users_json[user_id] = user_copy
    
    transactions_json = {}
    for transaction_id, transaction in transactions.items():
        transaction_copy = transaction.copy()
        if "created_at" in transaction_copy:
            transaction_copy["created_at"] = transaction_copy["created_at"].isoformat()
        if "completed_at" in transaction_copy:
            transaction_copy["completed_at"] = transaction_copy["completed_at"].isoformat()
        transactions_json[transaction_id] = transaction_copy
    
    # Save to files
    os.makedirs("data", exist_ok=True)
    with open("data/users.json", "w") as f:
        json.dump(users_json, f)
    
    with open("data/transactions.json", "w") as f:
        json.dump(transactions_json, f)

def load_data():
    """
    Load data from JSON files into memory
    """
    global users, transactions
    
    try:
        if os.path.exists("data/users.json"):
            with open("data/users.json", "r") as f:
                users_json = json.load(f)
                
            # Convert string dates back to datetime objects
            for user_id, user in users_json.items():
                if "created_at" in user:
                    user["created_at"] = datetime.fromisoformat(user["created_at"])
            
            users = users_json
        
        if os.path.exists("data/transactions.json"):
            with open("data/transactions.json", "r") as f:
                transactions_json = json.load(f)
                
            # Convert string dates back to datetime objects
            for transaction_id, transaction in transactions_json.items():
                if "created_at" in transaction:
                    transaction["created_at"] = datetime.fromisoformat(transaction["created_at"])
                if "completed_at" in transaction:
                    transaction["completed_at"] = datetime.fromisoformat(transaction["completed_at"])
            
            transactions = transactions_json
    except Exception as e:
        print(f"Error loading data: {e}")

# Load data at startup
load_data()

# Save data periodically (in a production app, this would be done differently)
@app.on_event("shutdown")
def shutdown_event():
    save_data()