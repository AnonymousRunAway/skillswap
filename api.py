from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from uuid import uuid4
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import os
from pathlib import Path
import shutil

app = FastAPI(title="SkillSwap API", version="1.0.0")

# Enable CORS for all origins (for hackathon speed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ DB Setup ------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://skillswapuser:skillswappass@localhost:5432/skillswap")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ------------------ Password Hashing ------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ------------------ Database Models ------------------
class UserDB(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    gender = Column(JSON, nullable=True)
    age = Column(Integer, nullable=True)
    pronouns = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    can_teach_list = Column(JSON, nullable=True)
    wants_to_learn_list = Column(JSON, nullable=True)
    teaching_style_list = Column(JSON, nullable=True)
    learning_preference_list = Column(JSON, nullable=True)
    can_teach = Column(Text, nullable=True)
    wants_to_learn = Column(Text, nullable=True)
    teaching_style = Column(Text, nullable=True)
    learning_preference = Column(Text, nullable=True)
    timezone = Column(String, nullable=True)
    languages = Column(JSON, nullable=True)
    availability = Column(JSON, nullable=True)
    score = Column(Float, default=5.0)
    image_path = Column(String, nullable=True)
    teach_embedding = Column(JSON, nullable=True)
    learn_embedding = Column(JSON, nullable=True)
    profile_embedding = Column(JSON, nullable=True)
    num_transactions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TransactionDB(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String, primary_key=True, index=True)
    teacher_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    learner_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    skill_taught = Column(String, nullable=False)
    skill_learned = Column(String, nullable=False)
    status = Column(String, default="pending")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    teacher_confirmed = Column(Boolean, default=False)
    learner_confirmed = Column(Boolean, default=False)
    teacher_rating = Column(Float, nullable=True)
    learner_rating = Column(Float, nullable=True)
    teacher_feedback = Column(Text, nullable=True)
    learner_feedback = Column(Text, nullable=True)
    scheduled_sessions = Column(JSON, nullable=True)

Base.metadata.create_all(bind=engine)

# ------------------ Pydantic Models ------------------
class TimeSlot(BaseModel):
    day: str = Field(..., example="Monday")
    start: str = Field(..., regex="^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", example="09:00")
    end: str = Field(..., regex="^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", example="17:00")

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8)
    gender: Optional[List[int]] = None
    age: Optional[int] = Field(None, ge=13, le=120)
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
    gender: Optional[List[int]] = None
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
    score: float = Field(5.0, ge=0, le=5)
    image_path: Optional[str] = None
    num_transactions: int = Field(0, ge=0)
    active_transactions: int = Field(0, ge=0)
    created_at: datetime
    updated_at: datetime

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
    started_at: datetime
    completed_at: Optional[datetime] = None
    teacher_confirmed: bool
    learner_confirmed: bool
    teacher_rating: Optional[float] = None
    learner_rating: Optional[float] = None
    teacher_feedback: Optional[str] = None
    learner_feedback: Optional[str] = None

class MatchResult(BaseModel):
    user_id: str
    username: str
    score: float
    matching_skills: List[str]
    availability_overlap: List[TimeSlot]
    explanation: str

# ------------------ Helper Functions ------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_embedding(text: str) -> List[float]:
    if not text:
        return []
    return model.encode(text).tolist()

def calculate_profile_embedding(user: UserCreate) -> List[float]:
    combined_text = " ".join([
        user.can_teach or "",
        user.wants_to_learn or "",
        user.teaching_style or "",
        user.learning_preference or "",
        user.bio or ""
    ]).strip()
    return get_embedding(combined_text)

def split_into_hour_slots(timeslot: TimeSlot) -> List[Dict[str, str]]:
    try:
        start_time = datetime.strptime(timeslot.start, "%H:%M")
        end_time = datetime.strptime(timeslot.end, "%H:%M")
    except ValueError:
        return []
    if start_time >= end_time:
        return []
    current_time = start_time
    slots = []
    while current_time + timedelta(hours=1) <= end_time:
        next_time = current_time + timedelta(hours=1)
        slots.append({
            "day": timeslot.day,
            "start": current_time.strftime("%H:%M"),
            "end": next_time.strftime("%H:%M")
        })
        current_time = next_time
    return slots

def normalize_availability(availability: List[TimeSlot]) -> List[Dict[str, str]]:
    normalized = []
    for slot in availability or []:
        normalized.extend(split_into_hour_slots(slot))
    return normalized

def calculate_availability_overlap(user1_avail: List[Dict], user2_avail: List[Dict]) -> List[Dict]:
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

def calculate_skill_overlap(user1_teach: List[int], user2_learn: List[int]) -> int:
    if not user1_teach or not user2_learn:
        return 0
    return len(set(user1_teach) & set(user2_learn))

def calculate_match_score(
    user: UserDB,
    other: UserDB,
    availability_overlap: List[Dict],
    skill_overlap: int
) -> float:
    profile_similarity = cosine_similarity(
        [user.profile_embedding or []],
        [other.profile_embedding or []]
    )[0][0] if user.profile_embedding and other.profile_embedding else 0

    teach_to_learn_sim = cosine_similarity(
        [user.teach_embedding or []],
        [other.learn_embedding or []]
    )[0][0] if user.teach_embedding and other.learn_embedding else 0

    learn_to_teach_sim = cosine_similarity(
        [user.learn_embedding or []],
        [other.teach_embedding or []]
    )[0][0] if user.learn_embedding and other.teach_embedding else 0

    cross_embedding_score = (teach_to_learn_sim + learn_to_teach_sim) / 2

    normalized_skill_overlap = min(skill_overlap / 5, 1.0)
    availability_score = min(len(availability_overlap) / 5, 1.0)
    language_score = len(set(user.languages or []) & set(other.languages or [])) / max(1, len(set(user.languages or [])))
    rating_score = other.score / 5.0

    final_score = (
        0.25 * profile_similarity +
        0.20 * cross_embedding_score +
        0.20 * normalized_skill_overlap +
        0.15 * availability_score +
        0.10 * language_score +
        0.10 * rating_score
    )
    return round(final_score, 4)

def generate_match_explanation(
    user: UserDB,
    other: UserDB,
    matching_skills: List[str],
    availability_overlap: List[Dict],
    score: float
) -> str:
    base_explanation = f"{other.username} is a great match (score: {score:.2f}/1.0) because:"
    skill_explanation = ""
    if matching_skills:
        skill_explanation = f" they can teach you {', '.join(map(str, matching_skills[:3]))}"
        if len(matching_skills) > 3:
            skill_explanation += f" and {len(matching_skills)-3} more skills"
    availability_explanation = ""
    if availability_overlap:
        days = {slot['day'] for slot in availability_overlap}
        availability_explanation = f" and is available on {', '.join(sorted(days))}"
    rating_explanation = ""
    if other.score >= 4.5:
        rating_explanation = " They have excellent ratings from previous exchanges."
    elif other.score >= 4.0:
        rating_explanation = " They have very good ratings from previous exchanges."
    return f"{base_explanation}{skill_explanation}{availability_explanation}.{rating_explanation}"

# ------------------ API Endpoints ------------------

@app.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    if db.query(UserDB).filter(UserDB.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(UserDB).filter(UserDB.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    teach_embedding = get_embedding(user_data.can_teach)
    learn_embedding = get_embedding(user_data.wants_to_learn)
    profile_embedding = calculate_profile_embedding(user_data)
    normalized_availability = normalize_availability(user_data.availability)
    image_path = None
    if file:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed")
        try:
            file_ext = os.path.splitext(file.filename)[1]
            user_id = str(uuid4())
            file_location = f"static/images/{user_id}{file_ext}"
            Path(file_location).parent.mkdir(parents=True, exist_ok=True)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_path = file_location
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    user_db = UserDB(
        user_id=str(uuid4()),
        username=user_data.username,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        gender=user_data.gender,
        age=user_data.age,
        pronouns=user_data.pronouns,
        bio=user_data.bio,
        can_teach_list=user_data.can_teach_list,
        wants_to_learn_list=user_data.wants_to_learn_list,
        teaching_style_list=user_data.teaching_style_list,
        learning_preference_list=user_data.learning_preference_list,
        can_teach=user_data.can_teach,
        wants_to_learn=user_data.wants_to_learn,
        teaching_style=user_data.teaching_style,
        learning_preference=user_data.learning_preference,
        timezone=user_data.timezone,
        languages=user_data.languages,
        availability=normalized_availability,
        score=5.0,
        image_path=image_path,
        teach_embedding=teach_embedding,
        learn_embedding=learn_embedding,
        profile_embedding=profile_embedding,
        num_transactions=0
    )
    db.add(user_db)
    db.commit()
    db.refresh(user_db)
    # Calculate active transactions
    active_tx = db.query(TransactionDB).filter(
        ((TransactionDB.teacher_id == user_db.user_id) | 
         (TransactionDB.learner_id == user_db.user_id)) &
        (TransactionDB.status.in_(["pending", "ongoing"]))
    ).count()
    return UserProfile(**user_db.__dict__, active_transactions=active_tx)

@app.get("/users/{user_id}", response_model=UserProfile)
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    active_tx = db.query(TransactionDB).filter(
        ((TransactionDB.teacher_id == user_id) | 
         (TransactionDB.learner_id == user_id)) &
        (TransactionDB.status.in_(["pending", "ongoing"]))
    ).count()
    return UserProfile(**user.__dict__, active_transactions=active_tx)

@app.get("/transactions/{transaction_id}", response_model=TransactionResponse)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    transaction = db.query(TransactionDB).filter(
        TransactionDB.transaction_id == transaction_id
    ).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction

@app.get("/matches/{user_id}", response_model=List[MatchResult])
def get_matches(user_id: str, limit: int = 5, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    potential_matches = db.query(UserDB).filter(
        UserDB.user_id != user_id,
        UserDB.can_teach_list.overlap(user.wants_to_learn_list or [])
    ).all()
    matches = []
    for other in potential_matches:
        matching_skills = list(set(user.wants_to_learn_list or []) & set(other.can_teach_list or []))
        availability_overlap = calculate_availability_overlap(
            user.availability or [], other.availability or []
        )
        score = calculate_match_score(
            user, other, availability_overlap, len(matching_skills)
        )
        explanation = generate_match_explanation(
            user, other, matching_skills, availability_overlap, score
        )
        matches.append({
            "user_id": other.user_id,
            "username": other.username,
            "score": score,
            "matching_skills": matching_skills,
            "availability_overlap": availability_overlap,
            "explanation": explanation
        })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:limit]

@app.post("/transactions", status_code=status.HTTP_201_CREATED)
def create_transaction(
    transaction_data: TransactionCreate,
    db: Session = Depends(get_db)
):
    teacher = db.query(UserDB).filter(UserDB.user_id == transaction_data.teacher_id).first()
    learner = db.query(UserDB).filter(UserDB.user_id == transaction_data.learner_id).first()
    if not teacher or not learner:
        raise HTTPException(status_code=404, detail="One or both users not found")
    if transaction_data.skill_taught not in (teacher.can_teach_list or []):
        raise HTTPException(status_code=400, detail="Teacher does not offer this skill")
    if transaction_data.skill_learned not in (learner.wants_to_learn_list or []):
        raise HTTPException(status_code=400, detail="Learner does not want this skill")
    existing_tx = db.query(TransactionDB).filter(
        (TransactionDB.teacher_id == transaction_data.teacher_id) &
        (TransactionDB.learner_id == transaction_data.learner_id) &
        (TransactionDB.skill_taught == transaction_data.skill_taught) &
        (TransactionDB.skill_learned == transaction_data.skill_learned) &
        (TransactionDB.status.in_(["pending", "ongoing"]))
    ).first()
    if existing_tx:
        raise HTTPException(status_code=400, detail="Transaction for these skills already exists")
    transaction = TransactionDB(
        transaction_id=str(uuid4()),
        teacher_id=transaction_data.teacher_id,
        learner_id=transaction_data.learner_id,
        skill_taught=transaction_data.skill_taught,
        skill_learned=transaction_data.skill_learned,
        status="pending"
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return {
        "transaction_id": transaction.transaction_id,
        "status": transaction.status,
        "message": "Transaction created successfully"
    }

@app.post("/transactions/{transaction_id}/confirm")
def confirm_transaction(
    transaction_id: str,
    user_id: str,
    rating: float = Field(..., ge=0, le=5),
    feedback: Optional[str] = None,
    db: Session = Depends(get_db)
):
    transaction = db.query(TransactionDB).filter(
        TransactionDB.transaction_id == transaction_id
    ).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if user_id == transaction.teacher_id:
        transaction.teacher_confirmed = True
        transaction.learner_rating = rating
        transaction.learner_feedback = feedback
    elif user_id == transaction.learner_id:
        transaction.learner_confirmed = True
        transaction.teacher_rating = rating
        transaction.teacher_feedback = feedback
    else:
        raise HTTPException(status_code=403, detail="User not part of this transaction")
    if transaction.teacher_confirmed and transaction.learner_confirmed:
        transaction.status = "completed"
        transaction.completed_at = datetime.utcnow()
        teacher = db.query(UserDB).filter(UserDB.user_id == transaction.teacher_id).first()
        learner = db.query(UserDB).filter(UserDB.user_id == transaction.learner_id).first()
        if teacher and learner:
            teacher.score = round(
                ((teacher.score * teacher.num_transactions) + (transaction.learner_rating or 0)) /
                (teacher.num_transactions + 1),
                2
            )
            teacher.num_transactions += 1
            learner.score = round(
                ((learner.score * learner.num_transactions) + (transaction.teacher_rating or 0)) /
                (learner.num_transactions + 1),
                2
            )
            learner.num_transactions += 1
    db.commit()
    return {
        "status": transaction.status,
        "message": "Transaction confirmed successfully"
    }

@app.get("/skills/search")
def search_by_skill(
    skill_id: int,
    user_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    user = db.query(UserDB).filter(UserDB.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    teachers = db.query(UserDB).filter(
        UserDB.can_teach_list.contains([skill_id]),
        UserDB.user_id != user_id
    ).all()
    results = []
    for teacher in teachers:
        availability_overlap = calculate_availability_overlap(
            user.availability or [],
            teacher.availability or []
        )
        score = calculate_match_score(
            user,
            teacher,
            availability_overlap,
            1
        )
        results.append({
            "user_id": teacher.user_id,
            "username": teacher.username,
            "score": score,
            "availability_overlap": availability_overlap,
            "image_path": teacher.image_path,
            "rating": teacher.score
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
