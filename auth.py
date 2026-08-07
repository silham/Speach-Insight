"""
Minimal auth + ownership layer.

Login only — there is no signup endpoint. Users are created by `seed_users.py`.

Two roles:
  * admin — sees every analysis, may index guideline documents
  * user  — sees only the analyses they uploaded
"""

import os
import datetime as dt

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set — add it to your .env file.")

# JWT_SECRET must be stable across restarts, otherwise every reload invalidates
# tokens users are already holding.
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set — add a long random value to your .env file. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
    )

JWT_ALGORITHM = "HS256"
TOKEN_TTL = dt.timedelta(hours=12)

ROLES = ("admin", "user")

# pool_pre_ping keeps pooled Neon connections from going stale between requests.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="user")
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
        }


class Analysis(Base):
    """One processed audio job, owned by the user who uploaded it.

    Only summary fields live here — the heavy artefacts (segments, report,
    per-segment audio) stay on disk under processed/<job_id>/.
    """

    __tablename__ = "analyses"

    job_id = Column(String(32), primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    total_speakers = Column(Integer)
    total_segments = Column(Integer)
    total_duration = Column(Float)
    lead_speaker = Column(String(64))
    total_score = Column(Float)

    def to_dict(self, owner_name=None):
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_speakers": self.total_speakers,
            "total_segments": self.total_segments,
            "total_duration": self.total_duration,
            "lead_speaker": self.lead_speaker,
            "total_score": self.total_score,
            "owner_id": self.owner_id,
            "owner_name": owner_name,
        }


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(engine)


# ── Passwords ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── Tokens ────────────────────────────────────────────────────────────

def create_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "exp": dt.datetime.now(dt.timezone.utc) + TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# ── FastAPI dependencies ──────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db=Depends(get_db),
) -> User:
    if creds is None:
        raise _UNAUTHORIZED
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise _UNAUTHORIZED

    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        # Token is well-formed but the account is gone.
        raise _UNAUTHORIZED
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


def visible_analysis_or_404(db, job_id: str, user: User) -> Analysis:
    """Fetch an analysis the user is allowed to see, else 404.

    Deliberately 404 rather than 403 for rows that exist but belong to someone
    else, so the endpoint doesn't leak which job IDs are real.
    """
    q = db.query(Analysis).filter(Analysis.job_id == job_id)
    if not user.is_admin:
        q = q.filter(Analysis.owner_id == user.id)
    row = q.first()
    if row is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row
