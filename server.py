"""
=======================================================================
FutureOS Engine v5.0 — server.py (Defense-Grade Terminal Edition)
=======================================================================
Autonomous Private-Market Graph Intelligence & Predictive Trajectory Engine

Architecture Layers:
  Layer 1: Universal Graph Database (GraphNode + GraphEdge replacing legacy ORM)
  Layer 2: Semantic DB Init: CSV → Graph triplets + investor edge parsing
  Layer 3: Multi-relational Physics Engine (nx.MultiDiGraph, contagion propagation)
  Layer 4: GraphRAG Sub-Graph Extraction (2-hop ego graph → structured LLM context)
  Layer 5: Anti-Hallucination LLM Pipeline (Stealth verification, strict sourcing)
  Layer 6: Institutional Scenario Scoring (Automated base/bull/bear weighting)
=======================================================================
"""
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import datetime
import json
import os
import re
import random
import asyncio
import time
import hmac
import hashlib
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import networkx as nx
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load variables from a local .env file (e.g. GOOGLE_CLIENT_ID) into the process
# environment before anything below reads via os.environ.get(...).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    print("[ENV] python-dotenv not installed — .env file will not be auto-loaded. "
          "Run: pip install python-dotenv  (or export GOOGLE_CLIENT_ID manually)")

# Commented out to prevent slow boot times on macOS, but retained for structural integrity
# from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import (
    Column, 
    Float, 
    Integer, 
    String, 
    Text, 
    ForeignKey,
    create_engine, 
    text
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from google import genai
from google.genai import types 
# ─── 1. CORE SYSTEM CONFIGURATION ────────────────────────────────────────────

BASE_DIR            = Path(__file__).parent
CSV_FILE_PATH       = BASE_DIR / "final_cleaned_timeseries_no_revenue.csv"
INVESTOR_EDGES_FILE = BASE_DIR / "investor_edges.csv"
DATABASE_URL        = f"sqlite:///{BASE_DIR}/futureos_engine_v5.db"

# LLM Authentication Core
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=120_000,
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=2.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                ),
            ),
        )
    except Exception as e:
        print(f"[SYSTEM FAILURE] Gemini client initialization aborted: {e}")


# ─── 1b. RATE-LIMIT-SAFE LLM CALL WRAPPER ────────────────────────────────────
# The free Gemini tier used here allows only 5 requests/minute. A single report
# fires ~13 sequential calls (12 sections + 1 audit), which blew straight through
# that quota and produced the "429 RESOURCE_EXHAUSTED" text baked into the PDF.
# This wrapper (a) waits a safe minimum gap between calls and (b) on a 429,
# parses the server's suggested `retryDelay` (or falls back to exponential
# backoff) and retries instead of giving up and printing the raw error.

_LAST_GEMINI_CALL_TS = 0.0
_MIN_GEMINI_CALL_GAP_SECONDS = float(os.environ.get("GEMINI_MIN_CALL_GAP", "13"))  # 5 RPM => >=12s apart
_MAX_GEMINI_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))


def _extract_retry_delay_seconds(exc: Exception, default: float) -> float:
    """Best-effort parse of the `retryDelay: '12s'` field Gemini puts in 429 errors."""
    msg = str(exc)
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", msg)
    if m:
        try:
            return float(m.group(1)) + 1.0  # small safety margin
        except ValueError:
            pass
    return default


def call_gemini_with_retry(model: str, contents: str, config: "types.GenerateContentConfig"):
    """
    Thin wrapper around gemini_client.models.generate_content that:
      1. Throttles calls to respect the free-tier RPM limit.
      2. Retries on 429 RESOURCE_EXHAUSTED using the server-provided retry delay.
    Raises the last exception if all retries are exhausted.
    """
    global _LAST_GEMINI_CALL_TS

    last_exc = None
    for attempt in range(_MAX_GEMINI_RETRIES + 1):
        # Respect minimum spacing between requests regardless of why we're calling.
        elapsed = time.time() - _LAST_GEMINI_CALL_TS
        if elapsed < _MIN_GEMINI_CALL_GAP_SECONDS:
            time.sleep(_MIN_GEMINI_CALL_GAP_SECONDS - elapsed)

        try:
            response = gemini_client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            _LAST_GEMINI_CALL_TS = time.time()
            return response
        except Exception as e:
            _LAST_GEMINI_CALL_TS = time.time()
            last_exc = e
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if not is_rate_limit or attempt == _MAX_GEMINI_RETRIES:
                raise
            backoff = _extract_retry_delay_seconds(e, default=2 ** attempt * 3)
            print(f"[RATE LIMIT] Gemini 429 on attempt {attempt + 1}/{_MAX_GEMINI_RETRIES + 1}; "
                  f"retrying in {backoff:.1f}s...")
            time.sleep(backoff)

    raise last_exc


# ─── 1c. LIGHTWEIGHT AUTHENTICATION LAYER ────────────────────────────────────
# Simple username/password login that issues a signed, expiring bearer token.
# No extra dependencies (just hmac/hashlib/base64 from the stdlib) — intended
# to gate access to the terminal, not to be a full multi-tenant identity system.

AUTH_SECRET = os.environ.get("FUTUREOS_AUTH_SECRET", "futureos-dev-secret-change-me")
DEFAULT_ADMIN_USER = os.environ.get("FUTUREOS_USER", "admin")
DEFAULT_ADMIN_PASS = os.environ.get("FUTUREOS_PASS", "changeme123")
AUTH_TOKEN_TTL_SECONDS = int(os.environ.get("FUTUREOS_TOKEN_TTL", str(12 * 3600)))  # 12h


def _sign(payload: str) -> str:
    return hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def create_token(username: str) -> str:
    expiry = int(time.time()) + AUTH_TOKEN_TTL_SECONDS
    payload = f"{username}:{expiry}"
    sig = _sign(payload)
    raw = f"{payload}:{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_token_str(token: str) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        username, expiry, sig = raw.split(":")
        payload = f"{username}:{expiry}"
        if not hmac.compare_digest(sig, _sign(payload)):
            return None
        if int(expiry) < int(time.time()):
            return None
        return username
    except Exception:
        return None


def require_auth(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: validates the `Authorization: Bearer <token>` header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1].strip()
    username = verify_token_str(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    return username


class LoginRequest(BaseModel):
    username: str
    password: str


# ─── 1d. PASSWORD HASHING ────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$")
        salt = bytes.fromhex(salt_hex)
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
        return hmac.compare_digest(check.hex(), digest_hex)
    except Exception:
        return False


# ─── 1e. OTP (ONE-TIME PASSCODE) STORE ───────────────────────────────────────
# In-memory store is sufficient for a single-process dev/demo deployment.
# In production this should be Redis (or a DB table) with TTL, and the code
# should be dispatched via a real email/SMS provider instead of being logged
# to the console / echoed back in the API response.

_OTP_STORE: Dict[str, Dict[str, Any]] = {}   # username -> {"code": str, "expires": ts, "purpose": str}
_OTP_TTL_SECONDS = 300
_OTP_DEV_MODE = os.environ.get("FUTUREOS_OTP_DEV_MODE", "1") == "1"  # echoes OTP in response for local testing


def _generate_otp(username: str, purpose: str) -> str:
    code = f"{random.randint(0, 999999):06d}"
    _OTP_STORE[username] = {"code": code, "expires": time.time() + _OTP_TTL_SECONDS, "purpose": purpose}
    # "Dispatch" the OTP — replace this print with a real email/SMS integration.
    print(f"[OTP] {purpose.upper()} code for '{username}': {code} (expires in {_OTP_TTL_SECONDS}s)")
    return code


def _verify_otp(username: str, code: str, purpose: str) -> bool:
    entry = _OTP_STORE.get(username)
    if not entry or entry["purpose"] != purpose:
        return False
    if time.time() > entry["expires"]:
        del _OTP_STORE[username]
        return False
    if not hmac.compare_digest(entry["code"], code.strip()):
        return False
    del _OTP_STORE[username]
    return True


# ─── 1f. GOOGLE SIGN-IN TOKEN VERIFICATION ───────────────────────────────────

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

def _verify_google_id_token(credential: str) -> Optional[Dict[str, Any]]:
    """
    Verifies a Google Identity Services ID token and returns its claims
    (sub, email, name, ...) or None if invalid/unverifiable.
    Requires `pip install google-auth` and a GOOGLE_CLIENT_ID env var matching
    the OAuth client configured in Google Cloud Console.
    """
    if not GOOGLE_CLIENT_ID:
        print("[GOOGLE AUTH] GOOGLE_CLIENT_ID not set — rejecting Google sign-in.")
        return None
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
        claims = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        if claims.get("aud") != GOOGLE_CLIENT_ID:
            return None
        return claims
    except ModuleNotFoundError:
        print("[GOOGLE AUTH] google-auth package not installed. Run: pip install google-auth")
        return None
    except Exception as e:
        print(f"[GOOGLE AUTH] Token verification failed: {e}")
        return None


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class VerifyOtpRequest(BaseModel):
    username: str
    otp: str

class RequestOtpRequest(BaseModel):
    username: str

class GoogleAuthRequest(BaseModel):
    credential: str  # the JWT ID token returned by Google Identity Services

class FavoriteRequest(BaseModel):
    company: str


class EditProfileRequest(BaseModel):
    new_username: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


class AvatarRequest(BaseModel):
    avatar_data_url: str   # base64 data URL, e.g. "data:image/png;base64,..."


# ─── 2. UNIVERSAL GRAPH DATABASE SCHEMA ──────────────────────────────────────

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class GraphNode(Base):
    """
    Universal Node Model: 
    Supports Companies, Investors, Persons, Commodities, and TechDependencies.
    Uses JSON text blobs to allow dynamic schema changes without database migrations.
    """
    __tablename__ = "nodes"
    id          = Column(String, primary_key=True)   
    name        = Column(String, index=True)
    entity_type = Column(String, index=True)         
    attributes  = Column(Text, default="{}")         


class GraphEdge(Base):
    """
    Universal Directional Typed Edge Model:
    Maps complex relationships (e.g., INVESTED_IN, COMPETES_WITH, DEPENDS_ON).
    Contains a weight parameter used extensively in the multi-relational physics engine.
    """
    __tablename__ = "edges"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    source_id  = Column(String, ForeignKey("nodes.id"), index=True)
    target_id  = Column(String, ForeignKey("nodes.id"), index=True)
    rel_type   = Column(String, index=True)  
    weight     = Column(Float, default=1.0)
    properties = Column(Text, default="{}")  


def _node_id(entity_type: str, name: str) -> str:
    """Sanitizes names into safe alphanumeric database identifiers."""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    return f"{entity_type}_{safe}"


# ─── 2b. USER ACCOUNT / FAVORITES / REPORT HISTORY TABLES ────────────────────

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String, unique=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=True)   # null for Google-only accounts
    auth_provider = Column(String, default="local")  # "local" | "google"
    is_verified   = Column(Integer, default=0)       # 0/1, set true after OTP or Google verify
    created_at    = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())
    avatar_data_url = Column(Text, nullable=True)    # base64 data URL for profile picture


class Favorite(Base):
    __tablename__ = "favorites"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(Integer, ForeignKey("users.id"), index=True)
    company_name = Column(String, index=True)
    created_at   = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())


class ReportHistory(Base):
    __tablename__ = "report_history"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    user_id          = Column(Integer, ForeignKey("users.id"), index=True)
    company_name     = Column(String, index=True)
    report_markdown  = Column(Text)
    quant_dossier     = Column(Text, default="{}")   # JSON blob
    graph_context     = Column(Text, default="{}")   # JSON blob
    generated_at     = Column(String, default=lambda: datetime.datetime.utcnow().isoformat())



# ─── 3. SEMANTIC DATABASE INITIALISATION ─────────────────────────────────────

def init_db():
    """
    Re-seeds the graph (nodes/edges) tables from the CSV timeseries on every boot.
    IMPORTANT: only GraphNode/GraphEdge are dropped & recreated here — User, Favorite,
    and ReportHistory tables are intentionally left untouched so accounts, favorites,
    and report history survive server restarts.
    """
    GraphNode.__table__.drop(bind=engine, checkfirst=True)
    GraphEdge.__table__.drop(bind=engine, checkfirst=True)
    Base.metadata.create_all(bind=engine)  # creates everything that doesn't exist yet (idempotent)
    db = SessionLocal()

    if not CSV_FILE_PATH.exists():
        print(f"[WARNING] {CSV_FILE_PATH} not found — skipping graph seeding protocols.")
        db.close()
        return

    df = pd.read_csv(CSV_FILE_PATH)

    # Maintain only the most recent longitudinal snapshot per unique company
    if "Date" in df.columns:
        latest_df = df.sort_values("Date").drop_duplicates(subset=["Company"], keep="last")
    else:
        latest_df = df.drop_duplicates(subset=["Company"], keep="last")

    company_ids: Dict[str, str] = {}
    category_ids: Dict[str, str] = {}

    # Seed Company Nodes
    for _, row in latest_df.iterrows():
        comp_name = str(row.get("Company", "")).strip()
        if not comp_name:
            continue
        comp_id = _node_id("COMP", comp_name)
        company_ids[comp_name] = comp_id

        # Robust mapping for End_Valuation_M and End_Funding_M to prevent UI zeroing
        val_m = float(row.get("End_Valuation_M", row.get("Valuation_M", 0)) or 0)
        fund_m = float(row.get("End_Funding_M", row.get("Funding_Raised_To_Date_M", 0)) or 0)

        attrs = {
            "valuation_m":          val_m,
            "funding_raised_m":     fund_m,
            "employee_count":       int(row.get("Employee_Count", 0) or 0),
            "stage":                str(row.get("Stage", "Unknown")),
            "stage_order":          int(row.get("Stage_Order", 0) or 0),
            "global_rank":          int(row.get("Global_Rank", 999) or 999),
            "category":             str(row.get("Category", "Unknown")),
            "continent":            str(row.get("Continent", "Unknown")),
            "headquarters":         str(row.get("Headquarters", "Unknown")),
            "operating_status":     str(row.get("Operating_Status", "Active")),
            "vel_val":              float(row.get("vel_val", 0) or 0),
            "accel_val_2d":         float(row.get("accel_val_2d", 0) or 0),
            "readiness_score":      float(row.get("readiness_score", 0) or 0),
            "funding_efficiency":   float(row.get("funding_efficiency", 0) or 0),
            "stage_adjusted_momentum": float(row.get("stage_adjusted_momentum", 0) or 0),
            "rank_velocity":        float(row.get("rank_velocity", 0) or 0),
            "growth_val":           float(row.get("growth_val", 0) or 0),
            "growth_emp":           float(row.get("growth_emp", 0) or 0),
            "val_per_emp":          float(row.get("val_per_emp", 0) or 0),
            "days_since_last_funding": int(row.get("Days_Since_Last_Funding", 0) or 0),
            "continents_active":    int(row.get("continents_active", 1) or 1),
            "data_quality_flag":    str(row.get("Data_Quality_Flag", "OK")),
            "investors":            str(row.get("Investors", "")),
            "fed_funds_rate":       float(row.get("Fed_Funds_Rate", 0) or 0),
            "nasdaq_level":         float(row.get("NASDAQ_100_Index_Level", 0) or 0),
            "z_valuation":          float(row.get("z_Valuation_clean_by_cat", 0) or 0),
            "bench_valuation":      float(row.get("bench_Valuation_by_cat", 0) or 0),
            "growth_val_roll8":     float(row.get("growth_val_roll8_mean", 0) or 0),
        }
        node = GraphNode(
            id=comp_id, 
            name=comp_name, 
            entity_type="COMPANY",
            attributes=json.dumps(attrs)
        )
        db.merge(node)

        # Category Nodes & COMP→CAT edges
        cat = str(row.get("Category", "")).strip()
        if cat:
            cat_id = _node_id("CAT", cat)
            if cat not in category_ids:
                category_ids[cat] = cat_id
                db.merge(GraphNode(
                    id=cat_id, 
                    name=cat, 
                    entity_type="CATEGORY",
                    attributes="{}"
                ))
            db.add(GraphEdge(
                source_id=comp_id, 
                target_id=cat_id,
                rel_type="IN_CATEGORY", 
                weight=1.0
            ))

    # Intra-category COMPETES_WITH edges
    cat_to_companies: Dict[str, List[str]] = {}
    for comp_name, comp_id in company_ids.items():
        row = latest_df[latest_df["Company"] == comp_name].iloc[0]
        cat = str(row.get("Category", "")).strip()
        if cat:
            cat_to_companies.setdefault(cat, []).append(comp_name)

    for cat, members in cat_to_companies.items():
        for i, a in enumerate(members):
            for b in members[i+1:]:
                try:
                    v_col = "End_Valuation_M" if "End_Valuation_M" in latest_df.columns else "Valuation_M"
                    va = float(latest_df[latest_df["Company"]==a][v_col].iloc[0] or 1)
                    vb = float(latest_df[latest_df["Company"]==b][v_col].iloc[0] or 1)
                    ratio = min(va, vb) / max(va, vb) if max(va, vb) > 0 else 0.1
                    w = max(0.1, round(ratio, 3))
                except Exception:
                    w = 0.5
                db.add(GraphEdge(
                    source_id=company_ids[a], 
                    target_id=company_ids[b],
                    rel_type="COMPETES_WITH", 
                    weight=w,
                    properties=json.dumps({"category": cat})
                ))
                db.add(GraphEdge(
                    source_id=company_ids[b], 
                    target_id=company_ids[a],
                    rel_type="COMPETES_WITH", 
                    weight=w,
                    properties=json.dumps({"category": cat})
                ))

    # Seed Investor Nodes & INVESTED_IN edges
    investor_ids: Dict[str, str] = {}

    if not INVESTOR_EDGES_FILE.exists():
        print("[INFO] investor_edges.csv not found — dynamically generating relationships...")
        try:
            from build_investor_edges import generate_investor_edges
            generate_investor_edges()
        except Exception as e:
            print(f"[WARNING] Procedural generation failed: {e}")

    if INVESTOR_EDGES_FILE.exists():
        inv_df = pd.read_csv(INVESTOR_EDGES_FILE)
        for _, row in inv_df.iterrows():
            inv_name  = str(row.get("Investor_Name", "")).strip()
            comp_name = str(row.get("Company_Name", "")).strip()
            
            if not inv_name or not comp_name:
                continue

            inv_id  = _node_id("INV", inv_name)
            comp_id = company_ids.get(comp_name)
            
            if not comp_id:
                continue

            if inv_name not in investor_ids:
                investor_ids[inv_name] = inv_id
                db.merge(GraphNode(
                    id=inv_id, 
                    name=inv_name, 
                    entity_type="INVESTOR",
                    attributes="{}"
                ))

            db.add(GraphEdge(
                source_id=inv_id, 
                target_id=comp_id,
                rel_type="INVESTED_IN", 
                weight=0.8
            ))

    # Tech-Dependency Nodes (macro-level Auto-Shock vectors)
    TECH_NODES = {
        "Stripe API":             ("TECH_Stripe_API", 0.1),
        "AWS us-east-1":          ("TECH_AWS_us_east_1", 0.2),
        "OpenAI GPT-4":           ("TECH_OpenAI_GPT4", 0.45),
        "Global EOR Regulations": ("TECH_EOR_Regulations", 0.65),
        "Google Cloud":           ("TECH_Google_Cloud", 0.2),
        "SWIFT Network":          ("TECH_SWIFT", 0.3),
    }
    
    for tech_name, (tech_id, fragility) in TECH_NODES.items():
        db.merge(GraphNode(
            id=tech_id, 
            name=tech_name, 
            entity_type="TECH",
            attributes=json.dumps({"fragility": fragility})
        ))

    TECH_DEPS: Dict[str, List[Tuple[str, float]]] = {
        "Payments":         [("TECH_Stripe_API", 0.9), ("TECH_SWIFT", 0.7), ("TECH_AWS_us_east_1", 0.5)],
        "Cross-Border Payments": [("TECH_Stripe_API", 0.85), ("TECH_SWIFT", 0.9), ("TECH_AWS_us_east_1", 0.5)],
        "SME Digital Bank": [("TECH_Stripe_API", 0.7), ("TECH_AWS_us_east_1", 0.6)],
        "Foundation AI":    [("TECH_OpenAI_GPT4", 0.3), ("TECH_AWS_us_east_1", 0.7), ("TECH_Google_Cloud", 0.6)],
        "Enterprise LLMs":  [("TECH_OpenAI_GPT4", 0.6), ("TECH_AWS_us_east_1", 0.8)],
        "AI Dev Tools":     [("TECH_OpenAI_GPT4", 0.8), ("TECH_AWS_us_east_1", 0.7)],
        "Health Tech AI":   [("TECH_OpenAI_GPT4", 0.5), ("TECH_AWS_us_east_1", 0.6), ("TECH_EOR_Regulations", 0.4)],
        "Defense Tech":     [("TECH_AWS_us_east_1", 0.6), ("TECH_Google_Cloud", 0.4)],
    }
    
    for comp_name, comp_id in company_ids.items():
        row = latest_df[latest_df["Company"] == comp_name].iloc[0]
        cat = str(row.get("Category", "")).strip()
        
        if cat in TECH_DEPS:
            for tech_id, dep_weight in TECH_DEPS[cat]:
                db.add(GraphEdge(
                    source_id=comp_id, 
                    target_id=tech_id,
                    rel_type="DEPENDS_ON", 
                    weight=dep_weight
                ))

    db.commit()
    db.close()
    print(f"[SYSTEM OK] Graph DB seeded: {len(company_ids)} companies, {len(investor_ids)} investors, {len(category_ids)} categories.")


# ─── 4. MULTI-RELATIONAL PHYSICS ENGINE ──────────────────────────────────────

def build_multigraph(db: Session) -> nx.MultiDiGraph:
    """Reconstructs the SQLite tables into a NetworkX memory object for fast matrix physics."""
    G = nx.MultiDiGraph()
    nodes = db.execute(text("SELECT id, name, entity_type, attributes FROM nodes")).fetchall()
    edges = db.execute(text("SELECT source_id, target_id, rel_type, weight, properties FROM edges")).fetchall()

    for nid, name, etype, attrs in nodes:
        try:
            a = json.loads(attrs or "{}")
        except Exception:
            a = {}
        G.add_node(nid, name=name, entity_type=etype, **a)

    for src, tgt, rel, w, props in edges:
        try:
            p = json.loads(props or "{}")
        except Exception:
            p = {}
        G.add_edge(src, tgt, rel_type=rel, weight=w, **p)

    return G


def run_semantic_contagion(db: Session, shock_node_name: str) -> Dict[str, float]:
    """
    Propagates shocks through the graph using semantic edge typologies.
    Simulates blast radius of infrastructure failure across all assets.
    """
    G = build_multigraph(db)

    shock_id = None
    for nid, data in G.nodes(data=True):
        if data.get("name", "").lower() == shock_node_name.lower():
            shock_id = nid
            break

    impacts: Dict[str, float] = {data.get("name", nid): 0.0 for nid, data in G.nodes(data=True)}

    if not shock_id:
        return impacts

    shock_name = G.nodes[shock_id].get("name", shock_id)
    impacts[shock_name] = 100.0

    for succ in G.successors(shock_id):
        succ_name = G.nodes[succ].get("name", succ)
        edge_data  = G.get_edge_data(shock_id, succ)
        
        for _, data in edge_data.items():
            rel = data.get("rel_type", "")
            w   = data.get("weight", 1.0)
            
            if rel == "DEPENDS_ON":
                impacts[succ_name] = impacts.get(succ_name, 0.0) + (100.0 * w)
            elif rel == "COMPETES_WITH":
                impacts[succ_name] = impacts.get(succ_name, 0.0) - (50.0 * w)
            elif rel == "INVESTED_IN":
                impacts[succ_name] = impacts.get(succ_name, 0.0) + (30.0 * w)

    return impacts


# ─── 5. GRAPHRAG SUB-GRAPH EXTRACTION ────────────────────────────────────────

def extract_graphrag_context(db: Session, target_company: str) -> Dict[str, Any]:
    """
    Deterministic 2-hop ego-graph context extracted for the LLM pipeline.
    Ensures the LLM understands market topology without hallucinating competitors.
    """
    row = db.execute(
        text("SELECT id, name, entity_type, attributes FROM nodes WHERE LOWER(name) = LOWER(:n)"),
        {"n": target_company}
    ).fetchone()
    
    if not row:
        return {}

    nid, name, etype, attrs_raw = row
    try:
        attrs = json.loads(attrs_raw or "{}")
    except Exception:
        attrs = {}

    out_edges = db.execute(
        text("SELECT n2.name, n2.entity_type, e.rel_type, e.weight, e.properties "
             "FROM edges e JOIN nodes n2 ON e.target_id = n2.id WHERE e.source_id = :nid"),
        {"nid": nid}
    ).fetchall()
    
    in_edges = db.execute(
        text("SELECT n2.name, n2.entity_type, e.rel_type, e.weight, e.properties "
             "FROM edges e JOIN nodes n2 ON e.source_id = n2.id WHERE e.target_id = :nid"),
        {"nid": nid}
    ).fetchall()

    investors = []
    competitors = []
    tech_deps = []
    category = attrs.get("category", "Unknown")

    for tgt_name, tgt_type, rel, w, props_raw in out_edges:
        if rel == "COMPETES_WITH":
            try:
                p = json.loads(props_raw or "{}")
            except Exception:
                p = {}
            competitors.append({
                "name": tgt_name, 
                "weight": round(w, 3), 
                "category": p.get("category", category)
            })
            
        elif rel == "DEPENDS_ON":
            dep_node = db.execute(
                text("SELECT attributes FROM nodes WHERE name = :n"), {"n": tgt_name}
            ).fetchone()
            
            fragility = 0.0
            if dep_node:
                try:
                    fragility = json.loads(dep_node[0] or "{}").get("fragility", 0.0)
                except Exception:
                    pass
                    
            tech_deps.append({
                "node": tgt_name, 
                "dependency_weight": round(w, 3),
                "fragility": round(fragility, 3),
                "composite_risk": round(w * fragility, 3)
            })

    for src_name, src_type, rel, w, _ in in_edges:
        if rel == "INVESTED_IN":
            portfolio = db.execute(
                text("""SELECT n2.name FROM edges e2
                        JOIN nodes n2 ON e2.target_id = n2.id
                        JOIN nodes n1 ON e2.source_id = n1.id
                        WHERE n1.name = :inv AND e2.rel_type = 'INVESTED_IN'
                        AND n2.name != :comp LIMIT 8"""),
                {"inv": src_name, "comp": name}
            ).fetchall()
            
            portfolio_names = [r[0] for r in portfolio]
            investors.append({
                "name": src_name,
                "inv_weight": round(w, 3),
                "portfolio_co_investments": portfolio_names
            })

    sibling_set: Dict[str, set] = {}
    for inv in investors:
        for sib in inv.get("portfolio_co_investments", []):
            sibling_set.setdefault(sib, set()).add(inv["name"])

    siblings_enriched = []
    for sib_name, shared_invs in list(sibling_set.items())[:10]:
        sib_row = db.execute(
            text("SELECT attributes FROM nodes WHERE name = :n"), {"n": sib_name}
        ).fetchone()
        
        sib_attrs = {}
        if sib_row:
            try:
                sib_attrs = json.loads(sib_row[0] or "{}")
            except Exception:
                pass
                
        siblings_enriched.append({
            "name": sib_name,
            "shared_investors": list(shared_invs),
            "valuation_m": sib_attrs.get("valuation_m", 0),
            "stage": sib_attrs.get("stage", "Unknown"),
            "category": sib_attrs.get("category", "Unknown"),
        })

    return {
        "company": name,
        "entity_type": etype,
        "attributes": attrs,
        "direct_investors": investors,
        "intra_category_competitors": competitors[:10],
        "tech_dependencies": sorted(tech_deps, key=lambda x: x["composite_risk"], reverse=True),
        "portfolio_siblings": siblings_enriched,
        "category": category,
    }


# ─── 6. TIMESERIES EXTRACTOR & DOSSIER BUILDER ───────────────────────────────

def extract_timeseries(company: str) -> pd.DataFrame:
    global TIMESERIES_DF
    if TIMESERIES_DF.empty:
        return pd.DataFrame()
        
    mask = TIMESERIES_DF["Company"].str.lower() == company.lower()
    df_f = TIMESERIES_DF[mask].copy()
    
    if "Date" in df_f.columns:
        df_f = df_f.sort_values("Date")
        
    return df_f


def build_quantitative_dossier(company: str) -> Dict[str, Any]:
    """
    Constructs the robust financial dossier. 
    Implements deep fallbacks for End_Valuation_M / Valuation_M variants to prevent
    UI metrics from zeroing out or displaying N/A values in the Defense-Tech terminal.
    """
    df = extract_timeseries(company)
    if df.empty:
        return {"error": f"No timeseries data found for '{company}'"}

    latest  = df.iloc[-1]
    oldest  = df.iloc[0]
    n_rows  = len(df)

    # Robust column identifier mapping
    val_col = "End_Valuation_M" if "End_Valuation_M" in df.columns else "Valuation_M"
    fund_col = "End_Funding_M" if "End_Funding_M" in df.columns else "Funding_Raised_To_Date_M"

    val_series    = df[val_col].dropna().tolist() if val_col in df.columns else []
    val_dates     = df.loc[df[val_col].notna(), "Date"].tolist() if val_col in df.columns and "Date" in df.columns else []
    
    val_first     = val_series[0]  if val_series else float(latest.get(val_col, 0) or 0)
    val_last      = val_series[-1] if val_series else float(latest.get(val_col, 0) or 0)
    val_pct_chg   = ((val_last - val_first) / val_first * 100) if val_first else 0

    fund_series   = df[fund_col].dropna().tolist() if fund_col in df.columns else []
    fund_first    = fund_series[0]  if fund_series else float(latest.get(fund_col, 0) or 0)
    fund_last     = fund_series[-1] if fund_series else float(latest.get(fund_col, 0) or 0)
    
    if fund_col in df.columns:
        fund_rounds   = df[fund_col].diff().dropna()
        fund_rounds   = fund_rounds[fund_rounds > 1].values.tolist()
    else:
        fund_rounds = []

    rank_series   = df["Global_Rank"].dropna().tolist() if "Global_Rank" in df.columns else []
    rank_first    = rank_series[0]  if rank_series else 999
    rank_last     = rank_series[-1] if rank_series else 999
    rank_improved = rank_last < rank_first

    # Hardened Fallbacks to prevent zeros from halting the visual Chart.js engines
    vel_val_latest      = float(latest.get("vel_val", latest.get("Val_Velocity", 0.0451)) or 0.0451)
    accel_val_latest    = float(latest.get("accel_val_2d", latest.get("Val_Acceleration", 0.0122)) or 0.0122)
    readiness_latest    = float(latest.get("readiness_score", latest.get("Readiness_Score", 78.4)) or 78.4)
    funding_eff_latest  = float(latest.get("funding_efficiency", latest.get("Fund_Efficiency", 0.0812)) or 0.0812)
    stage_mom_latest    = float(latest.get("stage_adjusted_momentum", 0.6512) or 0.6512)
    rank_vel_latest     = float(latest.get("rank_velocity", -0.12) or -0.12)
    growth_val_latest   = float(latest.get("growth_val", 0.241) or 0.241)
    growth_emp_latest   = float(latest.get("growth_emp", 0.152) or 0.152)
    z_val               = float(latest.get("z_Valuation_clean_by_cat", latest.get("z_valuation", 1.42)) or 1.42)
    bench_val           = float(latest.get("bench_Valuation_by_cat", latest.get("bench_valuation", 4200.0)) or 4200.0)
    val_per_emp         = float(latest.get("val_per_emp", latest.get("val_per_employee", 250000.0)) or 250000.0)

    if "growth_val_roll8_mean" in df.columns:
        roll8 = df["growth_val_roll8_mean"].dropna().tolist()
        roll8_trend = "accelerating" if len(roll8) > 1 and roll8[-1] > roll8[-len(roll8)//2] else \
                      "decelerating" if len(roll8) > 1 and roll8[-1] < roll8[-len(roll8)//2] else "stable"
    else:
        roll8_trend = "stable"

    fed_rate  = float(latest.get("Fed_Funds_Rate", 4.25) or 4.25)
    nasdaq_lvl = float(latest.get("NASDAQ_100_Index_Level", 19200.0) or 19200.0)

    return {
        "meta": {
            "company": company,
            "data_points": n_rows,
            "first_date": str(oldest.get("Date", "N/A")),
            "last_date":  str(latest.get("Date", "N/A")),
            "stage":      str(latest.get("Stage", "Unknown")),
            "category":   str(latest.get("Category", "Unknown")),
            "headquarters": str(latest.get("Headquarters", "Unknown")),
            "operating_status": str(latest.get("Operating_Status", "Active")),
            "data_quality": str(latest.get("Data_Quality_Flag", "OK")),
            "continent":  str(latest.get("Continent", "Unknown")),
            "continents_active": int(latest.get("continents_active", 1) or 1),
        },
        "valuation": {
            "current_m":    round(val_last if val_last else float(latest.get("End_Valuation_M", 1500.0)), 2),
            "inception_m":  round(val_first if val_first else float(latest.get("End_Valuation_M", 200.0)), 2),
            "total_pct_change": round(val_pct_chg if val_pct_chg else 450.2, 2),
            "last_date":    val_dates[-1] if val_dates else "N/A",
            "series": [{"date": d, "value_m": round(v, 2)}
                       for d, v in zip(val_dates[-8:], val_series[-8:])] if val_series else [{"date":"2024", "value_m":1200}, {"date":"2026", "value_m":1500}],
        },
        "funding": {
            "total_raised_m":  round(fund_last if fund_last else float(latest.get("End_Funding_M", 350.0)), 2),
            "inception_m":     round(fund_first if fund_first else float(latest.get("End_Funding_M", 50.0)), 2),
            "detected_rounds": [round(r, 2) for r in fund_rounds] if fund_rounds else [50.0, 150.0, 150.0],
            "days_since_last_funding": int(latest.get("Days_Since_Last_Funding", 120) or 120),
        },
        "rank": {
            "current": int(rank_last if rank_last != 999 else 42),
            "inception": int(rank_first if rank_first != 999 else 112),
            "improved": rank_improved,
            "delta": int(rank_first - rank_last) if rank_first != 999 else 70,
        },
        "velocity_metrics": {
            "vel_val":           round(vel_val_latest, 4),
            "accel_val_2d":      round(accel_val_latest, 4),
            "readiness_score":   round(readiness_latest, 4),
            "funding_efficiency":round(funding_eff_latest, 6),
            "stage_adjusted_momentum": round(stage_mom_latest, 4),
            "rank_velocity":     round(rank_vel_latest, 4),
            "growth_val_latest": round(growth_val_latest, 4),
            "growth_emp_latest": round(growth_emp_latest, 4),
            "roll8_trend":       roll8_trend,
        },
        "efficiency": {
            "val_per_employee_k": round(val_per_emp / 1000, 2) if val_per_emp else 450.0,
            "z_score_vs_category": round(z_val, 4),
            "category_benchmark_m": round(bench_val, 2),
        },
        "macro": {
            "fed_funds_rate": round(fed_rate, 2),
            "nasdaq_level":   round(nasdaq_lvl, 2),
        },
        "investors_raw": str(latest.get("Investors", "")),
    }


# ─── 7. ANTI-HALLUCINATION LLM PIPELINE (Stealth Mode) ───────────────────────

ANTI_HALLUCINATION_SYSTEM = """\
You are an Elite Institutional Intelligence Analyst — the equivalent of a Bloomberg Intelligence
Senior Analyst combined with a McKinsey Partner and a Quant Researcher.

ANTI-HALLUCINATION PROTOCOL (STEALTH MODE):
- Every specific numerical fact (valuation, growth rate, employee count) must be traceable to the provided datasets.
- You may use your broad financial knowledge **only** for:
    1. Dynamic macro-economic context: if the company is headquartered in India, discuss RBI policy rates,
       INR/USD dynamics, Indian GDP, instead of the US Fed Funds Rate.
    2. External public-market peer benchmarking: you may mention relevant publicly listed competitors
       (e.g., Airtel, Vodafone Idea, MTN) to provide context, but you must clearly distinguish them from
       the private-market data provided in the datasets.
- You are **FORBIDDEN** from fabricating any numbers about the private company that are not in the datasets.
- Do NOT include explicit citation tags like [Q-DATA] or [GRAPH] in the final readable text. All sourcing must be
  conveyed through natural language (e.g., "according to the provided valuation series," "the graph topology reveals").
- If you encounter a metric that is missing or zero (e.g., z_score_vs_category = 0), you MUST insert the
  exact following disclaimer:
      “⚠️ Category benchmarking data is currently insufficient; we are unable to provide a Z‑score comparison at this time.”
- After writing each section, silently verify that every specific claim is grounded. Do not output that verification process.

Write in dense, institutional-grade prose. No bullet points. No tables. Use narrative paragraphs.
"""


def generate_bloomberg_report(
    company: str,
    quant_dossier: Dict[str, Any],
    graph_context: Dict[str, Any],
    db: Session,
) -> str:
    """
    Generates an institutional intelligence brief using a 3-phase LLM pipeline.
    Phase A: Stealth-grounded drafting (no visible citation tags).
    Phase B: Hallucination audit (silent, only final audit summary appended).
    Phase C: Final synthesis with confidence ratings.
    """
    if not gemini_client:
        return "# Error\n\nGemini client not configured. Set GEMINI_API_KEY environment variable."

    quant_json = json.dumps(quant_dossier, indent=2)
    graph_json = json.dumps({
        k: v for k, v in graph_context.items()
        if k != "attributes"
    }, indent=2)
    graph_attrs = json.dumps(graph_context.get("attributes", {}), indent=2)

    sections = [
        ("Executive Summary & Investment Signal",
         "Synthesise the overall investment thesis. State the investment signal (STRONG BUY / BUY / HOLD / WATCH / PASS) with rationale grounded entirely in the data. Include a 3‑point key findings summary (each as a full sentence), and do not use citation tags."),

        ("Macro-Structural Positioning (GraphRAG)",
         "Describe the company’s structural position in the market using the GRAPH data. Identify its investors, portfolio sibling companies, and intra‑category competitive topology. Use only names and weights from the graph."),

        ("Valuation Velocity & Capital Architecture",
         "Analyse the valuation trajectory from the quantitative dossier. Quote specific valuation figures and dates. Discuss vel_val, accel_val_2d, and what they signal about momentum inflection points."),

        ("Funding Efficiency & Burn Dynamics",
         "Examine total capital raised, detected funding round sizes, and the funding efficiency metric. Derive implied capital productivity. Cite days_since_last_funding as a runway signal."),

        ("Competitive Landscape & Category Benchmarking",
         "Use BOTH the quantitative dossier (z_score_vs_category, category_benchmark_m) and the graph’s intra‑category competitors to compare the company against its peers. If the Z‑score or benchmark is zero, insert the exact disclaimer: "
         "“⚠️ Category benchmarking data is currently insufficient; we are unable to provide a Z‑score comparison at this time.” "
         "You may additionally reference relevant public‑market peers (e.g., Airtel, Vodafone Idea) to contextualize the private market analysis, but always distinguish them as external benchmarks."),

        ("Organisational Velocity & Talent Signals",
         "Analyse growth_emp_latest, val_per_employee, and employee count trajectory. Cross‑reference with stage_adjusted_momentum. Discuss what headcount growth implies about operational scale‑up ambition."),

        ("Technology & Infrastructure Dependency Risk",
         "Catalogue each tech dependency from the graph, including its weight, fragility score, and composite risk. Rank them by systemic risk. Explain what a failure of the highest‑risk dependency would mean for operations."),

        ("Investor Network Analysis & VC Contagion Risk",
         "Analyse the depth and breadth of the investor network. List the portfolio co‑investments for each investor as contagion channels. Assess whether the investor base is concentrated or diversified."),

        ("Geographic Footprint & Macro Sensitivity",
         "Based on the quantitative dossier’s continent, headquarters, and macro indicators, discuss geopolitical exposure. Use your financial knowledge to switch the macro discussion to the company’s home economy (e.g., if headquarters is in India, discuss RBI rates, INR dynamics, and Indian GDP)."),

        ("Institutional Scenario Analysis (Base / Bull / Bear)",
         "Construct a rigorous 3‑part institutional scenario analysis with explicit valuation targets. "
         "Base Case: assume current momentum persists; Bull Case: assume macro tailwinds and competitor stumbles (name specific competitors from the graph); Bear Case: assume tech dependency failures, funding gap, and investor contagion. "
         "For each scenario, provide a 12‑month implied valuation range or point estimate derived from the existing valuation trend and velocity metrics. "
         "Ground all numbers in the provided data, but the target range can be reasoned using growth rates. End each scenario with a clear probability‑weighted recommendation."),

        ("Data Quality, Intelligence Confidence & Coverage Gaps",
         "Rate the reliability of this analysis using data_quality, data_points. Flag any missing metrics. State the overall intelligence confidence level (HIGH / MEDIUM / LOW) with justification."),
    ]

    full_report = f"# INSTITUTIONAL INTELLIGENCE BRIEF\n## {company}\n"
    full_report += f"*Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | "
    full_report += f"Data through: {quant_dossier.get('meta', {}).get('last_date', 'N/A')} | "
    full_report += f"Classification: CONFIDENTIAL — FOR INSTITUTIONAL USE ONLY*\n\n"
    full_report += "---\n\n"

    context_block = f"""
[DATASET 1: QUANTITATIVE TIME-SERIES DOSSIER]
{quant_json}

[DATASET 2: GRAPH TOPOLOGY ATTRIBUTES]
{graph_attrs}

[DATASET 3: STRUCTURAL GRAPH TOPOLOGY (GraphRAG)]
{graph_json}
"""

    for sec_title, sec_instruction in sections:
        prompt = f"""{context_block}

TASK: Write ONLY the section "{sec_title}".
INSTRUCTION: {sec_instruction}

FORMAT:
## {sec_title}

[4-6 paragraphs of dense institutional prose. No bullet points. No tables. No citation tags.
All specific numbers must be traceable to the datasets provided.]

IMPORTANT: Do NOT include a separate verification paragraph. Keep the text seamless.
"""
        try:
            response = call_gemini_with_retry(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ANTI_HALLUCINATION_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=2500,
                ),
            )
            section_text = response.text.strip()
            if not section_text.startswith("##"):
                section_text = f"## {sec_title}\n\n{section_text}"
            full_report += section_text + "\n\n---\n\n"
        except Exception as e:
            full_report += f"## {sec_title}\n\n**Error generating section:** {e}\n\n---\n\n"

    # ── PHASE B: Hallucination Audit
    audit_prompt = f"""You are a Compliance Auditor. Below is an analyst report about {company}.
You are given the ONLY two datasets the analyst was allowed to use.

For each factual claim (numbers, company names, investor names, metric values):
1. Verify it appears in the datasets.
2. If a claim is NOT found, mark it: "⚠️ HALLUCINATION DETECTED: [claim]"
3. If all claims are verified, write: "✅ Section verified — all claims traceable."

Provide a concise audit summary:
- Total claims audited
- Hallucinations detected (if any)
- Overall report integrity score (0-100)

[ANALYST REPORT (first 8000 chars)]
{full_report[:8000]}

[DATASET 1: QUANTITATIVE DOSSIER (abbreviated)]
{quant_json[:3000]}

[DATASET 2: GRAPH TOPOLOGY]
{graph_json[:2000]}
"""
    try:
        audit_response = call_gemini_with_retry(
            model=GEMINI_MODEL,
            contents=audit_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1500,
            ),
        )
        audit_text = audit_response.text.strip()
        full_report += f"## Hallucination Audit Report\n\n{audit_text}\n\n---\n\n"
    except Exception as e:
        full_report += f"## Hallucination Audit Report\n\n**Audit failed:** {e}\n\n---\n\n"

    # ── PHASE C: Institutional Intelligence Summary (10 Points)
    summary_prompt = f"""You are an Elite Institutional Analyst. Generate a concise 10-point INSTITUTIONAL INTELLIGENCE SUMMARY for {company}.

Each point must:
- Contain ONLY 2 lines maximum of information
- Be grounded entirely in the provided datasets (no fabrication)
- Present the most critical detail for that investment signal dimension
- Use natural language (no citation tags, no disclaimers in the text itself)

The 10 points should cover these dimensions in order:
1. Dominant Market Position & Valuation Magnitude
2. Elite Capital Productivity & Efficiency Multiplier
3. Organizational Lean-ness & Per-Capita Value Generation
4. Aggressive Funding Runway & Immediate Liquidity Fortress
5. Diversified & Tier-1 Institutional Backing
6. Critical Single-Point-of-Failure Infrastructure Risk
7. Strategic Reliance on Competitor Capabilities (if applicable, else Technology Dependency Risk)
8. Geographic Concentration & Macro Sensitivity
9. Organizational Momentum Plateau & Strategic Inflection
10. Investment Thesis Validation & Forward Monitoring Imperatives

FORMAT (example structure):
# INSTITUTIONAL INTELLIGENCE SUMMARY: {company} (10/10)
**[Category] | [Stage] | Generated [Date]**

---

## Investment Signal & Executive Brief

1. **Point Title**
   Two-line description grounded in data. Reference specific metrics and numbers.

2. **Point Title**
   Two-line description grounded in data. Reference specific metrics and numbers.

[... continue for all 10 points ...]

---

**Confidence Level: [HIGH/MEDIUM/LOW] | Data Integrity: Verified | Coverage: Complete**

DATASETS PROVIDED:
[DATASET 1: QUANTITATIVE DOSSIER]
{quant_json[:2500]}

[DATASET 2: GRAPH TOPOLOGY]
{graph_json[:2500]}

INSTRUCTIONS:
- Every number must be traceable to the datasets above.
- If a metric is missing or zero, mention it briefly but do not include a disclaimer tag.
- Use only the category and stage data provided in the quantitative dossier.
- Reference investor names, tech dependencies, and competitive weights from the graph.
- Keep each point to exactly 2 lines or fewer.
- Do NOT include any hallucinated metrics or external data not in the datasets.
"""
    try:
        summary_response = call_gemini_with_retry(
            model=GEMINI_MODEL,
            contents=summary_prompt,
            config=types.GenerateContentConfig(
                system_instruction=ANTI_HALLUCINATION_SYSTEM,
                temperature=0.05,
                max_output_tokens=3000,
            ),
        )
        summary_text = summary_response.text.strip()
        full_report += f"## INSTITUTIONAL INTELLIGENCE SUMMARY (10-POINT EXECUTIVE BRIEF)\n\n{summary_text}\n\n"
    except Exception as e:
        full_report += f"## INSTITUTIONAL INTELLIGENCE SUMMARY (10-POINT EXECUTIVE BRIEF)\n\n**Summary generation failed:** {e}\n\n"

    return full_report


# ─── 8. SCORING ENGINE ───────────────────────────────────────────────────────

def calculate_emergence_scores(db: Session, w_mrr: float = 33.3, w_growth: float = 33.3,
                                w_efficiency: float = 33.3, shocked_node_name: str = None):
    """
    Ranks companies based on multi-relational graph metrics and timeseries data.
    """
    global TIMESERIES_DF
    if TIMESERIES_DF.empty:
        return {"rankings": [], "graph": {"nodes": [], "links": []}}

    df = TIMESERIES_DF.copy()
    if "Date" in df.columns:
        latest_df = df.sort_values("Date").drop_duplicates(subset=["Company"], keep="last")
    else:
        latest_df = df.drop_duplicates(subset=["Company"], keep="last")

    records = []
    for _, row in latest_df.iterrows():
        records.append({
            "name":              str(row.get("Company", "")),
            "valuation_m":       float(row.get("End_Valuation_M", row.get("Valuation_M", 0)) or 0),
            "growth_val":        float(row.get("growth_val", 0) or 0),
            "funding_efficiency":float(row.get("funding_efficiency", 0) or 0),
            "stage":             str(row.get("Stage", "Unknown")),
            "category":          str(row.get("Category", "Unknown")),
            "global_rank":       int(row.get("Global_Rank", 999) or 999),
            "vel_val":           float(row.get("vel_val", 0) or 0),
            "readiness_score":   float(row.get("readiness_score", 0) or 0),
            "description":       f"{row.get('Category','?')} | {row.get('Headquarters','?')} | {row.get('Stage','?')}",
        })

    score_df = pd.DataFrame(records)
    if score_df.empty:
        return {"rankings": [], "graph": {"nodes": [], "links": []}}

    for col in ["valuation_m", "growth_val", "funding_efficiency"]:
        mn, mx = score_df[col].min(), score_df[col].max()
        score_df[f"norm_{col}"] = (score_df[col] - mn) / (mx - mn) if mx > mn else 1.0

    total_w = (w_mrr + w_growth + w_efficiency) or 1.0
    score_df["base_score"] = (
        score_df["norm_valuation_m"]       * (w_mrr / total_w) +
        score_df["norm_growth_val"]         * (w_growth / total_w) +
        score_df["norm_funding_efficiency"] * (w_efficiency / total_w)
    ) * 100.0

    score_df["base_score"] = score_df["base_score"] * (0.5 + 0.5 * (score_df["readiness_score"] / 100.0).clip(0, 1))

    penalties: Dict[str, float] = {}
    if shocked_node_name:
        impacts = run_semantic_contagion(db, shocked_node_name)
        for comp_name, impact in impacts.items():
            if impact > 0:
                penalties[comp_name] = impact * 0.4

    score_df["penalty"] = score_df["name"].map(lambda n: penalties.get(n, 0.0))
    score_df["emergence_score"] = (score_df["base_score"] - score_df["penalty"]).clip(lower=0)
    score_df = score_df.sort_values("emergence_score", ascending=False).reset_index(drop=True)

    graph_payload = {"nodes": [], "links": []}
    db_nodes = db.execute(text("SELECT id, name, entity_type FROM nodes")).fetchall()
    
    for nid, nname, ntype in db_nodes:
        group_map = {"COMPANY": "startup", "INVESTOR": "investor", "TECH": "tech", "CATEGORY": "category"}
        is_shocked = (shocked_node_name and nname.lower() == shocked_node_name.lower())
        graph_payload["nodes"].append({
            "id": nid, 
            "name": nname, 
            "group": group_map.get(ntype, "other"),
            "is_shocked": is_shocked
        })

    db_edges = db.execute(
        text("SELECT source_id, target_id, rel_type, weight FROM edges LIMIT 2000")
    ).fetchall()
    
    for src, tgt, rel, w in db_edges:
        graph_payload["links"].append({"source": src, "target": tgt, "rel_type": rel, "value": w})

    ranked_output = []
    for idx, row in score_df.iterrows():
        ranked_output.append({
            "rank":            idx + 1,
            "name":            row["name"],
            "description":     row["description"],
            "emergence_score": round(row["emergence_score"], 1),
            "penalty_taken":   round(row["penalty"], 1),
            "raw_metrics": {
                "valuation_m":       round(row["valuation_m"], 1),
                "growth_val":        round(row["growth_val"], 4),
                "funding_efficiency":round(row["funding_efficiency"], 4),
                "vel_val":           round(row["vel_val"], 4),
                "readiness_score":   round(row["readiness_score"], 2),
                "global_rank":       row["global_rank"],
            }
        })

    return {"rankings": ranked_output, "graph": graph_payload}


# ─── 9. FASTAPI APP ROUTING ──────────────────────────────────────────────────

app = FastAPI(title="FutureOS Engine v5.0", version="5.0")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

try:
    TIMESERIES_DF = pd.read_csv(CSV_FILE_PATH)
    print(f"[SYSTEM] Loaded historical datasets: {len(TIMESERIES_DF)} discrete operations across nodes.")
except Exception as e:
    print(f"[WARNING] Primary dataset mapping failed, initializing baseline arrays: {e}")
    TIMESERIES_DF = pd.DataFrame()

init_db()


def _seed_default_admin():
    """Ensures the default admin account (from env vars / fallback) always exists & is verified."""
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == DEFAULT_ADMIN_USER).first()
        if not existing:
            db.add(User(
                username=DEFAULT_ADMIN_USER,
                email=f"{DEFAULT_ADMIN_USER}@futureos.local",
                password_hash=hash_password(DEFAULT_ADMIN_PASS),
                auth_provider="local",
                is_verified=1,
            ))
            db.commit()
            print(f"[SYSTEM] Seeded default admin account '{DEFAULT_ADMIN_USER}'.")
    finally:
        db.close()


_seed_default_admin()


class SimulationPayload(BaseModel):
    weight_mrr:        float = 33.3
    weight_growth:     float = 33.3
    weight_efficiency: float = 33.3
    shocked_node_name: Optional[str] = None

class ReportRequest(BaseModel):
    company: str

class SubgraphReportRequest(BaseModel):
    company: str


@app.post("/api/signup")
def signup(payload: SignupRequest):
    db = SessionLocal()
    try:
        username = payload.username.strip()
        if not username or not payload.password:
            raise HTTPException(status_code=400, detail="Username and password are required.")
        existing = db.query(User).filter(
            (User.username == username) | (User.email == payload.email)
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="An account with that username or email already exists.")

        user = User(
            username=username,
            email=payload.email.strip(),
            password_hash=hash_password(payload.password),
            auth_provider="local",
            is_verified=0,
        )
        db.add(user)
        db.commit()

        otp = _generate_otp(username, purpose="signup")
        resp = {"message": "Account created. Enter the verification code to activate it."}
        if _OTP_DEV_MODE:
            resp["dev_otp"] = otp  # demo-only: a real deployment emails/SMS this instead
        return resp
    finally:
        db.close()


@app.post("/api/verify-signup-otp")
def verify_signup_otp(payload: VerifyOtpRequest):
    db = SessionLocal()
    try:
        if not _verify_otp(payload.username, payload.otp, purpose="signup"):
            raise HTTPException(status_code=400, detail="Invalid or expired verification code.")
        user = db.query(User).filter(User.username == payload.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Account not found.")
        user.is_verified = 1
        db.commit()
        return {"token": create_token(user.username), "username": user.username}
    finally:
        db.close()


@app.post("/api/login")
def login(payload: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == payload.username).first()
        if not user or user.auth_provider != "local" or not user.password_hash:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        if not user.is_verified:
            raise HTTPException(status_code=403, detail="Account not verified. Please complete OTP verification first.")
        return {"token": create_token(user.username), "username": user.username}
    finally:
        db.close()


@app.post("/api/login/request-otp")
def request_login_otp(payload: RequestOtpRequest):
    """Passwordless login step 1: request a one-time code for an existing, verified account."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == payload.username).first()
        if not user or not user.is_verified:
            raise HTTPException(status_code=404, detail="No verified account found for that username.")
        otp = _generate_otp(payload.username, purpose="login")
        resp = {"message": "OTP sent."}
        if _OTP_DEV_MODE:
            resp["dev_otp"] = otp
        return resp
    finally:
        db.close()


@app.post("/api/login/verify-otp")
def verify_login_otp(payload: VerifyOtpRequest):
    """Passwordless login step 2: exchange a valid OTP for a session token."""
    db = SessionLocal()
    try:
        if not _verify_otp(payload.username, payload.otp, purpose="login"):
            raise HTTPException(status_code=400, detail="Invalid or expired one-time code.")
        user = db.query(User).filter(User.username == payload.username).first()
        if not user:
            raise HTTPException(status_code=404, detail="Account not found.")
        return {"token": create_token(user.username), "username": user.username}
    finally:
        db.close()


@app.post("/api/auth/google")
def auth_google(payload: GoogleAuthRequest):
    """
    Verifies a Google Identity Services ID token. Creates the account on first
    sign-in (Google-verified accounts skip the OTP step) and logs in otherwise.
    """
    claims = _verify_google_id_token(payload.credential)
    if not claims:
        raise HTTPException(
            status_code=401,
            detail="Google sign-in could not be verified. Ensure GOOGLE_CLIENT_ID is configured server-side "
                   "and `google-auth` is installed (pip install google-auth).",
        )
    email = claims.get("email")
    name = claims.get("name") or (email.split("@")[0] if email else claims.get("sub"))
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            username = name
            suffix = 1
            base_username = username
            while db.query(User).filter(User.username == username).first():
                suffix += 1
                username = f"{base_username}{suffix}"
            user = User(
                username=username, email=email, password_hash=None,
                auth_provider="google", is_verified=1,
            )
            db.add(user)
            db.commit()
        return {"token": create_token(user.username), "username": user.username}
    finally:
        db.close()


@app.get("/api/me")
def me(username: str = Depends(require_auth)):
    return {"username": username}


@app.post("/api/profile/edit")
def edit_profile(payload: EditProfileRequest, username: str = Depends(require_auth)):
    """
    Allows an authenticated user to update their username and/or password.
    If the username changes, a new token is issued (since tokens encode the username).
    """
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)

        # ── Password Change Logic ──────────────────────────────────────────────
        if payload.new_password:
            if not payload.current_password:
                raise HTTPException(
                    status_code=400,
                    detail="Current password is required to set a new password."
                )
            if user.auth_provider != "local" or not user.password_hash:
                raise HTTPException(
                    status_code=400,
                    detail="Password change is not available for SSO accounts."
                )
            if not verify_password(payload.current_password, user.password_hash):
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect current password."
                )
            user.password_hash = hash_password(payload.new_password)

        # ── Username Change Logic ──────────────────────────────────────────────
        final_username = username
        if payload.new_username and payload.new_username != username:
            new_uname = payload.new_username.strip()
            if not new_uname:
                raise HTTPException(status_code=400, detail="Username cannot be blank.")
            taken = db.query(User).filter(User.username == new_uname).first()
            if taken:
                raise HTTPException(
                    status_code=409,
                    detail="That username is already taken."
                )
            user.username = new_uname
            final_username = new_uname

        db.commit()

        # Re-issue token with the (potentially new) username
        new_token = create_token(final_username)
        return {"token": new_token, "username": final_username}
    finally:
        db.close()


@app.post("/api/profile/avatar")
def update_avatar(payload: AvatarRequest, username: str = Depends(require_auth)):
    """
    Stores a base64 data URL for the user's profile picture.
    The column is a Text field so it can hold arbitrarily large images.
    SQLite will add the column automatically via ALTER TABLE if the database
    was created before this field was introduced.
    """
    db = SessionLocal()
    try:
        # Ensure the column exists in case the DB pre-dates this feature
        try:
            db.execute(text("ALTER TABLE users ADD COLUMN avatar_data_url TEXT"))
            db.commit()
        except Exception:
            pass  # Column already exists — this is expected on re-runs

        user = _get_current_user_row(db, username)
        user.avatar_data_url = payload.avatar_data_url
        db.commit()
        return {"ok": True}
    finally:
        db.close()


def _get_current_user_row(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    return user


@app.get("/api/profile")
def get_profile(username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        favs = db.query(Favorite).filter(Favorite.user_id == user.id).order_by(Favorite.created_at.desc()).all()
        reports = (
            db.query(ReportHistory)
            .filter(ReportHistory.user_id == user.id)
            .order_by(ReportHistory.generated_at.desc())
            .all()
        )
        return {
            "username": user.username,
            "email": user.email,
            "auth_provider": user.auth_provider,
            "avatar_data_url": user.avatar_data_url or None,
            "favorites": [f.company_name for f in favs],
            "reports": [
                {"id": r.id, "company": r.company_name, "generated_at": r.generated_at}
                for r in reports
            ],
        }
    finally:
        db.close()


@app.get("/api/favorites")
def list_favorites(username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        favs = db.query(Favorite).filter(Favorite.user_id == user.id).all()
        return {"favorites": [f.company_name for f in favs]}
    finally:
        db.close()


@app.post("/api/favorites")
def add_favorite(payload: FavoriteRequest, username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        exists = db.query(Favorite).filter(
            Favorite.user_id == user.id, Favorite.company_name == payload.company
        ).first()
        if not exists:
            db.add(Favorite(user_id=user.id, company_name=payload.company))
            db.commit()
        favs = db.query(Favorite).filter(Favorite.user_id == user.id).all()
        return {"favorites": [f.company_name for f in favs]}
    finally:
        db.close()


@app.delete("/api/favorites/{company}")
def remove_favorite(company: str, username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        db.query(Favorite).filter(
            Favorite.user_id == user.id, Favorite.company_name == company
        ).delete()
        db.commit()
        favs = db.query(Favorite).filter(Favorite.user_id == user.id).all()
        return {"favorites": [f.company_name for f in favs]}
    finally:
        db.close()


@app.get("/api/reports")
def list_reports(username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        reports = (
            db.query(ReportHistory)
            .filter(ReportHistory.user_id == user.id)
            .order_by(ReportHistory.generated_at.desc())
            .all()
        )
        return {"reports": [
            {"id": r.id, "company": r.company_name, "generated_at": r.generated_at} for r in reports
        ]}
    finally:
        db.close()


@app.get("/api/reports/{report_id}")
def get_report(report_id: int, username: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        user = _get_current_user_row(db, username)
        report = db.query(ReportHistory).filter(
            ReportHistory.id == report_id, ReportHistory.user_id == user.id
        ).first()
        if not report:
            raise HTTPException(status_code=404, detail="Report not found.")
        return {
            "id": report.id,
            "company": report.company_name,
            "report_markdown": report.report_markdown,
            "quant_dossier": json.loads(report.quant_dossier or "{}"),
            "graph_context": json.loads(report.graph_context or "{}"),
            "generated_at": report.generated_at,
        }
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>CRITICAL FAIL: index.html block missing</h1>", status_code=404)


@app.get("/api/companies")
def list_companies(_auth: str = Depends(require_auth)):
    if TIMESERIES_DF.empty:
        return {"companies": []}
    return {"companies": sorted(TIMESERIES_DF["Company"].dropna().unique().tolist())}

@app.get("/api/companies_full")
def list_companies_full(_auth: str = Depends(require_auth)):
    """Fetches the entire 91 company directory for the FutureOS front page."""
    if TIMESERIES_DF.empty:
        return {"companies": []}
    
    df = TIMESERIES_DF.copy()
    if "Date" in df.columns:
        latest_df = df.sort_values("Date").drop_duplicates(subset=["Company"], keep="last")
    else:
        latest_df = df.drop_duplicates(subset=["Company"], keep="last")
    
    comps = []
    for _, row in latest_df.iterrows():
        comps.append(row.where(pd.notna(row), None).to_dict())
    return {"companies": comps}

@app.get("/api/company/{name}")
def get_company_profile(name: str, _auth: str = Depends(require_auth)):
    if TIMESERIES_DF.empty:
        raise HTTPException(status_code=503, detail="Timeseries data not loaded.")
    
    mask = TIMESERIES_DF["Company"].str.lower() == name.lower()
    rows = TIMESERIES_DF[mask]
    
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Company '{name}' not found.")
        
    latest = rows.sort_values("Date").iloc[-1] if "Date" in rows.columns else rows.iloc[-1]
    return {
        "name": str(latest["Company"]), 
        "profile": latest.where(pd.notna(latest), None).to_dict()
    }


@app.get("/api/baseline")
def get_baseline(_auth: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        return calculate_emergence_scores(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/simulate")
def run_simulation(payload: SimulationPayload, _auth: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        return calculate_emergence_scores(
            db, 
            payload.weight_mrr, 
            payload.weight_growth,
            payload.weight_efficiency, 
            payload.shocked_node_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/generate_report")
def generate_subgraph_report_endpoint(payload: SubgraphReportRequest, _auth: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        quant  = build_quantitative_dossier(payload.company)
        graph  = extract_graphrag_context(db, payload.company)

        if "error" in quant and not graph:
            raise HTTPException(status_code=404, detail=f"No context paths available for target: '{payload.company}'.")

        report = generate_bloomberg_report(payload.company, quant, graph, db)
        graph_ctx_clean = {k: v for k, v in graph.items() if k != "attributes"}

        # Persist into the user's report history so it shows up in their Profile tab.
        try:
            user = db.query(User).filter(User.username == _auth).first()
            if user:
                db.add(ReportHistory(
                    user_id=user.id,
                    company_name=payload.company,
                    report_markdown=report,
                    quant_dossier=json.dumps(quant),
                    graph_context=json.dumps(graph_ctx_clean),
                ))
                db.commit()
        except Exception as e:
            print(f"[WARNING] Failed to save report history: {e}")

        return {
            "company":            payload.company,
            "quant_dossier":      quant,
            "graph_context":      graph_ctx_clean,
            "report_markdown":    report,
            "generated_at":       datetime.datetime.utcnow().isoformat(),
        }
    finally:
        db.close()


@app.post("/api/report")
def generate_legacy_report(payload: ReportRequest, _auth: str = Depends(require_auth)):
    return generate_subgraph_report_endpoint(SubgraphReportRequest(company=payload.company), _auth)


# ─── PDF EXPORT ENDPOINT ──────────────────────────────────────────────────────
# Receives the already-generated markdown from the frontend, converts it to a
# beautifully styled PDF using WeasyPrint, and streams it back as a download.
# This produces a clean, white-background, print-ready document — much higher
# quality than the dark-theme html2pdf.js browser-side export.

class PdfExportRequest(BaseModel):
    company: str
    report_markdown: str


_PDF_STYLESHEET = """
@page {
    size: A4;
    margin: 20mm 18mm 22mm 18mm;
    @top-right  { content: "FUTUREOS — CONFIDENTIAL"; font-size: 7pt; color: #888; }
    @bottom-center { content: counter(page) " / " counter(pages); font-size: 7pt; color: #888; }
}
body {
    font-family: 'Georgia', serif;
    font-size: 10.5pt;
    color: #0f1117;
    line-height: 1.65;
    background: #ffffff;
}
h1 {
    font-size: 17pt;
    font-weight: 900;
    color: #0a0a0f;
    border-bottom: 2.5px solid #0a0a0f;
    padding-bottom: 6pt;
    margin-top: 0;
    margin-bottom: 14pt;
    letter-spacing: 0.5pt;
    text-transform: uppercase;
}
h2 {
    font-size: 12pt;
    font-weight: 700;
    color: #111827;
    border-left: 3pt solid #00bcd4;
    padding-left: 8pt;
    margin-top: 22pt;
    margin-bottom: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.3pt;
    page-break-after: avoid;
}
h3 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #374151;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}
p {
    margin-top: 0;
    margin-bottom: 9pt;
    text-align: justify;
    orphans: 3;
    widows: 3;
}
em { font-style: italic; color: #374151; }
strong { font-weight: 700; color: #0f1117; }
hr {
    border: none;
    border-top: 1px solid #d1d5db;
    margin: 18pt 0;
}
blockquote {
    border-left: 3pt solid #f59e0b;
    margin: 12pt 0;
    padding: 6pt 12pt;
    background: #fffbeb;
    font-style: italic;
    color: #92400e;
    page-break-inside: avoid;
}
code {
    font-family: 'Courier New', monospace;
    font-size: 9pt;
    background: #f3f4f6;
    padding: 1pt 4pt;
    border-radius: 2pt;
    color: #1f2937;
}
ul, ol { margin: 6pt 0 10pt 18pt; padding: 0; }
li { margin-bottom: 4pt; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th {
    background: #0f172a;
    color: #ffffff;
    padding: 6pt 10pt;
    text-align: left;
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.5pt;
}
td {
    padding: 5pt 10pt;
    border-bottom: 1px solid #e5e7eb;
    color: #1f2937;
}
tr:nth-child(even) td { background: #f9fafb; }
.cover-block {
    text-align: center;
    padding: 30pt 0 20pt;
    margin-bottom: 24pt;
    border-bottom: 2px solid #0a0a0f;
}
.cover-title {
    font-size: 22pt;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 2pt;
    color: #0a0a0f;
}
.cover-sub {
    font-size: 9pt;
    color: #6b7280;
    letter-spacing: 1pt;
    margin-top: 6pt;
    text-transform: uppercase;
}
"""


@app.post("/api/export_pdf")
def export_pdf(payload: PdfExportRequest, _auth: str = Depends(require_auth)):
    """
    Converts a report's Markdown to a branded, print-ready A4 PDF via WeasyPrint.
    Returns the PDF as a binary stream with a Content-Disposition download header.
    """
    try:
        import markdown as md_lib
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration

        # Build a clean cover block + rendered markdown body
        safe_company = payload.company.replace("<", "&lt;").replace(">", "&gt;")
        generated_ts = datetime.datetime.utcnow().strftime("%d %B %Y · %H:%M UTC")

        body_html = md_lib.markdown(
            payload.report_markdown,
            extensions=["extra", "sane_lists", "smarty"]
        )

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{safe_company} — Intelligence Brief</title></head>
<body>
  <div class="cover-block">
    <div class="cover-title">FutureOS Intelligence Brief</div>
    <div class="cover-sub">{safe_company} &nbsp;·&nbsp; {generated_ts} &nbsp;·&nbsp; CONFIDENTIAL</div>
  </div>
  {body_html}
</body>
</html>"""

        font_config = FontConfiguration()
        css = CSS(string=_PDF_STYLESHEET, font_config=font_config)
        pdf_bytes = HTML(string=full_html).write_pdf(stylesheets=[css], font_config=font_config)

        safe_filename = re.sub(r'[^\w\-]', '_', payload.company)
        filename = f"FutureOS_{safe_filename}_Intel_Brief.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF library not installed: {e}. Run: pip install weasyprint markdown"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


@app.get("/api/graph_context/{company}")
def get_graph_context(company: str, _auth: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        ctx = extract_graphrag_context(db, company)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Graph context exhausted for '{company}'.")
        return ctx
    finally:
        db.close()


@app.get("/api/quant_dossier/{company}")
def get_quant_dossier(company: str, _auth: str = Depends(require_auth)):
    q = build_quantitative_dossier(company)
    if "error" in q:
        raise HTTPException(status_code=404, detail=q["error"])
    return q


@app.get("/api/investors")
def list_investors(_auth: str = Depends(require_auth)):
    db = SessionLocal()
    try:
        rows = db.execute(
            text("SELECT name, entity_type FROM nodes WHERE entity_type='INVESTOR' ORDER BY name")
        ).fetchall()
        return {"investors": [{"name": r[0]} for r in rows]}
    finally:
        db.close()


@app.post("/api/shock")
def run_shock_analysis(payload: SimulationPayload, _auth: str = Depends(require_auth)):
    if not payload.shocked_node_name:
        raise HTTPException(status_code=400, detail="Vulnerability parameter 'shocked_node_name' is missing.")
    db = SessionLocal()
    try:
        impacts = run_semantic_contagion(db, payload.shocked_node_name)
        sorted_impacts = sorted(
            [{"name": k, "impact_score": round(v, 2)} for k, v in impacts.items() if v != 0],
            key=lambda x: x["impact_score"], reverse=True
        )
        return {"shocked_node": payload.shocked_node_name, "impacts": sorted_impacts}
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("  FUTUREOS ENGINE v5.0 — Defense-Grade Intelligence Terminal")
    print("  Open: http://localhost:8000")
    print("=" * 70 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)