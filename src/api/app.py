from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.base_connaissance import creer_base_connaissance
from src.rag.pipeline import generer_avec_rag

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LOG_FILE = Path(os.getenv("LOG_FILE", "logs/requetes.jsonl"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Métriques Prometheus personnalisées ──────────────────────────────────────
rag_docs_retrieved = Counter(
    "rag_docs_retrieved_total",
    "Nombre total de documents RAG retrouvés",
)
rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "Durée de la recherche ChromaDB",
)
llm_generation_duration = Histogram(
    "llm_generation_duration_seconds",
    "Durée de l'inférence LLM",
)
llm_tokens_generated = Counter(
    "llm_tokens_generated_total",
    "Tokens générés au total",
)
rag_hors_domaine = Counter(
    "rag_hors_domaine_total",
    "Requêtes sans document pertinent trouvé",
)

# ── État global chargé une seule fois au démarrage ───────────────────────────
_state: dict = {}


def _init_langfuse():
    """Initialise le client Langfuse si les clés sont présentes."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    if pk.startswith("pk-lf-") and not pk.endswith("REMPLACER") and sk.startswith("sk-lf-"):
        try:
            from langfuse import Langfuse

            client = Langfuse()
            client.auth_check()
            logger.info("Langfuse connecté.")
            return client
        except Exception as exc:
            logger.warning("Langfuse indisponible : %s", exc)
    else:
        logger.info("Clés Langfuse non configurées — traces désactivées.")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Chargement des modèles...")
    _state["modele_embedding"] = SentenceTransformer(EMBEDDING_MODEL)
    _state["modele_crossencoder"] = CrossEncoder(CROSS_ENCODER_MODEL)
    _state["tokeniseur"] = AutoTokenizer.from_pretrained(LLM_MODEL)
    _state["modele_llm"] = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    _state["collection"] = creer_base_connaissance()
    _state["langfuse"] = _init_langfuse()

    # Prompt versionné depuis Langfuse (avec fallback local)
    _state["prompt_template"] = _charger_prompt_langfuse(_state["langfuse"])

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("API prête.")
    yield
    lf = _state.get("langfuse")
    if lf:
        lf.flush()
    _state.clear()


def _charger_prompt_langfuse(langfuse_client) -> str | None:
    if langfuse_client is None:
        return None
    try:
        prompt_obj = langfuse_client.get_prompt("assistant-service-client")
        logger.info("Prompt Langfuse chargé (version %s).", prompt_obj.version)
        return prompt_obj
    except Exception as exc:
        logger.warning("Impossible de charger le prompt Langfuse : %s", exc)
        return None


app = FastAPI(
    title="LLMOps RAG API",
    description="Assistant conversationnel avec pipeline RAG, Langfuse et Prometheus",
    version="0.1.0",
    lifespan=lifespan,
)

# Instrumentation automatique HTTP → expose /metrics
Instrumentator().instrument(app).expose(app)


# ── Schémas Pydantic ─────────────────────────────────────────────────────────


class RequeteGeneration(BaseModel):
    prompt: str = Field(..., min_length=1)
    nb_tokens_max: int = Field(200, ge=1, le=500)


class ReponseGeneration(BaseModel):
    reponse: str
    nb_tokens: int
    documents_sources: list[str]
    hors_domaine: bool
    duree_ms: float


class ReponseHealth(BaseModel):
    status: str
    modele: str


# ── Gestionnaire erreur validation ───────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"erreur": "Validation échouée", "details": exc.errors()},
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=ReponseHealth)
async def health():
    if not _state.get("modele_llm"):
        raise HTTPException(status_code=503, detail="Service non prêt")
    return {"status": "ok", "modele": LLM_MODEL}


@app.post("/generate", response_model=ReponseGeneration)
async def generate(requete: RequeteGeneration):
    if not _state.get("modele_llm"):
        raise HTTPException(status_code=503, detail="Service non prêt")

    lf = _state.get("langfuse")
    trace = None
    if lf:
        trace = lf.trace(
            name="rag-generate",
            input=requete.prompt,
            metadata={"nb_tokens_max": requete.nb_tokens_max, "modele": LLM_MODEL},
        )

    debut = time.perf_counter()
    try:
        resultat = generer_avec_rag(
            prompt_utilisateur=requete.prompt,
            collection=_state["collection"],
            modele_embedding=_state["modele_embedding"],
            modele_llm=_state["modele_llm"],
            tokeniseur=_state["tokeniseur"],
            modele_crossencoder=_state["modele_crossencoder"],
            nb_tokens_max=requete.nb_tokens_max,
            trace=trace,
        )
    except Exception as exc:
        logger.error("Erreur pipeline RAG : %s", exc, exc_info=True)
        if trace:
            trace.update(output=f"ERREUR: {exc}")
        raise HTTPException(status_code=500, detail="Erreur interne du pipeline RAG") from exc

    duree_ms = (time.perf_counter() - debut) * 1000

    # ── Métriques Prometheus ─────────────────────────────────────────────────
    rag_docs_retrieved.inc(resultat["nb_docs"])
    llm_tokens_generated.inc(resultat.get("nb_tokens_out", 0))
    if resultat["hors_domaine"]:
        rag_hors_domaine.inc()

    # ── Scores Langfuse ──────────────────────────────────────────────────────
    if trace:
        trace.score(name="nb_docs_retrouves", value=resultat["nb_docs"])
        trace.score(name="score_reranking_max", value=resultat["score_reranking_max"])
        trace.score(name="longueur_reponse_mots", value=len(resultat["reponse"].split()))
        trace.score(name="hors_domaine", value=int(resultat["hors_domaine"]))
        trace.update(output=resultat["reponse"])

    _logger_requete(
        prompt=requete.prompt,
        reponse=resultat["reponse"],
        duree_ms=duree_ms,
        documents_sources=resultat["documents_sources"],
        hors_domaine=resultat["hors_domaine"],
    )

    return ReponseGeneration(
        reponse=resultat["reponse"],
        nb_tokens=resultat.get("nb_tokens_out", len(resultat["reponse"].split())),
        documents_sources=resultat["documents_sources"],
        hors_domaine=resultat["hors_domaine"],
        duree_ms=round(duree_ms, 2),
    )


# ── Logging JSONL ─────────────────────────────────────────────────────────────


def _logger_requete(
    prompt: str,
    reponse: str,
    duree_ms: float,
    documents_sources: list[str],
    hors_domaine: bool,
) -> None:
    entree = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": prompt,
        "reponse": reponse,
        "longueur_prompt": len(prompt.split()),
        "longueur_reponse": len(reponse.split()),
        "duree_ms": round(duree_ms, 2),
        "documents_sources": documents_sources,
        "nb_docs_rag_trouves": len(documents_sources),
        "hors_domaine": hors_domaine,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entree, ensure_ascii=False) + "\n")
