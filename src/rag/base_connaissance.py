import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = Path(__file__).parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "faq_service_client"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _charger_faq(chemin: str | Path) -> list[dict]:
    entrees = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                entrees.append(json.loads(ligne))
    return entrees


def creer_base_connaissance(
    chemin_faq: str | Path = None,
    chroma_path: str | Path = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    chemin_faq = chemin_faq or Path(__file__).parent.parent.parent / "data" / "faq_service_client.jsonl"
    chroma_path = chroma_path or CHROMA_PATH

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Ne pas réinsérer si déjà peuplé
    if collection.count() > 0:
        return collection

    entrees = _charger_faq(chemin_faq)
    modele = SentenceTransformer(EMBEDDING_MODEL)

    # On encode la question (pas la réponse) : la recherche se fait question → question
    questions = [e["question"] for e in entrees]
    embeddings = modele.encode(questions, normalize_embeddings=True).tolist()

    collection.add(
        ids=[e["id"] for e in entrees],
        documents=[e["question"] for e in entrees],
        embeddings=embeddings,
        metadatas=[{"reponse": e["reponse"], "categorie": e["categorie"]} for e in entrees],
    )

    return collection


def get_collection(
    chroma_path: str | Path = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    chroma_path = chroma_path or CHROMA_PATH
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
