import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def decouper_document(
    texte: str,
    taille_chunk: int = 200,
    chevauchement: int = 40,
) -> list[str]:
    """Découpe un texte en chunks avec chevauchement (fenêtre glissante sur les mots)."""
    if not texte.strip():
        return []

    mots = texte.split()
    if len(mots) <= taille_chunk:
        return [texte]

    chunks = []
    increment = taille_chunk - chevauchement
    debut = 0
    while debut < len(mots):
        fin = debut + taille_chunk
        chunk = " ".join(mots[debut:fin])
        chunks.append(chunk)
        if fin >= len(mots):
            break
        debut += increment

    return chunks


def indexer_documents_longs(
    chemin_jsonl: str | Path,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    taille_chunk: int = 200,
    chevauchement: int = 40,
) -> int:
    """Charge un JSONL de documents longs, découpe et indexe dans ChromaDB."""
    chunks_total = 0
    with open(chemin_jsonl, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            doc = json.loads(ligne)
            texte = doc.get("texte", doc.get("reponse", ""))
            chunks = decouper_document(texte, taille_chunk, chevauchement)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}__chunk_{i}"
                vecteur = modele_embedding.encode([chunk], normalize_embeddings=True).tolist()
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=vecteur,
                    metadatas=[{"source_id": doc["id"], "chunk_index": i}],
                )
                chunks_total += 1

    return chunks_total
