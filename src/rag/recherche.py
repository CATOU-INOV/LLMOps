import chromadb
from sentence_transformers import SentenceTransformer

SEUIL_SIMILARITE = 0.4


def rechercher_documents(
    requete: str,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    top_k: int = 3,
    seuil: float = SEUIL_SIMILARITE,
) -> list[dict]:
    """Recherche sémantique dans ChromaDB.

    Retourne une liste de dicts {id, document, reponse, score} filtrés
    au-dessus du seuil de similarité.
    """
    vecteur = modele_embedding.encode([requete], normalize_embeddings=True).tolist()

    resultats = collection.query(
        query_embeddings=vecteur,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i, dist in enumerate(resultats["distances"][0]):
        # Distance cosinus normalisée : 0 = identique, 2 = opposé
        # Similarité = 1 - dist
        similarite = 1.0 - dist
        if similarite >= seuil:
            docs.append({
                "id": resultats["ids"][0][i],
                "document": resultats["documents"][0][i],
                "reponse": resultats["metadatas"][0][i]["reponse"],
                "score": similarite,
            })

    return docs
