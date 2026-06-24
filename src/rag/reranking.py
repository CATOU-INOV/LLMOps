from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reclasser_passages(
    requete: str,
    candidats: list[dict],
    modele_crossencoder: CrossEncoder,
    top_k_final: int = 3,
) -> list[dict]:
    """Reclasse les candidats du bi-encoder avec un cross-encoder.

    Entrée : liste de dicts {id, document, reponse, score}
    Sortie : top_k_final meilleurs candidats triés par score décroissant
    """
    if not candidats:
        return []

    paires = [(requete, c["document"]) for c in candidats]
    scores = modele_crossencoder.predict(paires).tolist()

    for candidat, score in zip(candidats, scores):
        candidat["score_reranking"] = score

    reclasses = sorted(candidats, key=lambda x: x["score_reranking"], reverse=True)
    return reclasses[:top_k_final]
