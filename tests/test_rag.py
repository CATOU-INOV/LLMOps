import pytest
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.rag.recherche import rechercher_documents
from src.rag.reranking import reclasser_passages

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FAQ_TEST = [
    {"id": "faq-01", "question": "Quel est le délai de retour ?", "reponse": "Retours acceptés sous 30 jours."},
    {"id": "faq-02", "question": "Comment suivre ma commande ?", "reponse": "Un email de suivi est envoyé à l'expédition."},
    {"id": "faq-03", "question": "Comment obtenir un remboursement ?", "reponse": "Remboursement sous 5 à 10 jours ouvrés."},
    {"id": "faq-04", "question": "Livrez-vous à l'international ?", "reponse": "Livraison dans 30 pays, 7 à 15 jours."},
    {"id": "faq-05", "question": "Comment annuler une commande ?", "reponse": "Annulation possible dans les 2 heures."},
]


@pytest.fixture(scope="module")
def modele_embedding():
    return SentenceTransformer(EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def modele_crossencoder():
    return CrossEncoder(CROSS_ENCODER_MODEL)


@pytest.fixture(scope="module")
def collection_test(modele_embedding):
    client = chromadb.EphemeralClient()
    col = client.create_collection(name="test_faq", metadata={"hnsw:space": "cosine"})

    questions = [e["question"] for e in FAQ_TEST]
    embeddings = modele_embedding.encode(questions, normalize_embeddings=True).tolist()
    col.add(
        ids=[e["id"] for e in FAQ_TEST],
        documents=questions,
        embeddings=embeddings,
        metadatas=[{"reponse": e["reponse"]} for e in FAQ_TEST],
    )
    return col


def test_recherche_retourne_resultats(collection_test, modele_embedding):
    resultats = rechercher_documents("retourner un article", collection_test, modele_embedding, top_k=3)
    assert len(resultats) > 0
    assert len(resultats) <= 3


def test_recherche_pertinence(collection_test, modele_embedding):
    resultats = rechercher_documents("retourner un article", collection_test, modele_embedding, top_k=3)
    ids = [r["id"] for r in resultats]
    assert "faq-01" in ids


def test_recherche_hors_domaine(collection_test, modele_embedding):
    # Seuil élevé (0.65) : une requête hors-domaine ne doit pas l'atteindre
    resultats = rechercher_documents(
        "recette de tarte aux pommes", collection_test, modele_embedding, top_k=3, seuil=0.65
    )
    assert len(resultats) == 0


def test_prompt_augmente_contient_contexte(collection_test, modele_embedding):
    from src.rag.pipeline import construire_prompt_augmente

    docs = rechercher_documents("retourner un article", collection_test, modele_embedding, top_k=3)
    prompt = construire_prompt_augmente("retourner un article", docs)
    assert "30 jours" in prompt or "retour" in prompt.lower()


def test_reranking_retourne_top_k(collection_test, modele_embedding, modele_crossencoder):
    candidats = rechercher_documents("retourner un article", collection_test, modele_embedding, top_k=5, seuil=0.0)
    reclasses = reclasser_passages("retourner un article", candidats, modele_crossencoder, top_k_final=2)
    assert len(reclasses) <= 2


def test_reranking_ordre_coherent(collection_test, modele_embedding, modele_crossencoder):
    candidats = rechercher_documents("délai de retour", collection_test, modele_embedding, top_k=5, seuil=0.0)
    reclasses = reclasser_passages("délai de retour", candidats, modele_crossencoder, top_k_final=3)
    if len(reclasses) >= 2:
        assert reclasses[0]["score_reranking"] >= reclasses[1]["score_reranking"]


def test_reranking_integre_pipeline(collection_test, modele_embedding, modele_crossencoder):
    from src.rag.pipeline import construire_prompt_augmente

    candidats = rechercher_documents("délai de retour", collection_test, modele_embedding, top_k=5, seuil=0.0)
    docs = reclasser_passages("délai de retour", candidats, modele_crossencoder, top_k_final=3)
    prompt = construire_prompt_augmente("délai de retour", docs)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
