import pytest
import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.chunking import decouper_document, indexer_documents_longs


def test_decoupe_sans_chevauchement():
    texte = " ".join([f"mot{i}" for i in range(100)])
    chunks = decouper_document(texte, taille_chunk=30, chevauchement=0)
    # 100 / 30 = 3 pleins + 1 reste = 4 chunks
    assert len(chunks) == 4
    assert all(isinstance(c, str) for c in chunks)


def test_decoupe_avec_chevauchement():
    mots = [f"mot{i}" for i in range(20)]
    texte = " ".join(mots)
    chunks = decouper_document(texte, taille_chunk=10, chevauchement=3)
    # Les 3 derniers mots du chunk 1 doivent apparaître au début du chunk 2
    mots_fin_chunk1 = chunks[0].split()[-3:]
    mots_debut_chunk2 = chunks[1].split()[:3]
    assert mots_fin_chunk1 == mots_debut_chunk2


def test_chunk_vide():
    assert decouper_document("") == []
    assert decouper_document("   ") == []


def test_texte_plus_court_que_chunk():
    texte = " ".join([f"mot{i}" for i in range(50)])
    chunks = decouper_document(texte, taille_chunk=200, chevauchement=40)
    assert len(chunks) == 1
    assert chunks[0] == texte


def test_metadonnees_source():
    import tempfile, json, pathlib

    modele = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    col = client.create_collection("test_chunks", metadata={"hnsw:space": "cosine"})

    doc = {"id": "doc-test", "texte": " ".join([f"mot{i}" for i in range(50)])}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(doc) + "\n")
        chemin = f.name

    indexer_documents_longs(chemin, col, modele, taille_chunk=20, chevauchement=5)

    resultats = col.get(include=["metadatas"])
    for meta in resultats["metadatas"]:
        assert meta["source_id"] == "doc-test"
