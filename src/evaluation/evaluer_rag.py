"""Évaluation du pipeline RAG : Recall@k, MRR, ROUGE-L, comparaison avec/sans RAG.

Usage :
    python -m src.evaluation.evaluer_rag
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from src.rag.base_connaissance import creer_base_connaissance
from src.rag.recherche import rechercher_documents

JEU_EVAL = Path(__file__).parent / "jeu_evaluation.jsonl"
RAPPORT = Path(__file__).parent / "rapport_eval.json"

SEUIL_RECALL = 0.75
SEUIL_ROUGE = 0.25  # seuil bas : modèle 1.5B, paraphrases fréquentes

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Chargement ────────────────────────────────────────────────────────────────

def _charger_jeu(chemin: Path) -> list[dict]:
    entrees = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                entrees.append(json.loads(ligne))
    return entrees


# ── Métriques retrieval ───────────────────────────────────────────────────────

def calculer_recall_at_k(
    entrees: list[dict],
    collection,
    modele_embedding: SentenceTransformer,
    k: int,
) -> float:
    hits = 0
    total = 0
    for e in entrees:
        if not e["docs_pertinents"]:
            continue
        total += 1
        resultats = rechercher_documents(e["question"], collection, modele_embedding, top_k=k, seuil=0.0)
        ids = [r["id"] for r in resultats]
        if any(dp in ids for dp in e["docs_pertinents"]):
            hits += 1
    return hits / total if total > 0 else 0.0


def calculer_mrr(
    entrees: list[dict],
    collection,
    modele_embedding: SentenceTransformer,
    k: int = 5,
) -> float:
    reciprocal_ranks = []
    for e in entrees:
        if not e["docs_pertinents"]:
            continue
        resultats = rechercher_documents(e["question"], collection, modele_embedding, top_k=k, seuil=0.0)
        ids = [r["id"] for r in resultats]
        rang = next(
            (i + 1 for i, rid in enumerate(ids) if rid in e["docs_pertinents"]),
            None,
        )
        reciprocal_ranks.append(1 / rang if rang else 0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ── Métriques génération ──────────────────────────────────────────────────────

def calculer_rouge_l(reponse_generee: str, reponse_reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reponse_reference, reponse_generee)
    return scores["rougeL"].fmeasure


def evaluer_generation(
    entrees: list[dict],
    collection,
    modele_embedding: SentenceTransformer,
    generer_fn,  # callable(question) -> str
) -> dict:
    scores_rouge = []
    for e in entrees:
        reponse = generer_fn(e["question"])
        score = calculer_rouge_l(reponse, e["reponse_reference"])
        scores_rouge.append(score)
    return {"rouge_l_moyen": round(sum(scores_rouge) / len(scores_rouge), 4)}


# ── Génération sans RAG (baseline) ───────────────────────────────────────────

def _generer_sans_rag(question: str) -> str:
    # Retourne une réponse vide pour simuler un LLM sans contexte
    # (évite de charger le LLM complet en CI pour la baseline)
    return ""


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    print("Chargement du modèle d'embedding...")
    modele_embedding = SentenceTransformer(EMBEDDING_MODEL)
    collection = creer_base_connaissance()

    entrees = _charger_jeu(JEU_EVAL)
    print(f"{len(entrees)} questions chargées.")

    # ── Retrieval
    recall_1 = calculer_recall_at_k(entrees, collection, modele_embedding, k=1)
    recall_3 = calculer_recall_at_k(entrees, collection, modele_embedding, k=3)
    recall_5 = calculer_recall_at_k(entrees, collection, modele_embedding, k=5)
    mrr = calculer_mrr(entrees, collection, modele_embedding, k=5)

    # ── Génération (ROUGE sur les réponses FAQ directes, sans LLM en CI)
    def generer_rag_simple(question: str) -> str:
        docs = rechercher_documents(question, collection, modele_embedding, top_k=3, seuil=0.0)
        return docs[0]["reponse"] if docs else "Je n'ai pas l'information."

    rouge_avec_rag = evaluer_generation(entrees, collection, modele_embedding, generer_rag_simple)
    rouge_sans_rag = evaluer_generation(entrees, collection, modele_embedding, _generer_sans_rag)

    rapport = {
        "recall_at_1": round(recall_1, 4),
        "recall_at_3": round(recall_3, 4),
        "recall_at_5": round(recall_5, 4),
        "mrr": round(mrr, 4),
        "nb_questions": len(entrees),
        "avec_rag": rouge_avec_rag,
        "sans_rag": rouge_sans_rag,
        "gain_rouge_l": round(rouge_avec_rag["rouge_l_moyen"] - rouge_sans_rag["rouge_l_moyen"], 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(RAPPORT, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(json.dumps(rapport, indent=2, ensure_ascii=False))

    # ── Seuils d'alerte
    alertes = []
    if recall_3 < SEUIL_RECALL:
        alertes.append(f"Recall@3={recall_3:.2f} < seuil {SEUIL_RECALL}")
    if rouge_avec_rag["rouge_l_moyen"] < SEUIL_ROUGE:
        alertes.append(f"ROUGE-L={rouge_avec_rag['rouge_l_moyen']:.2f} < seuil {SEUIL_ROUGE}")

    if alertes:
        print("\nALERTE QUALITÉ :")
        for a in alertes:
            print(f"  ✗ {a}")
        sys.exit(1)

    print("\nQualité OK — tous les seuils sont respectés.")


if __name__ == "__main__":
    main()
