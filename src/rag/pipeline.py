from __future__ import annotations

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.rag.recherche import rechercher_documents
from src.rag.reranking import reclasser_passages

PROMPT_SYSTEME = """Tu es un assistant service client. Réponds uniquement en te basant sur le contexte fourni.
Si la réponse n'est pas dans le contexte, réponds : "Je n'ai pas l'information."

Contexte :
{contexte}

Question : {question}
Réponse :"""


def construire_prompt_augmente(question: str, documents: list[dict]) -> str:
    if not documents:
        contexte = "Aucun document pertinent trouvé."
    else:
        lignes = [f"[{i + 1}] {doc['reponse']}" for i, doc in enumerate(documents)]
        contexte = "\n".join(lignes)
    return PROMPT_SYSTEME.format(contexte=contexte, question=question)


def generer_avec_rag(
    prompt_utilisateur: str,
    collection: chromadb.Collection,
    modele_embedding: SentenceTransformer,
    modele_llm: PreTrainedModel,
    tokeniseur: PreTrainedTokenizer,
    modele_crossencoder: CrossEncoder | None = None,
    top_k_biencoder: int = 10,
    top_k_final: int = 3,
    nb_tokens_max: int = 200,
    trace=None,
) -> dict:
    """Pipeline RAG complet : retrieval → reranking → génération LLM."""

    # ── 1. Retrieval bi-encoder ──────────────────────────────────────────────
    span_retrieval = trace.span(name="retrieval", input=prompt_utilisateur) if trace else None

    candidats = rechercher_documents(
        requete=prompt_utilisateur,
        collection=collection,
        modele_embedding=modele_embedding,
        top_k=top_k_biencoder,
    )

    if span_retrieval:
        span_retrieval.end(
            output=[{"id": d["id"], "score": round(d["score"], 4)} for d in candidats],
            metadata={"nb_candidats": top_k_biencoder, "seuil": 0.4},
        )

    # ── 2. Reranking cross-encoder ───────────────────────────────────────────
    if modele_crossencoder and candidats:
        span_reranking = (
            trace.span(name="reranking", input=[d["id"] for d in candidats]) if trace else None
        )
        docs_selectionnes = reclasser_passages(
            requete=prompt_utilisateur,
            candidats=candidats,
            modele_crossencoder=modele_crossencoder,
            top_k_final=top_k_final,
        )
        if span_reranking:
            span_reranking.end(
                output=[
                    {"id": d["id"], "score_reranking": round(d.get("score_reranking", 0), 4)}
                    for d in docs_selectionnes
                ]
            )
    else:
        docs_selectionnes = candidats[:top_k_final]

    # ── 3. Construction du prompt augmenté ───────────────────────────────────
    prompt_augmente = construire_prompt_augmente(prompt_utilisateur, docs_selectionnes)

    # ── 4. Génération LLM ────────────────────────────────────────────────────
    inputs = tokeniseur(prompt_augmente, return_tensors="pt")
    nb_tokens_prompt = inputs["input_ids"].shape[1]

    span_generation = (
        trace.generation(
            name="llm",
            model="qwen2.5-1.5b-instruct",
            input=prompt_augmente,
        )
        if trace
        else None
    )

    outputs = modele_llm.generate(
        **inputs,
        max_new_tokens=nb_tokens_max,
        do_sample=False,
        pad_token_id=tokeniseur.eos_token_id,
    )

    tokens_generes = outputs[0][nb_tokens_prompt:]
    reponse = tokeniseur.decode(tokens_generes, skip_special_tokens=True).strip()
    nb_tokens_out = len(tokens_generes)

    if span_generation:
        span_generation.end(
            output=reponse,
            usage={
                "input": nb_tokens_prompt,
                "output": nb_tokens_out,
                "total": nb_tokens_prompt + nb_tokens_out,
            },
        )

    score_max = max(
        (d.get("score_reranking", d["score"]) for d in docs_selectionnes),
        default=0.0,
    )

    return {
        "reponse": reponse,
        "prompt_augmente": prompt_augmente,
        "documents_sources": [d["id"] for d in docs_selectionnes],
        "nb_docs": len(docs_selectionnes),
        "hors_domaine": len(docs_selectionnes) == 0,
        "score_reranking_max": score_max,
        "nb_tokens_out": nb_tokens_out,
    }
