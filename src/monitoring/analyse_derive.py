"""Détection de dérive avec Evidently AI.

Usage :
    python -m src.monitoring.analyse_derive --source fichier
    python -m src.monitoring.analyse_derive --source langfuse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

SEUIL_DERIVE = 0.3  # alerte si >30 % des colonnes dérivent


# ── Chargement des données ────────────────────────────────────────────────────

def _charger_jsonl(chemin: str | Path) -> pd.DataFrame:
    lignes = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return pd.DataFrame(lignes)


def _charger_depuis_langfuse(limit: int = 500) -> pd.DataFrame:
    from dotenv import load_dotenv
    from langfuse import Langfuse

    load_dotenv()
    lf = Langfuse()
    traces = lf.fetch_traces(limit=limit).data
    lignes = []
    for t in traces:
        scores = {s.name: s.value for s in (t.scores or [])}
        lignes.append({
            "prompt": t.input if isinstance(t.input, str) else str(t.input),
            "longueur_prompt": len(str(t.input).split()),
            "longueur_reponse": scores.get("longueur_reponse_mots", 0),
            "nb_docs_rag_trouves": scores.get("nb_docs_retrouves", 0),
            "hors_domaine": bool(scores.get("hors_domaine", 0)),
        })
    return pd.DataFrame(lignes)


# ── Rapport Evidently ─────────────────────────────────────────────────────────

def generer_rapport(df_ref: pd.DataFrame, df_prod: pd.DataFrame, dossier_sortie: Path) -> dict:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
    from evidently.report import Report

    dossier_sortie.mkdir(parents=True, exist_ok=True)

    mapping = ColumnMapping(
        numerical_features=["longueur_prompt", "longueur_reponse", "nb_docs_rag_trouves"],
        text_features=["prompt"],
    )

    rapport = Report(metrics=[DataDriftPreset(), TextOverviewPreset(column_name="prompt")])
    rapport.run(reference_data=df_ref, current_data=df_prod, column_mapping=mapping)

    rapport.save_html(str(dossier_sortie / "rapport_derive.html"))
    rapport.save_json(str(dossier_sortie / "rapport_derive.json"))
    print(f"Rapport sauvegardé dans {dossier_sortie}/")

    with open(dossier_sortie / "rapport_derive.json", encoding="utf-8") as f:
        return json.load(f)


# ── Détection automatique ─────────────────────────────────────────────────────

def detecter_alerte(rapport_json: dict, seuil: float = SEUIL_DERIVE) -> bool:
    for metrique in rapport_json.get("metrics", []):
        if "DatasetDriftMetric" in metrique.get("metric", ""):
            share = metrique.get("result", {}).get("share_of_drifted_columns", 0)
            if share > seuil:
                print(f"ALERTE DÉRIVE : {share:.0%} des colonnes ont dérivé (seuil={seuil:.0%})")
                return True
    return False


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fichier", "langfuse"], default="fichier")
    parser.add_argument("--production", default="logs/requetes.jsonl")
    parser.add_argument("--reference", default="data/corpus_reference.jsonl")
    parser.add_argument("--sortie", default="monitoring/rapports")
    args = parser.parse_args()

    df_ref = _charger_jsonl(args.reference)

    if args.source == "langfuse":
        print("Chargement des traces depuis Langfuse...")
        df_prod = _charger_depuis_langfuse()
    else:
        df_prod = _charger_jsonl(args.production)

    if df_prod.empty:
        print("Aucune donnée de production disponible.")
        sys.exit(0)

    rapport_json = generer_rapport(df_ref, df_prod, Path(args.sortie))

    if detecter_alerte(rapport_json):
        sys.exit(1)


if __name__ == "__main__":
    main()
