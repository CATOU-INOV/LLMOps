# LLMOps RAG — Assistant Conversationnel

Pipeline RAG complet avec observabilité, dérive et CI/CD. Projet LLMOps Ynov.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-FF6B35?logo=databricks&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-Observabilité-7C3AED?logo=opsgenie&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linter-D7FF64?logo=ruff&logoColor=black)

---

## Architecture

```
Requête utilisateur
       │
       ▼
  FastAPI /generate
       │
   ┌───┴────────────────────────────┐
   │  Pipeline RAG                  │
   │  1. Bi-encoder (ChromaDB)      │
   │  2. Cross-encoder reranking    │
   │  3. LLM Qwen2.5-1.5B          │
   └───┬────────────────────────────┘
       │
   ┌───┴──────────────────┐
   │  Observabilité        │
   │  Langfuse (traces)    │
   │  Prometheus (métriques│
   │  Evidently (dérive)   │
   └──────────────────────┘
```

## Démarrage rapide

### 1. Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Remplir les clés Langfuse après avoir lancé docker compose
```

### 3. Stack monitoring (Langfuse + Prometheus + Grafana)

```bash
cd monitoring
docker compose up -d
```

- Langfuse : http://localhost:3000 (créer un compte local + projet)
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3001 (admin / admin)

### 4. Lancer l'API

```bash
PYTHONPATH=. uvicorn src.api.app:app --reload
```

- API : http://localhost:8000
- Docs : http://localhost:8000/docs
- Métriques : http://localhost:8000/metrics

### 5. Exemple de requête

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Sous quel délai puis-je retourner un article ?"}'
```

## Structure du projet

```
├── src/
│   ├── rag/
│   │   ├── base_connaissance.py   # ChromaDB, indexation FAQ
│   │   ├── recherche.py           # Bi-encoder, seuil similarité
│   │   ├── chunking.py            # Découpage documents longs
│   │   ├── reranking.py           # Cross-encoder reranking
│   │   └── pipeline.py            # Pipeline complet + traces
│   ├── api/
│   │   └── app.py                 # FastAPI, Prometheus, Langfuse
│   ├── monitoring/
│   │   └── analyse_derive.py      # Evidently AI drift detection
│   └── evaluation/
│       ├── evaluer_rag.py         # Recall@k, MRR, ROUGE-L
│       └── jeu_evaluation.jsonl   # 17 questions de test
├── monitoring/
│   ├── docker-compose.yml         # Langfuse + Prometheus + Grafana
│   ├── prometheus.yml
│   └── alertes.yml
├── data/
│   ├── faq_service_client.jsonl   # 8 entrées FAQ
│   ├── documents_longs.jsonl      # Documents pour chunking
│   └── corpus_reference.jsonl     # Référence dérive Evidently
├── tests/                         # 19 tests unitaires
└── .github/workflows/
    ├── codecheck.yaml             # pre-commit (ruff, mypy)
    ├── tests.yaml                 # matrice ubuntu×windows × 3.10-3.12
    └── rag_eval.yaml              # évaluation hebdomadaire (lundi 2h UTC)
```

## Métriques RAG

| Métrique | Valeur |
|---|---|
| Recall@1 | 0.87 |
| Recall@3 | **1.00** |
| MRR | 0.93 |
| ROUGE-L avec RAG | 0.57 |
| ROUGE-L sans RAG | 0.00 |

## Lancer les tests

```bash
PYTHONPATH=. pytest tests/ -v
```

## Évaluation RAG

```bash
PYTHONPATH=. python -m src.evaluation.evaluer_rag
```

## Analyse de dérive

```bash
# Sur les logs locaux
PYTHONPATH=. python -m src.monitoring.analyse_derive --source fichier

# Sur les traces Langfuse
PYTHONPATH=. python -m src.monitoring.analyse_derive --source langfuse
```

## CI/CD

| Workflow | Déclencheur |
|---|---|
| `codecheck` | Chaque push/PR |
| `tests` | Chaque push/PR (ubuntu + windows × py3.10/3.11/3.12) |
| `rag_eval` | Chaque lundi à 2h UTC + manuel |

## Modèles utilisés

| Rôle | Modèle |
|---|---|
| Embedding | `all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` |
