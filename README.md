# RAG sur le programme de Physique-Chimie, Secondes littéraires (Togo)

Devoir RodiumAI Bootcamp, Justin Bina.

Ce dépôt contient un pipeline RAG (Retrieval-Augmented Generation) sans framework (ni LangChain ni LlamaIndex) qui répond à des questions sur le programme de physique-chimie de la classe de seconde littéraire (Togo), à partir du document officiel "Programme et guide 2nd A4 et S.pdf".

## Fichiers

- `rag_pipeline.py` : version script Python du pipeline, avec des cellules délimitées par `# %%`.
- `rag_physique_chimie_2nde.ipynb` : même pipeline sous forme de notebook (Colab).
- `README.md` : ce document.

Le PDF source n'est pas inclus dans ce dépôt. Il doit être fourni localement (voir Configuration).

## Fonctionnement du pipeline

1. Extraction du texte : les pages 5 à 28 du PDF (section 2nde A, littéraire) sont lues avec `pypdf`. Les espaces sont normalisés.
2. Découpage en chunks : chaque page est découpée en segments de 120 mots avec un recouvrement de 20 mots. Chaque chunk reçoit un identifiant de la forme `PXX-YY` (numéro de page, numéro du chunk dans la page).
3. Recherche dense : les chunks sont encodés avec le modèle `intfloat/multilingual-e5-small` (préfixe `passage: ` pour les documents, `query: ` pour les questions). Les vecteurs sont normalisés et la similarité cosinus est calculée par produit scalaire avec NumPy.
4. Recherche lexicale : index BM25 (`rank_bm25`) sur les tokens en minuscules, qui conserve les sigles comme PHY.2 ou CHI.3.
5. Fusion hybride : les classements dense et BM25 sont fusionnés par RRF (Reciprocal Rank Fusion, k = 60).
6. Génération : un client compatible OpenAI pointant vers l'API RodiumAI (température 0) reçoit les chunks récupérés sous la forme `[C1]`, `[C2]`, etc. Le prompt impose de citer chaque affirmation et de répondre exactement "INFORMATION ABSENTE" si la réponse n'est pas dans le contexte.
7. Évaluation : pour chaque méthode (dense, bm25, hybrid), le script calcule hit@1, hit@3, hit@5, la précision des citations et le taux de refus correct sur un jeu de 5 questions de programme et 1 question hors document.

## Configuration

Les paramètres se règlent par variables d'environnement :

- `RAG_PDF` : chemin du PDF (par défaut `/content/Programme et guide 2nd A4 et S.pdf`, pour Colab).
- `RODIUM_API_KEY` : clé API RodiumAI. Si elle n'est pas définie, le script se rabat sur `OPENAI_API_KEY`. Sans clé, la génération est désactivée et seuls les chunks récupérés sont affichés.
- `RODIUM_MODEL` : modèle de génération GPT sur RodiumAI. Si elle n'est pas définie, le script se rabat sur `LLM_MODEL`, puis sur `gpt-4o-mini`. Vous pouvez aussi utiliser `gpt-3.5-turbo` ou `gpt-4o`.

L'URL de base de l'API est fixée dans le code : `https://api.rodiumai.io/v1` (client `OpenAI(base_url=..., api_key=...)`, sans le chemin `/chat/completions`, ajouté automatiquement par le client).

### Configurer les identifiants RodiumAI

Dans un terminal (Linux ou macOS) :

```bash
export RODIUM_API_KEY="votre_clé_rodium"
export RODIUM_MODEL="gpt-4o-mini"   # optionnel : gpt-3.5-turbo ou gpt-4o
```

Dans Windows PowerShell :

```powershell
$env:RODIUM_API_KEY = "votre_clé_rodium"
$env:RODIUM_MODEL = "gpt-4o-mini"
```

Sur Colab, ajoutez la clé dans la cellule de configuration avant d'exécuter le pipeline :

```python
import os
os.environ["RODIUM_API_KEY"] = "votre_clé_rodium"
```

Ne committez jamais la clé dans le dépôt.

## Installation

```bash
pip install pypdf sentence-transformers rank_bm25 numpy openai
```

## Utilisation

Dans un terminal :

```bash
export RAG_PDF="chemin/vers/Programme et guide 2nd A4 et S.pdf"
export RODIUM_API_KEY="votre_clé_rodium"
python rag_pipeline.py
```

Sur Colab, téléverser le PDF dans `/content`, définir `RODIUM_API_KEY`, puis exécuter les cellules du notebook dans l'ordre.

La fonction `ask(retriever, client, query, method="hybrid")` permet de poser une question libre et affiche la réponse avec ses sources.

## Jeu d'évaluation

Le jeu de test associe chaque question à des pages de référence (`gold_pages`). Un chunk est compté comme pertinent s'il provient d'une de ces pages. La question hors programme (relativité générale) doit recevoir la réponse "INFORMATION ABSENTE".

Les résultats détaillés dépendent du PDF et du modèle utilisés ; le script les affiche dans un tableau à la fin de son exécution.
