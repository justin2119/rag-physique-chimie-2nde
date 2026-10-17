# %% [markdown]
# # RAG sur le programme de Physique-Chimie, Secondes littéraires (Togo)
# Devoir RodiumAI Bootcamp, Justin Bina.
#
# Pipeline RAG sans framework (ni LangChain ni LlamaIndex) :
# - extraction du texte avec `pypdf` (pages 5 à 28 du document "Programme et guide 2nd A4 et S.pdf", section 2nde A / littéraire) ;
# - découpage en chunks identifiés (format PXX-YY) ;
# - recherche dense avec `intfloat/multilingual-e5-small` (préfixes `passage: ` et `query: `), similarité cosinus par produit scalaire NumPy sur vecteurs normalisés ;
# - recherche lexicale avec `rank_bm25` ;
# - fusion hybride par RRF (Reciprocal Rank Fusion, k=60) ;
# - génération avec un client OpenAI compatible, prompt imposant les citations [Cn] et le refus strict "INFORMATION ABSENTE" ;
# - évaluation : hit@1, hit@3, hit@5 et précision des citations sur 5 questions de programme et 1 question hors document.

# %%
# Installation (Colab). La ligne suivante est une commande IPython dans le notebook et un commentaire dans un script Python classique.
# %pip install -q pypdf sentence-transformers rank_bm25 numpy openai

# %%
import os
import re
import numpy as np
import openai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI

# %%
# Configuration. Dans Colab : téléverser le PDF dans /content et définir RODIUM_API_KEY (voir le README).
# La clé API et le modèle ne sont pas figés ici : ils sont lus au moment de l'appel (voir make_llm_client et get_llm_model).
PDF_PATH = os.environ.get("RAG_PDF", "/content/Programme et guide 2nd A4 et S.pdf")
PAGE_START, PAGE_END = 5, 28
CHUNK_WORDS, CHUNK_OVERLAP = 120, 20
RRF_K = 60
TOP_K = 5
EMB_MODEL_NAME = "intfloat/multilingual-e5-small"
RODIUM_BASE_URL = "https://api.rodiumai.io/v1"  # URL de base RodiumAI (compatible OpenAI)

# %% [markdown]
# ## 1. Extraction et découpage du PDF

# %%
def load_pages(pdf_path, start=PAGE_START, end=PAGE_END):
    """Retourne un dictionnaire {numéro de page: texte nettoyé} pour les pages demandées (numérotation à partir de 1)."""
    reader = PdfReader(pdf_path)
    pages = {}
    for num in range(start, end + 1):
        raw = reader.pages[num - 1].extract_text() or ""
        pages[num] = re.sub(r"\s+", " ", raw).strip()
    return pages


def chunk_pages(pages, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Découpe chaque page en chunks de `size` mots avec recouvrement. Identifiant : PXX-YY."""
    chunks = []
    step = size - overlap
    for page, text in pages.items():
        words = text.split()
        if not words:
            continue
        for i, start in enumerate(range(0, len(words), step), start=1):
            piece = " ".join(words[start:start + size])
            chunks.append({"id": "P{:02d}-{:02d}".format(page, i), "page": page, "text": piece})
            if start + size >= len(words):
                break
    return chunks

# %% [markdown]
# ## 2. Index dense, index BM25 et fusion RRF

# %%
def tokenize(text):
    """Tokenisation simple et insensible à la casse, adaptée au français et aux sigles (PHY.2, CHI.3)."""
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rankings, k=RRF_K):
    """Reciprocal Rank Fusion : score = somme de 1 / (k + rang) sur chaque classement."""
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(fused.items(), key=lambda item: -item[1])]


class Retriever:
    def __init__(self, chunks, model_name=EMB_MODEL_NAME):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        # Vecteurs normalisés : le produit scalaire équivaut au cosinus.
        self.doc_vecs = self.model.encode(
            ["passage: " + c["text"] for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def dense_order(self, query):
        q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        sims = self.doc_vecs @ q
        return list(np.argsort(-sims))

    def bm25_order(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores))

    def search(self, query, method="hybrid", k=TOP_K):
        if method == "dense":
            order = self.dense_order(query)
        elif method == "bm25":
            order = self.bm25_order(query)
        else:
            order = rrf_fuse([self.dense_order(query), self.bm25_order(query)])
        return [self.chunks[int(i)] for i in order[:k]]

# %% [markdown]
# ## 3. Génération avec citations et refus strict

# %%
SYSTEM_PROMPT = """Tu es un assistant pédagogique qui répond uniquement à partir du CONTEXTE fourni.
Règles strictes :
1. Chaque affirmation doit être suivie d'une citation au format [C1], [C2], etc., qui renvoie au chunk du contexte.
2. N'utilise aucune connaissance extérieure au contexte.
3. Si la réponse n'est pas explicitement présente dans le contexte, réponds exactement : INFORMATION ABSENTE.
4. Réponds en français."""


def build_context(hits):
    return "\n\n".join(
        "[C{}] (page {}, {}) {}".format(i, h["page"], h["id"], h["text"])
        for i, h in enumerate(hits, start=1)
    )


def get_llm_model():
    """Modèle lu à l'appel : RODIUM_MODEL, sinon LLM_MODEL, sinon gpt-4o-mini."""
    return os.environ.get("RODIUM_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"


def make_llm_client():
    """Retourne un client OpenAI compatible RodiumAI.

    La clé est lue à chaque appel : RODIUM_API_KEY en priorité, sinon OPENAI_API_KEY.
    Retourne None et affiche un message clair si aucune clé n'est définie.
    """
    api_key = os.environ.get("RODIUM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[RODIUM] Aucune clé API trouvée. Définissez RODIUM_API_KEY (ou OPENAI_API_KEY). Génération désactivée.")
        return None
    return OpenAI(api_key=api_key, base_url=RODIUM_BASE_URL)


def generate_answer(client, query, hits):
    """Appelle le LLM. Retourne le texte généré, ou None si l'appel échoue (le message d'erreur est affiché)."""
    if client is None:
        print("[RODIUM] Pas de client : génération ignorée.")
        return None
    model = get_llm_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "CONTEXTE :\n" + build_context(hits) + "\n\nQUESTION : " + query},
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    except openai.AuthenticationError as e:
        print("[RODIUM] Échec d'authentification : clé API invalide ou expirée. Vérifiez RODIUM_API_KEY. Détail : {}".format(e))
        return None
    except openai.NotFoundError as e:
        print("[RODIUM] Modèle introuvable : '{}' n'existe pas sur RodiumAI. Vérifiez RODIUM_MODEL. Détail : {}".format(model, e))
        return None
    except openai.BadRequestError as e:
        print("[RODIUM] Requête refusée (nom de modèle invalide ou paramètre incorrect ?) avec le modèle '{}'. Détail : {}".format(model, e))
        return None
    except openai.RateLimitError as e:
        print("[RODIUM] Limite de débit atteinte ou quota épuisé. Réessayez plus tard. Détail : {}".format(e))
        return None
    except openai.APIConnectionError as e:
        print("[RODIUM] Connexion impossible à https://api.rodiumai.io/v1. Vérifiez le réseau. Détail : {}".format(e))
        return None
    except openai.APIStatusError as e:
        print("[RODIUM] Erreur HTTP {} renvoyée par l'API. Détail : {}".format(e.status_code, e))
        return None
    except Exception as e:  # filet de sécurité : ne pas faire planter le pipeline
        print("[RODIUM] Erreur inattendue lors de l'appel à l'API : {}: {}".format(type(e).__name__, e))
        return None
    content = resp.choices[0].message.content if resp.choices else None
    if not content:
        print("[RODIUM] Réponse vide reçue du modèle '{}'.".format(model))
        return None
    return content.strip()


def ask(retriever, client, query, method="hybrid", k=TOP_K):
    """Pose une question, affiche la réponse et les sources utilisées."""
    hits = retriever.search(query, method=method, k=k)
    if client is None:
        print("Pas de clé API : seuls les chunks récupérés sont affichés.")
        answer = None
    else:
        answer = generate_answer(client, query, hits)
        print(answer if answer is not None else "Aucune réponse générée (voir le message d'erreur ci-dessus).")
    print("\nSources :")
    for i, h in enumerate(hits, start=1):
        print("  [C{}] page {} ({})".format(i, h["page"], h["id"]))
    return answer, hits

# %% [markdown]
# ## 4. Jeu d'évaluation
#
# Vérité terrain : pages attendues (`gold_pages`). Un chunk est un hit s'il provient d'une page attendue.
# La question hors document doit recevoir la réponse "INFORMATION ABSENTE".

# %%
EVAL_SET = [
    {"q": "Quels sont les thèmes du programme de physique-chimie de la seconde littéraire ?",
     "gold_pages": [7], "refusal": False},
    {"q": "Qu'est-ce que la leçon PHY.2 sur la tension électrique ?",
     "gold_pages": [8, 14], "refusal": False},
    {"q": "Que contient la leçon CHI.3 sur la mole et les grandeurs molaires ?",
     "gold_pages": [11, 23, 24], "refusal": False},
    {"q": "Qu'est-ce qu'une équation bilan d'une réaction chimique ?",
     "gold_pages": [11, 25], "refusal": False},
    {"q": "Que traite la leçon PHY.6 sur la force en mécanique ?",
     "gold_pages": [10, 20], "refusal": False},
    {"q": "Quelle est l'équation de la relativité générale d'Einstein ?",
     "gold_pages": [], "refusal": True},
]

# %%
def hit_at_k(hits, gold_pages):
    return int(any(h["page"] in gold_pages for h in hits))


def citation_precision(text, hits, gold_pages):
    """Part des citations [Cn] qui pointent vers une page attendue. None si aucune citation."""
    cited = {int(n) for n in re.findall(r"\[C(\d+)\]", text)}
    if not cited:
        return None
    correct = sum(1 for n in cited if 1 <= n <= len(hits) and hits[n - 1]["page"] in gold_pages)
    return correct / len(cited)


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def run_evaluation(retriever, client=None):
    report = {}
    failed_calls = 0
    for method in ["dense", "bm25", "hybrid"]:
        rows = {"hit@1": [], "hit@3": [], "hit@5": [], "cite_prec": [], "refusal_ok": []}
        for ex in EVAL_SET:
            hits = retriever.search(ex["q"], method=method, k=TOP_K)
            text = generate_answer(client, ex["q"], hits) if client is not None else None
            if client is not None and text is None:
                failed_calls += 1
            if ex["refusal"]:
                if text is not None:
                    rows["refusal_ok"].append("INFORMATION ABSENTE" in text)
                continue
            rows["hit@1"].append(hit_at_k(hits[:1], ex["gold_pages"]))
            rows["hit@3"].append(hit_at_k(hits[:3], ex["gold_pages"]))
            rows["hit@5"].append(hit_at_k(hits[:5], ex["gold_pages"]))
            if text is not None:
                cp = citation_precision(text, hits, ex["gold_pages"])
                if cp is not None:
                    rows["cite_prec"].append(cp)
        report[method] = rows

    print("{:<8} {:>6} {:>6} {:>6} {:>12} {:>10}".format("methode", "hit@1", "hit@3", "hit@5", "cite_prec", "refus"))
    for method, rows in report.items():
        refus = "n/a" if not rows["refusal_ok"] else str(all(rows["refusal_ok"]))
        print("{:<8} {:>6.2f} {:>6.2f} {:>6.2f} {:>12.2f} {:>10}".format(
            method, mean(rows["hit@1"]), mean(rows["hit@3"]), mean(rows["hit@5"]),
            mean(rows["cite_prec"]), refus))
    if client is None:
        print("Sans clé API : la précision des citations et le refus ne sont pas mesurés.")
    if failed_calls:
        print("[RODIUM] {} appel(s) au LLM ont échoué : précision des citations et refus incomplets (voir les messages ci-dessus).".format(failed_calls))
    return report

# %% [markdown]
# ## 5. Exécution

# %%
def main():
    pages = load_pages(PDF_PATH)
    chunks = chunk_pages(pages)
    print("Pages : {} | chunks : {}".format(len(pages), len(chunks)))
    retriever = Retriever(chunks)
    client = make_llm_client()
    return run_evaluation(retriever, client)


if __name__ == "__main__":
    main()
