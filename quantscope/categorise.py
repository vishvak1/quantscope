# quantscope/categorise.py
# Analysis 2: Disagreement Categorisation
# Generic — no hardcoded domain knowledge.
# Uses TF-IDF vocabulary rarity, spaCy NER, and text complexity metrics.

import re
import numpy as np
import spacy
from collections import Counter
from scipy import stats

# Load spaCy model once at import time
nlp = spacy.load("en_core_web_sm")


# ── Vocabulary rarity ─────────────────────────────────────────────────────
def build_training_vocab(texts: list, min_count: int = 5) -> set:
    """
    Build vocabulary from a list of texts.
    Only keeps tokens seen at least min_count times.
    Tokens not in this vocab are considered 'rare'.
    """
    counter = Counter()
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        counter.update(tokens)
    return {token for token, count in counter.items() if count >= min_count}


def vocab_rarity(text: str, training_vocab: set) -> float:
    """% of tokens in text that are rare (not in training vocab)."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return 0.0
    rare = sum(1 for t in tokens if t not in training_vocab)
    return rare / len(tokens)


# ── Text complexity ───────────────────────────────────────────────────────
NEGATION_WORDS = {
    "no", "not", "without", "denied", "denies", "negative", "absence",
    "absent", "neither", "nor", "never", "none", "unlikely",
}

def text_complexity(text: str) -> dict:
    """
    Generic text complexity features that work for any domain.
    """
    tokens     = re.findall(r'\b\w+\b', text.lower())
    char_len   = len(text)
    n_tokens   = len(tokens)
    n_sentences = len(re.split(r'[.!?]+', text.strip()))

    # Length bucket
    if char_len < 50:
        length_bucket = "short"
    elif char_len <= 150:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    # Avg word length — proxy for technical vocabulary
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0

    # Negation presence
    has_negation = any(neg in tokens for neg in NEGATION_WORDS)

    # Clause count proxy — number of commas + conjunctions
    clause_count = text.count(',') + len(re.findall(r'\b(and|but|or|because|although|however)\b', text.lower()))

    return {
        "char_len"      : char_len,
        "length_bucket" : length_bucket,
        "n_tokens"      : n_tokens,
        "avg_word_len"  : round(float(avg_word_len), 3),
        "has_negation"  : has_negation,
        "clause_count"  : clause_count,
        "n_sentences"   : n_sentences,
    }


# ── Named entity analysis ─────────────────────────────────────────────────
def entity_features(text: str) -> dict:
    """
    Uses spaCy NER to detect named entities generically.
    Works for any domain — medical, legal, financial, etc.
    """
    doc        = nlp(text)
    entities   = [(ent.text, ent.label_) for ent in doc.ents]
    entity_types = [label for _, label in entities]

    return {
        "n_entities"    : len(entities),
        "entity_types"  : entity_types,
        "has_entities"  : len(entities) > 0,
        "unique_types"  : list(set(entity_types)),
    }


# ── Full feature extraction ───────────────────────────────────────────────
def categorise_text(text: str, training_vocab: set = None) -> dict:
    """Extract all features from a single text."""
    features = text_complexity(text)
    features.update(entity_features(text))

    if training_vocab is not None:
        features["rarity_pct"] = round(vocab_rarity(text, training_vocab), 4)
    else:
        features["rarity_pct"] = None

    return features


def categorise_all(texts: list, training_vocab: set = None) -> list:
    """Extract features from all texts."""
    print(f"Categorising {len(texts)} texts...")
    features = []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}")
        features.append(categorise_text(text, training_vocab))
    return features


# ── Statistical comparison ────────────────────────────────────────────────
def compare_categories(features_all: list, disagree_indices: list) -> dict:
    """
    Compare feature distributions between disagreement and agreement inputs.
    Fully generic — no domain knowledge required.
    """
    n_total      = len(features_all)
    disagree_set = set(disagree_indices)
    agree_idx    = [i for i in range(n_total) if i not in disagree_set]

    def d_feat(key):
        return [features_all[i][key] for i in disagree_indices]

    def a_feat(key):
        return [features_all[i][key] for i in agree_idx]

    results = {}

    # Length bucket
    for bucket in ["short", "medium", "long"]:
        d_pct = np.mean([f["length_bucket"] == bucket for f in [features_all[i] for i in disagree_indices]])
        a_pct = np.mean([f["length_bucket"] == bucket for f in [features_all[i] for i in agree_idx]])
        results[f"length_{bucket}"] = {
            "disagree_pct": round(float(d_pct * 100), 1),
            "agree_pct"   : round(float(a_pct * 100), 1),
            "ratio"       : round(float(d_pct / max(a_pct, 0.001)), 2),
        }

    # Vocabulary rarity — KS test
    d_rarity = [f["rarity_pct"] for f in [features_all[i] for i in disagree_indices] if f["rarity_pct"] is not None]
    a_rarity = [f["rarity_pct"] for f in [features_all[i] for i in agree_idx] if f["rarity_pct"] is not None]
    if d_rarity and a_rarity:
        ks_stat, ks_p = stats.ks_2samp(d_rarity, a_rarity)
        results["vocab_rarity"] = {
            "disagree_mean": round(float(np.mean(d_rarity)), 4),
            "agree_mean"   : round(float(np.mean(a_rarity)), 4),
            "ks_stat"      : round(float(ks_stat), 4),
            "ks_p"         : round(float(ks_p), 4),
        }

    # Avg word length — KS test
    d_wlen = d_feat("avg_word_len")
    a_wlen = a_feat("avg_word_len")
    ks_stat, ks_p = stats.ks_2samp(d_wlen, a_wlen)
    results["avg_word_length"] = {
        "disagree_mean": round(float(np.mean(d_wlen)), 4),
        "agree_mean"   : round(float(np.mean(a_wlen)), 4),
        "ks_stat"      : round(float(ks_stat), 4),
        "ks_p"         : round(float(ks_p), 4),
    }

    # Negation
    d_neg = np.mean([features_all[i]["has_negation"] for i in disagree_indices])
    a_neg = np.mean([features_all[i]["has_negation"] for i in agree_idx])
    results["negation"] = {
        "disagree_pct": round(float(d_neg * 100), 1),
        "agree_pct"   : round(float(a_neg * 100), 1),
        "ratio"       : round(float(d_neg / max(a_neg, 0.001)), 2),
    }

    # Entity presence
    d_ent = np.mean([features_all[i]["has_entities"] for i in disagree_indices])
    a_ent = np.mean([features_all[i]["has_entities"] for i in agree_idx])
    results["has_entities"] = {
        "disagree_pct": round(float(d_ent * 100), 1),
        "agree_pct"   : round(float(a_ent * 100), 1),
        "ratio"       : round(float(d_ent / max(a_ent, 0.001)), 2),
    }

    # Entity count — KS test
    d_nent = d_feat("n_entities")
    a_nent = a_feat("n_entities")
    ks_stat, ks_p = stats.ks_2samp(d_nent, a_nent)
    results["entity_count"] = {
        "disagree_mean": round(float(np.mean(d_nent)), 4),
        "agree_mean"   : round(float(np.mean(a_nent)), 4),
        "ks_stat"      : round(float(ks_stat), 4),
        "ks_p"         : round(float(ks_p), 4),
    }

    return results


def print_category_report(comparison: dict):
    print("Disagreement Categorisation")
    print("=" * 60)
    print(f"\n{'Feature':<25} {'Disagree':>10} {'Agree':>10} {'Ratio/p':>10}")
    print("-" * 60)
    for key, val in comparison.items():
        if "disagree_pct" in val:
            ratio = val.get("ratio", "-")
            print(
                f"{key:<25} {str(val['disagree_pct'])+'%':>10} "
                f"{str(val['agree_pct'])+'%':>10} {ratio:>10}"
            )
        else:
            print(
                f"{key:<25} {'m='+str(val['disagree_mean']):>10} "
                f"{'m='+str(val['agree_mean']):>10} "
                f"{'p='+str(val['ks_p']):>10}"
            )