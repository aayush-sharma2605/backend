import re
import numpy as np
from typing import List, Dict, Any, Tuple

# Advanced Keyword Lexicons
JOB_KEYWORDS = ["architect", "engineer", "lead", "senior", "director", "manager", "scaling", "solutions", "stack", "agile", "automation", "consulting"]
POSITIVE_ADJECTIVES = ["excellent", "impressive", "brilliant", "great", "innovative", "unique", "remarkable", "outstanding", "value"]
URGENCY_KEYWORDS = ["urgent", "immediately", "asap", "invoice", "overdue", "prize", "winner", "cash", "account", "unlocked"]
PERSONAL_PRONOUNS = ["you", "your", "yours", "we", "us", "our"]

def get_syllable_count(word: str) -> int:
    """Estimates syllables in a word."""
    word = word.lower()
    if not word: return 0
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels: count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'): count -= 1
    if count == 0: count = 1
    return count

def extract_features(text: str) -> np.ndarray:
    """
    Enhanced feature extraction for v3.14 model.
    Captures sophisticated linguistic signals.
    """
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    # 1. Basic Readability
    avg_sentence_len = word_count / sentence_count
    
    # 2. Syllable count for Flesch Score
    syllable_count = sum(get_syllable_count(w) for w in words)
    # Flesch formula: 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
    flesch_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * (syllable_count / max(word_count, 1)))
    
    # 3. Personalization Depth (Entities)
    # Count occurrences of Name/Company patterns
    name_patterns = [r"hi\s+[a-z][a-z]+", r"dear\s+[a-z][a-z]+", r"hello\s+[a-z][a-z]+"]
    has_name = 1 if any(re.search(p, text_lower) for p in name_patterns) else 0
    
    # 4. Keyword Densities
    job_match_count = sum(1 for kw in JOB_KEYWORDS if kw in text_lower)
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    positivity_count = sum(1 for kw in POSITIVE_ADJECTIVES if kw in text_lower)
    
    # 5. Engagement (Personal vs Generic)
    personal_count = sum(1 for p in PERSONAL_PRONOUNS if p in words)
    engagement_density = personal_count / max(word_count, 1)

    return np.array([
        word_count,
        sentence_count,
        avg_sentence_len,
        flesch_score,
        has_name,
        job_match_count,
        urgency_count,
        positivity_count,
        engagement_density
    ])

def compute_breakdown(text: str) -> Dict[str, int]:
    """
    Explainable scores for the dashboard.
    """
    features = extract_features(text)
    
    # Personalization [0-100]
    # Weighted by name presence and engagement density
    pers_val = (features[4] * 60) + (min(features[8] * 200, 40))
    
    # Clarity (Flesch based) [0-100]
    # Optimal Flesch is 60-70 for standard outreach
    flesch = features[3]
    if 60 <= flesch <= 80:
        clarity_val = 100
    else:
        clarity_val = max(0, 100 - abs(70 - flesch))
        
    # Tone [0-100] (Positivity - Urgency)
    tone_val = 50 + (features[7] * 15) - (features[6] * 25)
    
    # Relevance [0-100]
    rel_val = min(features[5] * 20, 100)
    
    # Length efficiency [0-100] (Optimal 60-120 words)
    wc = features[0]
    if 60 <= wc <= 120:
        len_val = 100
    elif wc < 60:
        len_val = wc * 1.5
    else:
        len_val = max(0, 100 - (wc - 120) / 2)
        
    return {
        "personalization": int(min(pers_val, 100)),
        "clarity": int(min(clarity_val, 100)),
        "relevance": int(min(rel_val, 100)),
        "tone": int(max(0, min(tone_val, 100))),
        "length": int(min(len_val, 100))
    }

def get_recommendations(breakdown: Dict[str, int]) -> List[str]:
    recs = []
    if breakdown['personalization'] < 50:
        recs.append("Add more personal references like the recipient's name or specific role mentions.")
    if breakdown['clarity'] < 60:
        recs.append("Shorten complex sentences to improve the reading ease score.")
    if breakdown['tone'] < 40:
        recs.append("Reduce urgent or 'spammy' language; try to sound more consultative.")
    if breakdown['relevance'] < 50:
        recs.append("Include more industry-specific keywords to show target alignment.")
    if breakdown['length'] < 70:
        recs.append("Your email is quite short. Consider adding a clear value proposition or call to action.")
    
    if not recs:
        recs.append("Your email is perfectly optimized for modern outreach!")
        
    return recs

def get_full_analysis(text: str, model_proba: float) -> Dict[str, Any]:
    breakdown = compute_breakdown(text)
    
    # Hybrid calculation for the final score
    # We combine model probability with local heuristic penalties/boosts
    final_score = int(model_proba * 100)
    
    # Penalties for critical failures
    if breakdown['personalization'] < 20: final_score -= 15
    if breakdown['tone'] < 30: final_score -= 20
    
    # Final clamping
    final_score = max(0, min(final_score, 100))
    
    if final_score >= 75: label = "STRONG EMAIL"
    elif final_score >= 45: label = "AVERAGE EMAIL"
    else: label = "WEAK EMAIL"
    
    return {
        "score": final_score,
        "label": label,
        "confidence": round(float(model_proba), 2),
        "breakdown": breakdown,
        "insights": [f"Readability Score: {round(compute_breakdown(text)['clarity'], 0)}/100"],
        "suggestions": get_recommendations(breakdown)
    }
