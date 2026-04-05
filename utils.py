import re
import numpy as np
from typing import List, Dict, Any

# Keyword Lists
JOB_KEYWORDS = ["developer", "engineer", "api", "python", "backend", "frontend", "fullstack", "data", "ml", "ai", "cloud", "devops", "security", "software"]
POLITE_WORDS = ["please", "would", "appreciate", "thank you", "if you have time", "best regards", "looking forward"]
SPAM_WORDS = ["urgent", "immediately", "best ever", "guaranteed", "click here", "limited offer", "winner", "cash"]
COMPANY_KEYWORDS = ["company", "role", "team", "growth", "culture", "position", "scaling", "mission"]

def extract_features(text: str) -> np.ndarray:
    """
    Extracts structured numerical features from email text.
    Returns a numpy array of features.
    """
    text_lower = text.lower()
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 1. Word count
    word_count = len(words)
    
    # 2. Sentence count
    sentence_count = len(sentences)
    
    # 3. Average sentence length
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # 4. Readability proxy: Sentence length variance
    if sentence_count > 1:
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_variance = np.var(sentence_lengths)
    else:
        sentence_variance = 0
        
    # 5. Personalization features
    # Presence of recipient name (e.g., "Hi John")
    name_patterns = [r"hi\s+[a-z][a-z]+", r"dear\s+[a-z][a-z]+", r"hello\s+[a-z][a-z]+", r"hey\s+[a-z][a-z]+"]
    has_name = 1 if any(re.search(p, text_lower) for p in name_patterns) else 0
    
    # Presence of company-related words
    has_company = 1 if any(kw in text_lower for kw in COMPANY_KEYWORDS) else 0
    
    # 6. Keyword relevance
    keyword_matches = sum(1 for kw in JOB_KEYWORDS if kw in text_lower)
    
    # 7. Tone indicators
    polite_count = sum(1 for w in POLITE_WORDS if w in text_lower)
    spam_count = sum(1 for w in SPAM_WORDS if w in text_lower)
    
    # Return as a flat array for the model
    return np.array([
        word_count,
        sentence_count,
        avg_sentence_length,
        sentence_variance,
        has_name,
        has_company,
        keyword_matches,
        polite_count,
        spam_count
    ])

def compute_breakdown(text: str) -> Dict[str, int]:
    """
    Computes explainable 0-100 scores for different categories.
    """
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Personalization Score
    name_patterns = [r"hi\s+[a-z][a-z]+", r"dear\s+[a-z][a-z]+", r"hello\s+[a-z][a-z]+", r"hey\s+[a-z][a-z]+"]
    has_name = any(re.search(p, text_lower) for p in name_patterns)
    has_company = any(kw in text_lower for kw in COMPANY_KEYWORDS)
    pers_score = 0
    if has_name: pers_score += 60
    if has_company: pers_score += 40
    pers_score = min(pers_score, 100)
    
    # Length Score (Optimal 50-120 words)
    if 50 <= word_count <= 120:
        length_score = 100
    elif word_count < 50:
        length_score = max(0, word_count * 2) # Penalize short
    else:
        length_score = max(0, 100 - (word_count - 120) // 2) # Penalize long
        
    # Clarity Score (based on sentence length + readability)
    avg_len = word_count / max(sentence_count, 1)
    if 10 <= avg_len <= 20: # Sweet spot for clarity
        clarity_score = 100
    else:
        clarity_score = max(0, 100 - abs(avg_len - 15) * 5)
        
    # Relevance Score (Keyword matching)
    keyword_matches = sum(1 for kw in JOB_KEYWORDS if kw in text_lower)
    relevance_score = min(keyword_matches * 25, 100)
    
    # Tone Score (Polite vs Spam)
    polite_count = sum(1 for w in POLITE_WORDS if w in text_lower)
    spam_count = sum(1 for w in SPAM_WORDS if w in text_lower)
    tone_score = 70 # Neutral start
    tone_score += polite_count * 10
    tone_score -= spam_count * 20
    tone_score = max(0, min(tone_score, 100))
    
    return {
        "personalization": int(pers_score),
        "clarity": int(clarity_score),
        "relevance": int(relevance_score),
        "tone": int(tone_score),
        "length": int(length_score)
    }

def get_recommendations(breakdown: Dict[str, int]) -> List[str]:
    suggestions = []
    if breakdown['personalization'] < 60:
        suggestions.append("Add the recipient's name or mention their company/role specific details.")
    if breakdown['length'] < 80:
        suggestions.append("Aim for an optimal length between 50-120 words.")
    if breakdown['relevance'] < 50:
        suggestions.append("Incorporate more industry-specific keywords like 'engineer', 'api', or 'python'.")
    if breakdown['tone'] < 60:
        suggestions.append("Avoid urgent or spammy language; use more polite transitions.")
    if not suggestions:
        suggestions.append("Your email is exceptionally well-crafted!")
    return suggestions

def get_full_analysis(text: str, model_proba: float) -> Dict[str, Any]:
    breakdown = compute_breakdown(text)
    
    # Post-processing logic: Boost/Penalize final score
    # Scaling model_proba to 0-100
    base_score = model_proba * 100
    
    # Penalties
    if breakdown['personalization'] == 0: base_score -= 15
    if breakdown['length'] < 40: base_score -= 10
    if breakdown['tone'] < 40: base_score -= 20
    
    # Boosts
    if breakdown['relevance'] > 70: base_score += 10
    if breakdown['personalization'] > 80: base_score += 5
    
    final_score = max(0, min(int(base_score), 100))
    
    if final_score < 40:
        label = "WEAK EMAIL"
    elif final_score < 70:
        label = "AVERAGE EMAIL"
    else:
        label = "STRONG EMAIL"
        
    return {
        "score": final_score,
        "label": label,
        "confidence": round(float(model_proba), 2),
        "breakdown": breakdown,
        "insights": [f"Tone Score: {breakdown['tone']}", f"Relevance: {breakdown['relevance']}%"],
        "suggestions": get_recommendations(breakdown)
    }
