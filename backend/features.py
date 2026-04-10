"""
Shared Feature Extraction Module for AuthentiWrite
Extracts 28 stylometric/statistical features from text.
Used by both train_model.py and app.py to ensure consistency.
"""

import re
import math
import string
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Feature names in order (must match extract_features output)
FEATURE_NAMES = [
    'avg_sentence_length',
    'sentence_length_cv',
    'max_sentence_length',
    'min_sentence_length',
    'vocabulary_diversity',
    'yules_k',
    'hapax_ratio',
    'avg_word_length',
    'avg_syllable_count',
    'long_word_ratio',
    'short_word_ratio',
    'punctuation_ratio',
    'comma_ratio',
    'stop_word_ratio',
    'personal_pronoun_ratio',
    'contraction_count_norm',
    'sentence_starter_diversity',
    'transition_word_density',
    'formal_phrase_density',
    'question_ratio',
    'exclamation_ratio',
    'shannon_entropy',
    'bigram_entropy',
    'flesch_reading_ease',
    'gunning_fog',
    'paragraph_count_norm',
    'repetition_score',
    'passive_voice_ratio',
]

# Transition words commonly overused by AI
TRANSITION_WORDS = [
    'furthermore', 'moreover', 'additionally', 'consequently', 'therefore',
    'however', 'nevertheless', 'nonetheless', 'in addition', 'as a result',
    'subsequently', 'accordingly', 'thus', 'hence', 'meanwhile',
]

# Formal phrases typical of AI-generated text
FORMAL_PHRASES = [
    'it is important to note', 'it can be observed', 'it is worth mentioning',
    'it is essential to', 'it should be noted', 'this demonstrates',
    'this highlights', 'in summary', 'to summarize', 'in conclusion',
    'as a result', 'due to the fact that', 'in light of', 'with regard to',
    'it is evident that', 'plays a crucial role', 'it is imperative',
    'serves as a testament', 'in the realm of', 'a myriad of',
]

# Personal pronouns
PERSONAL_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our',
                      'ours', 'ourselves', 'you', 'your', 'yours', 'yourself'}

# Hedging words (indicate human uncertainty)
HEDGING_WORDS = [
    'i think', 'i believe', 'i feel', 'maybe', 'perhaps', 'probably',
    'might', 'seems', 'in my opinion', 'i guess', 'kind of', 'sort of',
    'i suppose', 'apparently', 'arguably',
]

# Contraction patterns
CONTRACTION_PATTERNS = re.compile(
    r"\b\w+(n't|'m|'re|'ve|'ll|'d|'s)\b", re.IGNORECASE
)

# Passive voice pattern (approximate)
PASSIVE_PATTERN = re.compile(
    r'\b(was|were|is|are|been|being|be|am)\s+\w+(ed|en|t)\b', re.IGNORECASE
)


def count_syllables(word):
    """Count syllables in a word using vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    # Remove trailing silent-e
    if len(word) > 2 and word.endswith('e') and word[-2] not in 'aeiou':
        word = word[:-1]
    vowel_groups = re.findall(r'[aeiouy]+', word)
    count = len(vowel_groups)
    return max(count, 1)


def calculate_shannon_entropy(words):
    """Calculate word-level Shannon entropy."""
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_bigram_entropy(words):
    """Calculate bigram-level entropy for word sequence predictability."""
    if len(words) < 3:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    freq = Counter(bigrams)
    total = len(bigrams)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_yules_k(words):
    """Calculate Yule's K measure of vocabulary richness.
    Lower K = richer vocabulary. Returns 0 for very short texts.
    """
    if len(words) < 10:
        return 0.0
    freq = Counter(words)
    freq_of_freq = Counter(freq.values())
    m1 = len(words)
    m2 = sum(i * i * v for i, v in freq_of_freq.items())
    if m1 == 0 or m1 == 1:
        return 0.0
    k = 10000.0 * (m2 - m1) / (m1 * m1)
    return max(k, 0.0)


def calculate_flesch_reading_ease(total_words, total_sentences, total_syllables):
    """Calculate Flesch Reading Ease score."""
    if total_sentences == 0 or total_words == 0:
        return 0.0
    score = (206.835
             - 1.015 * (total_words / total_sentences)
             - 84.6 * (total_syllables / total_words))
    return score


def calculate_gunning_fog(total_words, total_sentences, complex_word_count):
    """Calculate Gunning Fog Index."""
    if total_sentences == 0 or total_words == 0:
        return 0.0
    score = 0.4 * ((total_words / total_sentences)
                    + 100.0 * (complex_word_count / total_words))
    return score


def calculate_repetition_score(words):
    """Calculate average distance between repeated words.
    Lower score = more repetitive/predictable. Normalized by text length.
    """
    if len(words) < 10:
        return 0.0
    positions = {}
    distances = []
    for i, w in enumerate(words):
        if w in positions:
            distances.append(i - positions[w])
        positions[w] = i
    if not distances:
        return 1.0
    return np.mean(distances) / len(words)


def extract_features(text):
    """Extract 28 numerical features from text.

    Returns a list of floats in the order defined by FEATURE_NAMES.
    """
    text = text.strip()
    if len(text) < 20:
        return [0.0] * len(FEATURE_NAMES)

    # Tokenize
    sentences = sent_tokenize(text)
    words_raw = text.split()
    words_lower = [w.lower().strip(string.punctuation) for w in words_raw]
    words_lower = [w for w in words_lower if w]  # remove empty
    text_lower = text.lower()

    total_words = len(words_lower)
    total_sentences = max(len(sentences), 1)
    total_chars = len(text)

    if total_words < 5:
        return [0.0] * len(FEATURE_NAMES)

    # Sentence lengths
    sent_lens = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0
    sent_len_cv = (np.std(sent_lens) / avg_sent_len) if avg_sent_len > 0 else 0
    max_sent_len = max(sent_lens) if sent_lens else 0
    min_sent_len = min(sent_lens) if sent_lens else 0

    # Vocabulary
    unique_words = set(words_lower)
    vocab_diversity = len(unique_words) / total_words
    yules_k = calculate_yules_k(words_lower)
    word_freq = Counter(words_lower)
    hapax_count = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax_count / total_words

    # Word-level stats
    word_lengths = [len(w) for w in words_lower]
    avg_word_len = np.mean(word_lengths)
    syllable_counts = [count_syllables(w) for w in words_lower]
    avg_syllable = np.mean(syllable_counts)
    total_syllables = sum(syllable_counts)
    long_word_ratio = sum(1 for l in word_lengths if l > 6) / total_words
    short_word_ratio = sum(1 for l in word_lengths if l <= 3) / total_words

    # Punctuation
    punct_count = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = punct_count / total_chars if total_chars else 0
    comma_ratio = text.count(',') / total_words

    # Stop words
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    stop_word_ratio = sum(1 for w in words_lower if w in stop_words) / total_words

    # Personal pronouns
    pronoun_count = sum(1 for w in words_lower if w in PERSONAL_PRONOUNS)
    personal_pronoun_ratio = pronoun_count / total_words

    # Contractions
    contractions = CONTRACTION_PATTERNS.findall(text)
    contraction_count_norm = len(contractions) / total_words

    # Sentence starters
    starters = []
    for s in sentences:
        toks = s.strip().split()
        if toks:
            starters.append(toks[0].lower().strip(string.punctuation))
    unique_starters = len(set(starters))
    starter_diversity = unique_starters / total_sentences if total_sentences else 0

    # Transition word density
    transition_count = sum(1 for tw in TRANSITION_WORDS if tw in text_lower)
    transition_density = transition_count / total_sentences

    # Formal phrase density
    formal_count = sum(1 for fp in FORMAL_PHRASES if fp in text_lower)
    formal_density = formal_count / total_sentences

    # Question / exclamation ratios
    question_count = sum(1 for s in sentences if s.strip().endswith('?'))
    exclamation_count = sum(1 for s in sentences if s.strip().endswith('!'))
    question_ratio = question_count / total_sentences
    exclamation_ratio = exclamation_count / total_sentences

    # Entropy measures
    shannon_ent = calculate_shannon_entropy(words_lower)
    bigram_ent = calculate_bigram_entropy(words_lower)

    # Readability
    complex_word_count = sum(1 for sc in syllable_counts if sc >= 3)
    flesch = calculate_flesch_reading_ease(total_words, total_sentences, total_syllables)
    fog = calculate_gunning_fog(total_words, total_sentences, complex_word_count)

    # Paragraph structure
    paragraphs = [p for p in text.split('\n') if p.strip()]
    para_count_norm = len(paragraphs) / total_words if total_words else 0

    # Repetition
    rep_score = calculate_repetition_score(words_lower)

    # Passive voice
    passive_matches = PASSIVE_PATTERN.findall(text)
    passive_ratio = len(passive_matches) / total_sentences

    features = [
        avg_sent_len,
        sent_len_cv,
        max_sent_len,
        min_sent_len,
        vocab_diversity,
        yules_k,
        hapax_ratio,
        avg_word_len,
        avg_syllable,
        long_word_ratio,
        short_word_ratio,
        punctuation_ratio,
        comma_ratio,
        stop_word_ratio,
        personal_pronoun_ratio,
        contraction_count_norm,
        starter_diversity,
        transition_density,
        formal_density,
        question_ratio,
        exclamation_ratio,
        shannon_ent,
        bigram_ent,
        flesch,
        fog,
        para_count_norm,
        rep_score,
        passive_ratio,
    ]

    assert len(features) == len(FEATURE_NAMES), \
        f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}"

    return features
