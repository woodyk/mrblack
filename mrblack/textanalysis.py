"""
Text Analysis Module

A comprehensive text analysis library with various NLP techniques.
"""

import re
import math
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set, Union, Optional
from collections import Counter, defaultdict

# Third-party imports
import nltk
import numpy as np
from textblob import TextBlob
from langdetect import detect
from scipy.spatial import distance
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans

# Setup logger
logger = logging.getLogger(__name__)

# Ensure required NLTK data is available
def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')


def tokenize_text(text: str) -> Dict[str, Any]:
    """
    Tokenize text into words, sentences, and paragraphs.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Tokenized text components
    """
    # Save original case for NER and other cases where case matters
    original_text = text
    original_words = nltk.word_tokenize(text)
    original_sentences = nltk.sent_tokenize(text)
    
    # Convert to lowercase for most analysis
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    
    # Filter out punctuation for word-based analysis
    words_no_punct = [word for word in words if word.isalpha()]
    
    # Get paragraphs (text blocks separated by two newlines)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Additional paragraph detection for different formats
    if len(paragraphs) <= 1:
        # Try other common paragraph separators
        paragraphs = re.split(r'\n[\t ]*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If still only one paragraph, try to detect paragraph by indentation
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'\n[\t ]+', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return {
        "original_text": original_text,
        "original_words": original_words,
        "original_sentences": original_sentences,
        "text": text,
        "sentences": sentences,
        "words": words,
        "words_no_punct": words_no_punct,
        "paragraphs": paragraphs
    }


def preprocess_text(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess text with stopword removal, stemming, and lemmatization.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Preprocessing results
    """
    words_no_punct = tokens["words_no_punct"]
    
    # Stopwords
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))
    
    # Remove stopwords
    filtered_words = [word for word in words_no_punct if word not in stop_words]
    
    # Stemming and Lemmatization
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    return {
        "filtered_words": filtered_words,
        "stemmed_words": stemmed_words,
        "lemmatized_words": lemmatized_words,
        "stop_words": stop_words
    }


def _analyze_basic_stats(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate basic text statistics like counts and averages.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Basic statistics analysis
    """
    text = tokens["text"]
    words = tokens["words"]
    words_no_punct = tokens["words_no_punct"]
    sentences = tokens["sentences"]
    paragraphs = tokens["paragraphs"]
    original_words = tokens["original_words"]
    
    # Basic counts
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    spaces = char_count - char_count_no_spaces
    
    avg_word_length = sum(len(word) for word in words_no_punct) / len(words_no_punct) if words_no_punct else 0
    avg_sent_length = len(words_no_punct) / len(sentences) if sentences else 0
    avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    return {
        "word_count": len(words),
        "unique_word_count": len(set(words_no_punct)),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sent_length,
        "avg_paragraph_length": avg_para_length,
        "spaces": spaces,
        "punctuation_count": len(original_words) - len(words_no_punct),
    }


def _analyze_ngrams(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate n-grams and calculate their frequencies.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: N-gram analysis
    """
    words_no_punct = tokens["words_no_punct"]
    
    # N-grams generation
    bigrams = list(nltk.ngrams(words_no_punct, 2))
    trigrams = list(nltk.ngrams(words_no_punct, 3))
    fourgrams = list(nltk.ngrams(words_no_punct, 4))
    
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    fourgram_freq = Counter(fourgrams)
    
    return {
        "bigrams": bigrams,
        "trigrams": trigrams,
        "fourgrams": fourgrams,
        "bigram_freq": bigram_freq,
        "trigram_freq": trigram_freq,
        "fourgram_freq": fourgram_freq
    }


def _analyze_pos(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze part-of-speech distribution and related metrics.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Part-of-speech analysis
    """
    original_words = tokens["original_words"]
    words = tokens["words"]
    
    # Part-of-speech tagging
    pos_tags = nltk.pos_tag(original_words)
    pos_counts = Counter([tag for word, tag in pos_tags])
    
    # Count specific parts of speech
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    
    # Lexical density (content words / total words)
    content_pos_tags = ['NN', 'VB', 'JJ', 'RB']  # Base forms
    content_words = sum(1 for _, tag in pos_tags if any(tag.startswith(pos) for pos in content_pos_tags))
    lexical_density = content_words / len(words) if words else 0
    
    return {
        "pos_tags": pos_tags,
        "distribution": dict(pos_counts),
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adj_count,
        "adverb_count": adv_count,
        "noun_to_verb_ratio": noun_count / verb_count if verb_count else 0,
        "lexical_density": lexical_density,
    }


def _analyze_ner(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Named Entity Recognition.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Named entity analysis
    """
    pos_tags = nltk.pos_tag(tokens["original_words"])
    named_entities = {}
    entity_counts = {}
    
    try:
        ne_chunks = nltk.ne_chunk(pos_tags)
        
        # Process tree to extract named entities
        for chunk in ne_chunks:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = ' '.join(c[0] for c in chunk.leaves())
                
                if entity_type not in named_entities:
                    named_entities[entity_type] = []
                
                named_entities[entity_type].append(entity_text)
        
        # Count entities by type
        entity_counts = {entity_type: len(entities) for entity_type, entities in named_entities.items()}
        
    except Exception as ne_error:
        logger.warning(f"NER error: {ne_error}")
        named_entities = {}
        entity_counts = {}
    
    return {
        "counts": entity_counts,
        "entities": named_entities,
        "total_entities": sum(entity_counts.values())
    }


def count_syllables(word: str) -> int:
    """
    Count the number of syllables in a word (approximation).
    
    Args:
        word (str): Input word
        
    Returns:
        int: Estimated syllable count
    """
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    # Remove silent e
    if word.endswith('e'):
        word = word[:-1]
        
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
        
    return max(1, count)  # Return at least 1 syllable


def _analyze_readability(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate various readability metrics.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Readability metrics
    """
    words_no_punct = tokens["words_no_punct"]
    sentences = tokens["sentences"]
    
    avg_sent_length = len(words_no_punct) / len(sentences) if sentences else 0
    
    # Calculate syllables
    syllable_counts = [count_syllables(word) for word in words_no_punct]
    total_syllables = sum(syllable_counts)
    
    # Readability formulas
    # Flesch Reading Ease
    flesch_reading_ease = 206.835 - (1.015 * avg_sent_length) - (84.6 * (total_syllables / len(words_no_punct))) if words_no_punct else 0
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * avg_sent_length + 11.8 * (total_syllables / len(words_no_punct)) - 15.59 if words_no_punct else 0
    
    # Gunning Fog Index
    complex_words = sum(1 for word in words_no_punct if count_syllables(word) >= 3)
    complex_word_percentage = complex_words / len(words_no_punct) if words_no_punct else 0
    gunning_fog = 0.4 * (avg_sent_length + 100 * complex_word_percentage) if words_no_punct else 0
    
    # SMOG Index
    if len(sentences) >= 30:
        smog_sentences = sentences[:30]  # Use first 30 sentences
    else:
        smog_sentences = sentences  # Use all available
        
    smog_words = [word for sent in smog_sentences for word in nltk.word_tokenize(sent) if word.isalpha()]
    smog_complex_words = sum(1 for word in smog_words if count_syllables(word) >= 3)
    smog_index = 1.043 * math.sqrt(smog_complex_words * (30 / len(smog_sentences)) if smog_sentences else 0) + 3.1291
    
    # Dale-Chall Readability Formula
    # This would require a list of common words, simplified version:
    dale_chall_diff_words = sum(1 for word in words_no_punct if len(word) >= 7)
    dale_chall_score = 0.1579 * (dale_chall_diff_words / len(words_no_punct) * 100 if words_no_punct else 0) + 0.0496 * avg_sent_length
    
    if dale_chall_diff_words / len(words_no_punct) > 0.05 if words_no_punct else 0:
        dale_chall_score += 3.6365
    
    # Calculate lexical diversity
    lexical_diversity = len(set(words_no_punct)) / len(words_no_punct) if words_no_punct else 0
    
    return {
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog_index": gunning_fog,
        "smog_index": smog_index,
        "dale_chall_score": dale_chall_score,
        "syllable_count": total_syllables,
        "avg_syllables_per_word": total_syllables / len(words_no_punct) if words_no_punct else 0,
        "complex_word_percentage": complex_word_percentage,
        "lexical_diversity": lexical_diversity,
    } len(words_no_punct) if words_no_punct else 0,
        "complex_word_percentage": complex_word_percentage,
        "lexical_diversity": lexical_diversity,
    }


def _analyze_frequency(tokens: Dict[str, Any], preprocessing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze word frequencies and related metrics.
    
    Args:
        tokens (Dict): Tokenized text components
        preprocessing (Dict): Preprocessing results
        
    Returns:
        Dict: Frequency analysis
    """
    words_no_punct = tokens["words_no_punct"]
    filtered_words = preprocessing["filtered_words"]
    
    # Word frequencies
    word_freq = Counter(words_no_punct)
    filtered_word_freq = Counter(filtered_words)
    
    return {
        "word_freq": word_freq,
        "filtered_word_freq": filtered_word_freq,
        "most_common_words": word_freq.most_common(20),
        "most_common_meaningful_words": filtered_word_freq.most_common(20),
        "hapax_legomena": [word for word, count in word_freq.items() if count == 1],  # Words occurring only once
        "hapax_percentage": sum(1 for _, count in word_freq.items() if count == 1) / len(word_freq) if word_freq else 0,
    }


def _analyze_sentiment(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform sentiment analysis on the text.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Sentiment analysis results
    """
    original_text = tokens["original_text"]
    original_sentences = tokens["original_sentences"]
    
    # Overall sentiment
    blob = TextBlob(original_text)
    sentiment = blob.sentiment
    
    # Subjectivity by sentence
    sentence_sentiments = [TextBlob(sent).sentiment for sent in original_sentences]
    sentence_polarities = [sent.polarity for sent in sentence_sentiments]
    sentence_subjectivities = [sent.subjectivity for sent in sentence_sentiments]
    
    # Sentiment variance
    polarity_variance = np.var(sentence_polarities) if sentence_polarities else 0
    subjectivity_variance = np.var(sentence_subjectivities) if sentence_subjectivities else 0
    
    # Sentiment extremes
    most_positive_sentence = original_sentences[np.argmax(sentence_polarities)] if sentence_polarities else ""
    most_negative_sentence = original_sentences[np.argmin(sentence_polarities)] if sentence_polarities else ""
    most_subjective_sentence = original_sentences[np.argmax(sentence_subjectivities)] if sentence_subjectivities else ""
    most_objective_sentence = original_sentences[np.argmin(sentence_subjectivities)] if sentence_subjectivities else ""

    # Averaged categorical sentiment
    positive_threshold = 0.05
    negative_threshold = -0.05
    positive_count = sum(1 for polarity in sentence_polarities if polarity > positive_threshold)
    negative_count = sum(1 for polarity in sentence_polarities if polarity < negative_threshold)
    neutral_count = sum(1 for polarity in sentence_polarities if positive_threshold >= polarity >= negative_threshold)

    # Calculate percentages
    total_sentences = len(sentence_polarities) if sentence_polarities else 1  # Avoid division by zero
    positive_percentage = (positive_count / total_sentences) * 100
    negative_percentage = (negative_count / total_sentences) * 100
    neutral_percentage = (neutral_count / total_sentences) * 100

    # Determine categorical sentiment
    if positive_percentage > 60:
        categorical_sentiment = "very positive"
    elif positive_percentage > 40:
        categorical_sentiment = "positive"
    elif negative_percentage > 60:
        categorical_sentiment = "very negative"
    elif negative_percentage > 40:
        categorical_sentiment = "negative"
    elif neutral_percentage > 60:
        categorical_sentiment = "neutral"
    else:
        categorical_sentiment = "mixed"
    
    return {
        "overall_polarity": sentiment.polarity,  # -1 to 1 (negative to positive)
        "overall_subjectivity": sentiment.subjectivity,  # 0 to 1 (objective to subjective)
        "polarity_variance": polarity_variance,
        "subjectivity_variance": subjectivity_variance,
        "most_positive_sentence": most_positive_sentence,
        "most_negative_sentence": most_negative_sentence,
        "most_subjective_sentence": most_subjective_sentence,
        "most_objective_sentence": most_objective_sentence,
        "sentiment_shifts": sum(1 for i in range(1, len(sentence_polarities))
                              if (sentence_polarities[i-1] > 0 and sentence_polarities[i] < 0) or
                                 (sentence_polarities[i-1] < 0 and sentence_polarities[i] > 0)),
        "sentiment_progression": "positive_trend" if sum(1 for i in range(1, len(sentence_polarities))
                                                   if sentence_polarities[i] > sentence_polarities[i-1]) > len(sentence_polarities) / 2
                             else "negative_trend" if sum(1 for i in range(1, len(sentence_polarities))
                                                   if sentence_polarities[i] < sentence_polarities[i-1]) > len(sentence_polarities) / 2
                             else "neutral_trend",
        "categorical_sentiment": {
            "label": categorical_sentiment,
            "positive_percentage": positive_percentage,
            "negative_percentage": negative_percentage,
            "neutral_percentage": neutral_percentage,
            "positive_sentence_count": positive_count,
            "negative_sentence_count": negative_count,
            "neutral_sentence_count": neutral_count
        }
    }


def _analyze_summarization(tokens: Dict[str, Any], frequency_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create extractive summary from the text.
    
    Args:
        tokens (Dict): Tokenized text components
        frequency_results (Dict): Results from frequency analysis
        
    Returns:
        Dict: Text summarization results
    """
    sentences = tokens["sentences"]
    original_sentences = tokens["original_sentences"]
    word_freq = frequency_results["word_freq"]
    
    # Calculate TF-IDF
    tfidf_top_terms = []
    if len(sentences) > 3:  # Only compute if we have enough sentences
        try:
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get top tfidf terms for each sentence
            for i, sentence in enumerate(sentences):
                if i < tfidf_matrix.shape[0]:  # Safety check
                    feature_index = tfidf_matrix[i,:].nonzero()[1]
                    tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
                    tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                    tfidf_top_terms.append([(feature_names[i], score) for i, score in tfidf_scores[:5]])
        except Exception as tfidf_error:
            logger.warning(f"TF-IDF error: {tfidf_error}")
            tfidf_top_terms = []
    
    # Text summarization - extractive (simplified)
    # Rank sentences by importance (using word frequency as proxy)
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = nltk.word_tokenize(sentence.lower())
        sentence_words = [word for word in sentence_words if word.isalpha()]
        score = sum(word_freq.get(word, 0) for word in sentence_words)
        sentence_scores[original_sentences[i]] = score
    
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    summary = ' '.join([s[0] for s in top_sentences])
    
    return {
        "extractive_summary": summary,
        "key_sentences": [s[0] for s in top_sentences],
        "tfidf_top_terms": tfidf_top_terms
    }


def _analyze_cohesion(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze text cohesion metrics.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Cohesion analysis results
    """
    words = tokens["words"]
    
    # Text cohesion metrics
    transitional_words = ["however", "therefore", "furthermore", "consequently", "nevertheless", 
                         "thus", "meanwhile", "indeed", "moreover", "whereas", "conversely",
                         "similarly", "in addition", "in contrast", "specifically", "especially",
                         "particularly", "for example", "for instance", "in conclusion", "finally"]
    
    # Count transitional words and their positions
    transition_count = 0
    transition_positions = []
    
    for i, word in enumerate(words):
        if word in transitional_words or any(phrase in ' '.join(words[i:i+4]) for phrase in transitional_words if ' ' in phrase):
            transition_count += 1
            transition_positions.append(i / len(words) if words else 0)  # Normalized position
    
    # Cohesion score - higher means more transitional elements
    cohesion_score = (transition_count / len(words) * 100) if words else 0
    
    # Distribution of transitions (beginning, middle, end)
    if transition_positions:
        transitions_beginning = sum(1 for pos in transition_positions if pos < 0.33)
        transitions_middle = sum(1 for pos in transition_positions if 0.33 <= pos < 0.66)
        transitions_end = sum(1 for pos in transition_positions if pos >= 0.66)
    else:
        transitions_beginning = transitions_middle = transitions_end = 0
    
    return {
        "transitional_word_count": transition_count,
        "cohesion_score": cohesion_score,
        "transitions_beginning": transitions_beginning,
        "transitions_middle": transitions_middle,
        "transitions_end": transitions_end,
        "connector_distribution": "front_loaded" if transitions_beginning > transitions_middle and transitions_beginning > transitions_end
                              else "end_loaded" if transitions_end > transitions_beginning and transitions_end > transitions_middle
                              else "evenly_distributed"
    }


def _analyze_topics(tokens: Dict[str, Any], advanced: bool = False) -> Dict[str, Any]:
    """
    Perform topic modeling and clustering.
    
    Args:
        tokens (Dict): Tokenized text components
        advanced (bool): Whether to use more advanced topic modeling techniques
        
    Returns:
        Dict: Topic analysis results
    """
    sentences = tokens["sentences"]
    preprocessing = preprocess_text(tokens)
    filtered_word_freq = Counter(preprocessing["filtered_words"])
    
    # Simple topic keywords
    topic_keywords = filtered_word_freq.most_common(10)
    
    results = {
        "possible_topics": topic_keywords
    }
    
    # Advanced topic modeling
    if advanced and len(sentences) >= 5:
        try:
            # Create a document-term matrix
            # Create Count Vectorizer
            count_vectorizer = CountVectorizer(stop_words='english', min_df=2)
            count_matrix = count_vectorizer.fit_transform(sentences)
            count_feature_names = count_vectorizer.get_feature_names_out()
            
            # Train LDA model if we have enough data
            if count_matrix.shape[0] >= 5 and count_matrix.shape[1] >= 10:
                n_topics = min(3, count_matrix.shape[0] - 1)  # Choose appropriate number of topics
                lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                lda_model.fit(count_matrix)
                
                # Get top words for each topic
                topics = []
                for topic_idx, topic in enumerate(lda_model.components_):
                    top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
                    top_words = [count_feature_names[i] for i in top_words_idx]
                    topics.append(top_words)
                    
                results["topics"] = topics
                
                # Alternative: Use TruncatedSVD (similar to LSA) for topic extraction
                svd_model = TruncatedSVD(n_components=n_topics, random_state=42)
                svd_model.fit(count_matrix)
                
                # Get top words for each component (topic)
                svd_topics = []
                for topic_idx, topic in enumerate(svd_model.components_):
                    top_words_idx = topic.argsort()[:-11:-1]  # Top 10 words
                    top_words = [count_feature_names[i] for i in top_words_idx]
                    svd_topics.append(top_words)
                    
                results["svd_topics"] = svd_topics
            else:
                results["topics"] = ["Insufficient data for topic modeling"]
                results["svd_topics"] = ["Insufficient data for topic modeling"]
        except Exception as topic_error:
            logger.warning(f"Topic modeling error: {topic_error}")
            results["topics"] = ["Error in topic modeling"]
            results["svd_topics"] = ["Error in topic modeling"]
            
        # Clustering sentences
        try:
            if len(sentences) >= 5:
                # Use TF-IDF vectors for clustering
                tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
                
                # Determine number of clusters
                n_clusters = min(3, len(sentences) - 1)
                km = KMeans(n_clusters=n_clusters, random_state=42)
                km.fit(tfidf_matrix)
                
                # Get sentence clusters
                clusters = km.labels_.tolist()
                
                # Organize sentences by cluster
                sentence_clusters = defaultdict(list)
                for i, cluster in enumerate(clusters):
                    sentence_clusters[cluster].append(tokens["original_sentences"][i])
                    
                results["sentence_clusters"] = dict(sentence_clusters)
                
                # Get top terms per cluster
                cluster_terms = {}
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                terms = tfidf_vectorizer.get_feature_names_out()
                
                for i in range(n_clusters):
                    cluster_top_terms = [terms[ind] for ind in order_centroids[i, :10]]
                    cluster_terms[i] = cluster_top_terms
                    
                results["cluster_terms"] = cluster_terms
            else:
                results["sentence_clusters"] = {"note": "Insufficient data for clustering"}
                results["cluster_terms"] = {"note": "Insufficient data for clustering"}
        except Exception as cluster_error:
            logger.warning(f"Clustering error: {cluster_error}")
            results["sentence_clusters"] = {"error": str(cluster_error)}
            results["cluster_terms"] = {"error": str(cluster_error)}
    
    return results_sentences"][i])
                    
                results["sentence_clusters"] = {"error": str(cluster_error)}
            results["cluster_terms"] = {"error": str(cluster_error)}
    
    return results


def _analyze_network(tokens: Dict[str, Any], preprocessing: Dict[str, Any], advanced: bool = False) -> Dict[str, Any]:
    """
    Perform text network analysis.
    
    Args:
        tokens (Dict): Tokenized text components
        preprocessing (Dict): Preprocessing results
        advanced (bool): Whether to perform advanced network analysis
        
    Returns:
        Dict: Network analysis results
    """
    if not advanced:
        return {"note": "Advanced analysis not requested"}
        
    sentences = tokens["sentences"]
    filtered_words = preprocessing["filtered_words"]
    stop_words = preprocessing["stop_words"]
    
    # Text network analysis
    try:
        # Create word co-occurrence network
        G = nx.Graph()
        
        # Add nodes (words)
        for word in set(filtered_words):
            G.add_node(word)
        
        # Add edges (co-occurrences within sentences)
        for sentence in sentences:
            sent_words = [word for word in nltk.word_tokenize(sentence.lower()) 
                         if word.isalpha() and word not in stop_words]
            
            # Add edges between all pairs of words in the sentence
            for i, word1 in enumerate(sent_words):
                for word2 in sent_words[i+1:]:
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] += 1
                    else:
                        G.add_edge(word1, word2, weight=1)
        
        # Calculate network metrics if we have enough data
        if G.number_of_nodes() > 2:
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            top_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Betweenness centrality for central connector words
            if G.number_of_nodes() < 1000:  # Skip for very large networks
                betweenness_centrality = nx.betweenness_centrality(G, k=min(G.number_of_nodes(), 100))
                top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                top_betweenness = [("Network too large", 0)]
            
            # Extract clusters/communities (simplified)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            
            return {
                "central_terms": [word for word, score in top_degree_centrality],
                "connector_terms": [word for word, score in top_betweenness],
                "clusters_count": len(components),
                "largest_cluster_size": len(largest_component)
            }
        else:
            return {"note": "Insufficient data for network analysis"}
    except Exception as network_error:
        logger.warning(f"Network analysis error: {network_error}")
        return {"error": str(network_error)}


def _analyze_similarity(tokens: Dict[str, Any], advanced: bool = False) -> Dict[str, Any]:
    """
    Analyze term similarity and co-occurrence patterns.
    
    Args:
        tokens (Dict): Tokenized text components
        advanced (bool): Whether to perform advanced similarity analysis
        
    Returns:
        Dict: Similarity analysis results
    """
    if not advanced or len(tokens["words_no_punct"]) < 20:
        return {"note": "Advanced analysis not requested or insufficient data"}
        
    sentences = tokens["sentences"]
    
    try:
        similarity_analysis = {}

        # Create co-occurrence matrix (simplified word embedding alternative)
        vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=2)
        X = vectorizer.fit_transform(sentences)
        features = vectorizer.get_feature_names_out()

        # Convert to array for easier manipulation
        X_array = X.toarray()

        # Compute pairwise distance between terms
        # Use cosine similarity between term vectors
        term_similarity = {}

        for i, term1 in enumerate(features):
            if i < len(X_array[0]):  # Safety check
                term_vec1 = X_array[:, i]
                for j, term2 in enumerate(features):
                    if j < len(X_array[0]) and i != j:  # Skip self-comparison
                        term_vec2 = X_array[:, j]
                        # Compute cosine similarity
                        similarity = 1 - distance.cosine(term_vec1, term_vec2)

                        if similarity > 0.5:  # Only keep high similarity pairs
                            if term1 not in term_similarity:
                                term_similarity[term1] = []
                            term_similarity[term1].append((term2, similarity))

        # Sort and keep top similar terms
        for term in term_similarity:
            term_similarity[term] = sorted(term_similarity[term], key=lambda x: x[1], reverse=True)[:5]

        # Get top terms with most connections
        top_connected_terms = sorted(term_similarity.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        similarity_analysis["term_similarity"] = {term: similar for term, similar in top_connected_terms}

        return similarity_analysis
    except Exception as sim_error:
        logger.warning(f"Similarity analysis error: {sim_error}")
        return {"error": str(sim_error)}


def _analyze_syntax(tokens: Dict[str, Any], advanced: bool = False) -> Dict[str, Any]:
    """
    Analyze syntactic complexity of the text.
    
    Args:
        tokens (Dict): Tokenized text components
        advanced (bool): Whether to perform advanced syntax analysis
        
    Returns:
        Dict: Syntactic complexity metrics
    """
    if not advanced:
        return {"note": "Advanced analysis not requested"}
        
    original_sentences = tokens["original_sentences"]
    
    try:
        # Syntactic complexity
        syntactic_complexity = {}
        
        # Count depth of clauses (approximation using POS patterns)
        clause_markers = [',', 'that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how']
        subordinating_conjunctions = ['after', 'although', 'as', 'because', 'before', 'if', 'since', 'though', 'unless', 'until', 'when', 'where', 'while']
        
        clause_complexity = []
        
        for sentence in original_sentences:
            tokens = nltk.word_tokenize(sentence.lower())
            clause_markers_count = sum(1 for token in tokens if token in clause_markers)
            subordinating_count = sum(1 for token in tokens if token in subordinating_conjunctions)
            
            # Approximate clause depth
            clause_depth = 1 + clause_markers_count + subordinating_count
            clause_complexity.append(clause_depth)
        
        syntactic_complexity["avg_clause_depth"] = sum(clause_complexity) / len(clause_complexity) if clause_complexity else 0
        syntactic_complexity["max_clause_depth"] = max(clause_complexity) if clause_complexity else 0
        
        # Approximation of phrase types
        sentence_pos = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in original_sentences]
        
        # Count noun phrases (approximated by adjective-noun sequences)
        noun_phrases = []
        for sentence_tags in sentence_pos:
            for i in range(len(sentence_tags) - 1):
                if sentence_tags[i][1].startswith('JJ') and sentence_tags[i+1][1].startswith('NN'):
                    noun_phrases.append(f"{sentence_tags[i][0]} {sentence_tags[i+1][0]}")
        
        # Count verb phrases (approximated by adverb-verb sequences)
        verb_phrases = []
        for sentence_tags in sentence_pos:
            for i in range(len(sentence_tags) - 1):
                if sentence_tags[i][1].startswith('RB') and sentence_tags[i+1][1].startswith('VB'):
                    verb_phrases.append(f"{sentence_tags[i][0]} {sentence_tags[i+1][0]}")
        
        syntactic_complexity["estimated_noun_phrases"] = len(noun_phrases)
        syntactic_complexity["estimated_verb_phrases"] = len(verb_phrases)
        syntactic_complexity["noun_verb_phrase_ratio"] = len(noun_phrases) / len(verb_phrases) if verb_phrases else 0
        
        return syntactic_complexity
    except Exception as syntax_error:
        logger.warning(f"Syntactic analysis error: {syntax_error}")
        return {"error": str(syntax_error)}


def _analyze_style(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze text style and formality.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Style analysis results
    """
    original_text = tokens["original_text"]
    word_freq = Counter(tokens["words"])
    
    # Detect writing style (tentative classification)
    style_markers = {}

    # Formality markers
    formal_markers = ["therefore", "thus", "consequently", "furthermore", "moreover", "hence",
                    "accordingly", "subsequently", "previously", "regarding", "concerning"]
    informal_markers = ["anyway", "basically", "actually", "kinda", "like", "so", "pretty",
                      "totally", "really", "hopefully", "maybe", "ok", "okay", "stuff"]

    style_markers["formal_marker_count"] = sum(word_freq.get(marker, 0) for marker in formal_markers)
    style_markers["informal_marker_count"] = sum(word_freq.get(marker, 0) for marker in informal_markers)
    style_markers["contraction_count"] = len(re.findall(r"\b\w+'[ts]|\b\w+n't\b|\b\w+'ll\b|\b\w+'re\b|\b\w+'ve\b", original_text))
    style_markers["exclamation_count"] = original_text.count("!")
    style_markers["question_count"] = original_text.count("?")
    style_markers["parenthetical_count"] = len(re.findall(r"\([^)]*\)", original_text))
    style_markers["semicolon_count"] = original_text.count(";")

    # Tentative style classification
    formality_score = (style_markers["formal_marker_count"] + style_markers["semicolon_count"] * 2 +
                      style_markers["parenthetical_count"]) - (style_markers["informal_marker_count"] +
                      style_markers["contraction_count"] + style_markers["exclamation_count"] * 2)

    if formality_score > 5:
        style_markers["style_classification"] = "formal_academic"
    elif formality_score > 0:
        style_markers["style_classification"] = "formal"
    elif formality_score > -5:
        style_markers["style_classification"] = "neutral"
    elif formality_score > -10:
        style_markers["style_classification"] = "informal"
    else:
        style_markers["style_classification"] = "very_informal"

    return style_markers


def _analyze_rhetoric(tokens: Dict[str, Any], ngram_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze rhetorical patterns in the text.
    
    Args:
        tokens (Dict): Tokenized text components
        ngram_results (Dict): Results from n-gram analysis
        
    Returns:
        Dict: Rhetorical pattern analysis
    """
    original_text = tokens["original_text"]
    original_words = tokens["original_words"]
    original_sentences = tokens["original_sentences"]
    bigram_freq = ngram_results["bigram_freq"]
    trigram_freq = ngram_results["trigram_freq"]
    
    # Detect potential rhetoric patterns
    rhetoric_patterns = {}

    # Repetition patterns
    repeated_bigrams = [bg for bg, count in bigram_freq.items() if count > 2]
    repeated_trigrams = [tg for tg, count in trigram_freq.items() if count > 2]

    # Question patterns
    rhetorical_questions = sum(1 for sentence in original_sentences if
                             sentence.endswith("?") and any(word in sentence.lower() for word in
                                                          ["why", "who", "what", "how", "when", "where"]))

    # Comparison patterns (similes)
    similes = len(re.findall(r"\b(like|as) a\b|\b(like|as) the\b", original_text.lower()))

    # Alliteration (simplified detection)
    alliterations = 0
    for i in range(len(original_words) - 2):
        if (len(original_words[i]) > 0 and len(original_words[i+1]) > 0 and len(original_words[i+2]) > 0 and
            original_words[i][0].lower() == original_words[i+1][0].lower() == original_words[i+2][0].lower()):
            alliterations += 1

    rhetoric_patterns["repeated_phrases"] = repeated_bigrams + repeated_trigrams
    rhetoric_patterns["rhetorical_questions"] = rhetorical_questions
    rhetoric_patterns["similes"] = similes
    rhetoric_patterns["alliterations"] = alliterations

    return rhetoric_patterns


def _analyze_bias(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect potential bias indicators in the text.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Bias analysis results
    """
    word_freq = Counter(tokens["words"])
    words = tokens["words"]
    
    # Potential bias indicators
    bias_indicators = {}

    # Extreme language
    extreme_markers = ["always", "never", "all", "none", "every", "only", "impossible",
                      "absolutely", "undoubtedly", "certainly", "definitely", "completely",
                      "total", "totally", "utterly", "best", "worst", "perfect"]
    extreme_language = sum(word_freq.get(marker, 0) for marker in extreme_markers)

    # Loaded language
    emotionally_loaded = ["amazing", "terrible", "awesome", "horrible", "wonderful", "dreadful",
                         "excellent", "awful", "extraordinary", "appalling", "incredible", "disgusting"]
    loaded_language = sum(word_freq.get(marker, 0) for marker in emotionally_loaded)

    bias_indicators["extreme_language_count"] = extreme_language
    bias_indicators["loaded_language_count"] = loaded_language
    bias_indicators["extreme_language_ratio"] = extreme_language / len(words) if words else 0
    bias_indicators["loaded_language_ratio"] = loaded_language / len(words) if words else 0

    # Simple bias classification
    if bias_indicators["extreme_language_ratio"] > 0.05 or bias_indicators["loaded_language_ratio"] > 0.05:
        bias_indicators["bias_classification"] = "potentially_biased"
    else:
        bias_indicators["bias_classification"] = "relatively_neutral"

    return bias_indicators


def _analyze_emotion(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze emotional content of text beyond sentiment.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Emotion analysis results
    """
    word_freq = Counter(tokens["words"])
    
    try:
        emotion_analysis = {}

        # Basic emotion lexicons
        emotions = {
            "joy": ["happy", "joy", "delight", "glad", "pleased", "excited", "thrilled", "elated"],
            "sadness": ["sad", "unhappy", "sorrow", "depressed", "miserable", "downcast", "gloomy"],
            "anger": ["angry", "mad", "furious", "outraged", "annoyed", "irritated", "livid"],
            "fear": ["afraid", "fear", "scared", "terrified", "worried", "anxious", "nervous"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"]
        }

        # Count emotion words
        emotion_counts = {}
        for emotion, emotion_words in emotions.items():
            emotion_counts[emotion] = sum(word_freq.get(word, 0) for word in emotion_words)

        # Calculate dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ("neutral", 0)

        # Calculate emotion intensity (percent of all emotion words that belong to dominant emotion)
        total_emotion_words = sum(emotion_counts.values())
        dominant_intensity = (dominant_emotion[1] / total_emotion_words) if total_emotion_words > 0 else 0

        emotion_analysis["emotion_counts"] = emotion_counts
        emotion_analysis["dominant_emotion"] = dominant_emotion[0]
        emotion_analysis["dominant_intensity"] = dominant_intensity
        emotion_analysis["emotional_diversity"] = len([e for e, c in emotion_counts.items() if c > 0])

        return emotion_analysis
    except Exception as emo_error:
        logger.warning(f"Emotion analysis error: {emo_error}")
        return {"error": str(emo_error)}


def _analyze_contextual(tokens: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze contextual elements in the text.
    
    Args:
        tokens (Dict): Tokenized text components
        
    Returns:
        Dict: Contextual analysis results
    """
    word_freq = Counter(tokens["words"])
    
    # Contextual analysis (identifying context patterns)
    contextual_analysis = {}

    # Temporal references
    temporal_markers = ["today", "yesterday", "tomorrow", "now", "then", "before", "after",
                       "while", "during", "soon", "later", "earlier", "recently", "ago"]
    temporal_references = sum(word_freq.get(marker, 0) for marker in temporal_markers)

    # Spatial references
    spatial_markers = ["here", "there", "above", "below", "behind", "in front", "nearby",
                      "inside", "outside", "around", "between", "among", "everywhere"]
    spatial_references = sum(word_freq.get(marker, 0) for marker in spatial_markers)

    # Personal references
    first_person = sum(word_freq.get(marker, 0) for marker in ["i", "me", "my", "mine", "we", "us", "our", "ours"])
    second_person = sum(word_freq.get(marker, 0) for marker in ["you", "your", "yours"])
    third_person = sum(word_freq.get(marker, 0) for marker in ["he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"])

    contextual_analysis["temporal_references"] = temporal_references
    contextual_analysis["spatial_references"] = spatial_references
    contextual_analysis["first_person_references"] = first_person
    contextual_analysis["second_person_references"] = second_person
    contextual_analysis["third_person_references"] = third_person
    contextual_analysis["narration_perspective"] = "first_person" if first_person > second_person and first_person > third_person else \
                                          "second_person" if second_person > first_person and second_person > third_person else \
                                          "third_person"

    return contextual_analysis


def _analyze_domain_specific(tokens: Dict[str, Any], domain_specific: str = None) -> Dict[str, Any]:
    """
    Perform domain-specific analysis.
    
    Args:
        tokens (Dict): Tokenized text components
        domain_specific (str): Specific domain to analyze ("academic", "social_media", "customer_reviews")
        
    Returns:
        Dict: Domain-specific analysis results
    """
    if not domain_specific:
        return {"note": "No domain-specific analysis requested"}
        
    original_text = tokens["original_text"]
    words = tokens["words"]
    word_freq = Counter(words)
    
    domain_analysis = {}
    
    if domain_specific == "academic":
        # Academic writing analysis
        academic_terms = ["hypothesis", "theory", "analysis", "data", "method", "research", 
                        "study", "evidence", "results", "conclusion", "findings", "literature",
                        "significant", "therefore", "thus", "however", "moreover"]
        hedge_words = ["may", "might", "could", "appears", "seems", "suggests", "indicates",
                      "possibly", "perhaps", "likely", "unlikely", "generally", "usually"]
        
        academic_term_count = sum(word_freq.get(term, 0) for term in academic_terms)
        hedge_word_count = sum(word_freq.get(term, 0) for term in hedge_words)
        
        domain_analysis["academic_term_density"] = academic_term_count / len(words) if words else 0
        domain_analysis["hedging_density"] = hedge_word_count / len(words) if words else 0
        domain_analysis["citation_count"] = original_text.count("et al") + re.findall(r"\(\d{4}\)", original_text).__len__()
        
    elif domain_specific == "social_media":
        # Social media analysis
        hashtags = re.findall(r"#\w+", original_text)
        mentions = re.findall(r"@\w+", original_text)
        urls = re.findall(r"https?://\S+", original_text)
        emojis = re.findall(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", original_text)
        
        slang_terms = ["lol", "omg", "wtf", "idk", "tbh", "imo", "fwiw", "ymmv", "tl;dr", "ftw"]
        slang_count = sum(1 for word in words if word.lower() in slang_terms)
        
        domain_analysis["hashtag_count"] = len(hashtags)
        domain_analysis["mention_count"] = len(mentions)
        domain_analysis["url_count"] = len(urls)
        domain_analysis["emoji_count"] = len(emojis)
        domain_analysis["slang_terms"] = slang_count
        domain_analysis["engagement_markers"] = len(hashtags) + len(mentions) + len(emojis) + slang_count
        
    elif domain_specific == "customer_reviews":
        # Customer review analysis
        product_terms = ["product", "quality", "price", "value", "recommend", "purchase",
                       "buy", "bought", "worth", "money", "shipping", "delivery", "package",
                       "arrived", "customer", "service", "return", "warranty", "replacement"]
        
        rating_terms = ["star", "stars", "rating", "rate", "perfect", "excellent", "good", 
                      "average", "poor", "terrible", "worst", "best"]
        
        feature_terms = ["feature", "features", "works", "worked", "functionality", "design",
                       "size", "weight", "color", "material", "battery", "screen", "interface"]
        
        product_term_count = sum(word_freq.get(term, 0) for term in product_terms)
        rating_term_count = sum(word_freq.get(term, 0) for term in rating_terms)
        feature_term_count = sum(word_freq.get(term, 0) for term in feature_terms)
        
        # Find potential ratings (e.g. "5 star", "3.5 out of 5")
        rating_patterns = re.findall(r"(\d+\.?\d*)\s*(star|stars|out of \d+)", original_text.lower())
        
        domain_analysis["product_term_density"] = product_term_count / len(words) if words else 0
        domain_analysis["rating_term_density"] = rating_term_count / len(words) if words else 0
        domain_analysis["feature_term_density"] = feature_term_count / len(words) if words else 0
        domain_analysis["potential_ratings"] = rating_patterns
        domain_analysis["recommendation_language"] = "recommend" in original_text.lower() or "would buy" in original_text.lower()
    
    return domain_analysis0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+", original_text)
        
        slang_terms = ["lol", "omg", "wtf", "idk", "tbh", "imo", "fwiw", "ymmv", "tl;dr", "ftw"]
        slang_count = sum(1 for word in words if word.lower() in slang_terms)
        
        domain_analysis["hashtag_count"] = len(hashtags)
        domain_analysis["mention_count"] = len(mentions)
        domain_analysis["url_count"] = len(urls)
        domain_analysis["emoji_count"] = len(emojis)
        domain_analysis["slang_terms"] = slang_count
        domain_analysis["engagement_markers"] = len(hashtags) + len(mentions) + len(emojis) + slang_count
        
    elif domain_specific == "customer_reviews":
        # Customer review analysis
        product_terms = ["product", "quality", "price", "value", "recommend", "purchase",
                       "buy", "bought", "worth", "money", "shipping", "delivery", "package",
                       "arrived", "customer", "service", "return", "warranty", "replacement"]
        
        rating_terms = ["star", "stars", "rating", "rate", "perfect", "excellent", "good", 
                      "average", "poor", "terrible", "worst", "best"]
        
        feature_terms = ["feature", "features", "works", "worked", "functionality", "design",
                       "size", "weight", "color", "material", "battery", "screen", "interface"]
        
        product_term_count = sum(word_freq.get(term, 0) for term in product_terms)
        rating_term_count = sum(word_freq.get(term, 0) for term in rating_terms)
        feature_term_count = sum(word_freq.get(term, 0) for term in feature_terms)
        
        # Find potential ratings (e.g. "5 star", "3.5 out of 5")
        rating_patterns = re.findall(r"(\d+\.?\d*)\s*(star|stars|out of \d+)", original_text.lower())
        
        domain_analysis["product_term_density"] = product_term_count / len(words) if words else 0
        domain_analysis["rating_term_density"] = rating_term_count / len(words) if words else 0
        domain_analysis["feature_term_density"] = feature_term_count / len(words) if words else 0
        domain_analysis["potential_ratings"] = rating_patterns
        domain_analysis["recommendation_language"] = "recommend" in original_text.lower() or "would buy" in original_text.lower()
    
    return domain_analysis


def analyze_language(text: str) -> str:
    """
    Detect the language of the text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected language code
    """
    try:
        return detect(text)
    except:
        return "unknown"


def analyze_text(text: str, advanced: bool = False, domain_specific: str = None) -> Dict[str, Any]:
    """
    Perform comprehensive text analytics with advanced NLP techniques.
    
    Args:
        text (str): Input text
        advanced (bool): Whether to perform computationally intensive advanced analysis
        domain_specific (str): Optional domain for specialized analysis (e.g., "academic", "social_media", "customer_reviews")
        
    Returns:
        Dict: Comprehensive analysis results
    """
    try:
        # Ensure NLTK data is available
        download_nltk_data()
        
        # Tokenize text into components
        tokens = tokenize_text(text)
        
        # Preprocess text
        preprocessing = preprocess_text(tokens)
        
        # Basic statistics analysis
        basic_stats = _analyze_basic_stats(tokens)
        
        # N-grams analysis
        ngram_results = _analyze_ngrams(tokens)
        
        # Part-of-speech analysis
        pos_results = _analyze_pos(tokens)
        
        # Named Entity Recognition
        ner_results = _analyze_ner(tokens)
        
        # Readability metrics
        readability_results = _analyze_readability(tokens)
        
        # Frequency analysis
        frequency_results = _analyze_frequency(tokens, preprocessing)
        
        # Sentiment analysis
        sentiment_results = _analyze_sentiment(tokens)
        
        # Text summarization
        summarization_results = _analyze_summarization(tokens, frequency_results)
        
        # Text cohesion metrics
        cohesion_results = _analyze_cohesion(tokens)
        
        # Style analysis
        style_results = _analyze_style(tokens)
        
        # Rhetorical patterns
        rhetoric_results = _analyze_rhetoric(tokens, ngram_results)
        
        # Bias indicators
        bias_results = _analyze_bias(tokens)
        
        # Emotion analysis
        emotion_results = _analyze_emotion(tokens)
        
        # Contextual analysis
        contextual_results = _analyze_contextual(tokens)
        
        # Topic analysis (basic or advanced)
        topic_results = _analyze_topics(tokens, advanced)
        
        # Collect all results
        analysis_results = {
            "basic_stats": basic_stats,
            "part_of_speech": pos_results,
            "named_entities": ner_results,
            "readability": readability_results,
            "frequency_analysis": {
                "most_common_words": frequency_results["most_common_words"],
                "most_common_meaningful_words": frequency_results["most_common_meaningful_words"],
                "most_common_bigrams": ngram_results["bigram_freq"].most_common(10),
                "most_common_trigrams": ngram_results["trigram_freq"].most_common(5),
                "most_common_fourgrams": ngram_results["fourgram_freq"].most_common(3),
                "hapax_legomena": frequency_results["hapax_legomena"],
                "hapax_percentage": frequency_results["hapax_percentage"],
            },
            "sentiment": sentiment_results,
            "preprocessing": {
                "filtered_words_count": len(preprocessing["filtered_words"]),
                "stopwords_removed": len(tokens["words_no_punct"]) - len(preprocessing["filtered_words"]),
                "stemmed_words_sample": preprocessing["stemmed_words"][:10] if preprocessing["stemmed_words"] else [],
                "lemmatized_words_sample": preprocessing["lemmatized_words"][:10] if preprocessing["lemmatized_words"] else [],
            },
            "summarization": summarization_results,
            "cohesion": cohesion_results,
            "style_analysis": style_results,
            "rhetoric_patterns": rhetoric_results,
            "bias_indicators": bias_results,
            "emotion_analysis": emotion_results,
            "contextual_analysis": contextual_results,
            "topic_analysis": topic_results,
            "language": analyze_language(text)
        }
        
        # Advanced analysis if requested
        if advanced:
            # Network analysis
            network_results = _analyze_network(tokens, preprocessing, advanced)
            analysis_results["network_analysis"] = network_results
            
            # Term similarity analysis
            similarity_results = _analyze_similarity(tokens, advanced)
            analysis_results["similarity_analysis"] = similarity_results
            
            # Syntactic complexity
            syntax_results = _analyze_syntax(tokens, advanced)
            analysis_results["syntactic_complexity"] = syntax_results
        
        # Domain-specific analysis if requested
        if domain_specific:
            domain_results = _analyze_domain_specific(tokens, domain_specific)
            analysis_results["domain_analysis"] = domain_results
        
        # Get metadata about the analysis
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "analysis_version": "2.0",
            "text_length_category": "short" if len(tokens["words"]) < 100 else "medium" if len(tokens["words"]) < 500 else "long",
            "advanced_analysis_performed": advanced,
            "domain_specific_analysis": domain_specific
        }
        
        analysis_results["metadata"] = metadata
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Individual analysis functions (to be called separately)

# Individual analysis functions (to be called directly)

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Perform sentiment analysis on text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Sentiment analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_sentiment(tokens)
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {"error": str(e)}


def analyze_basic_stats(text: str) -> Dict[str, Any]:
    """
    Calculate basic text statistics.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Basic text statistics
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_basic_stats(tokens)
    except Exception as e:
        logger.error(f"Basic stats analysis error: {e}")
        return {"error": str(e)}


def analyze_readability(text: str) -> Dict[str, Any]:
    """
    Calculate readability metrics for text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Readability metrics
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_readability(tokens)
    except Exception as e:
        logger.error(f"Readability analysis error: {e}")
        return {"error": str(e)}


def analyze_ngrams(text: str) -> Dict[str, Any]:
    """
    Generate and analyze n-grams in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: N-gram analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_ngrams(tokens)
    except Exception as e:
        logger.error(f"N-gram analysis error: {e}")
        return {"error": str(e)}


def analyze_pos(text: str) -> Dict[str, Any]:
    """
    Analyze part-of-speech distribution in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Part-of-speech analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_pos(tokens)
    except Exception as e:
        logger.error(f"POS analysis error: {e}")
        return {"error": str(e)}


def analyze_ner(text: str) -> Dict[str, Any]:
    """
    Perform Named Entity Recognition on text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Named entity analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_ner(tokens)
    except Exception as e:
        logger.error(f"NER analysis error: {e}")
        return {"error": str(e)}


def analyze_frequency(text: str) -> Dict[str, Any]:
    """
    Analyze word frequency distribution in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Frequency analysis results
    """
    try:
        tokens = tokenize_text(text)
        preprocessing = preprocess_text(tokens)
        return _analyze_frequency(tokens, preprocessing)
    except Exception as e:
        logger.error(f"Frequency analysis error: {e}")
        return {"error": str(e)}


def analyze_summarization(text: str) -> Dict[str, Any]:
    """
    Generate extractive summary from text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Text summarization results
    """
    try:
        tokens = tokenize_text(text)
        frequency_results = _analyze_frequency(tokens, preprocess_text(tokens))
        return _analyze_summarization(tokens, frequency_results)
    except Exception as e:
        logger.error(f"Summarization analysis error: {e}")
        return {"error": str(e)}


def analyze_cohesion(text: str) -> Dict[str, Any]:
    """
    Analyze text cohesion metrics.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Cohesion analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_cohesion(tokens)
    except Exception as e:
        logger.error(f"Cohesion analysis error: {e}")
        return {"error": str(e)}


def analyze_topics(text: str, advanced: bool = False) -> Dict[str, Any]:
    """
    Perform topic modeling on text.
    
    Args:
        text (str): Input text
        advanced (bool): Whether to use advanced topic modeling
        
    Returns:
        Dict: Topic analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_topics(tokens, advanced)
    except Exception as e:
        logger.error(f"Topic analysis error: {e}")
        return {"error": str(e)}


def analyze_network(text: str) -> Dict[str, Any]:
    """
    Perform text network analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Network analysis results
    """
    try:
        tokens = tokenize_text(text)
        preprocessing = preprocess_text(tokens)
        return _analyze_network(tokens, preprocessing, True)
    except Exception as e:
        logger.error(f"Network analysis error: {e}")
        return {"error": str(e)}


def analyze_similarity(text: str) -> Dict[str, Any]:
    """
    Analyze term similarity in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Similarity analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_similarity(tokens, True)
    except Exception as e:
        logger.error(f"Similarity analysis error: {e}")
        return {"error": str(e)}


def analyze_syntax(text: str) -> Dict[str, Any]:
    """
    Analyze syntactic complexity of text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Syntactic complexity metrics
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_syntax(tokens, True)
    except Exception as e:
        logger.error(f"Syntax analysis error: {e}")
        return {"error": str(e)}


def analyze_style(text: str) -> Dict[str, Any]:
    """
    Analyze writing style and formality.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Style analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_style(tokens)
    except Exception as e:
        logger.error(f"Style analysis error: {e}")
        return {"error": str(e)}


def analyze_rhetoric(text: str) -> Dict[str, Any]:
    """
    Analyze rhetorical patterns in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Rhetorical pattern analysis
    """
    try:
        tokens = tokenize_text(text)
        ngram_results = _analyze_ngrams(tokens)
        return _analyze_rhetoric(tokens, ngram_results)
    except Exception as e:
        logger.error(f"Rhetoric analysis error: {e}")
        return {"error": str(e)}


def analyze_bias(text: str) -> Dict[str, Any]:
    """
    Detect potential bias indicators in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Bias analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_bias(tokens)
    except Exception as e:
        logger.error(f"Bias analysis error: {e}")
        return {"error": str(e)}


def analyze_emotion(text: str) -> Dict[str, Any]:
    """
    Analyze emotional content of text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Emotion analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_emotion(tokens)
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return {"error": str(e)}


def analyze_contextual(text: str) -> Dict[str, Any]:
    """
    Analyze contextual elements in text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Contextual analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_contextual(tokens)
    except Exception as e:
        logger.error(f"Contextual analysis error: {e}")
        return {"error": str(e)}


def analyze_domain_specific(text: str, domain: str) -> Dict[str, Any]:
    """
    Perform domain-specific analysis on text.
    
    Args:
        text (str): Input text
        domain (str): Specific domain to analyze
        
    Returns:
        Dict: Domain-specific analysis results
    """
    try:
        tokens = tokenize_text(text)
        return _analyze_domain_specific(tokens, domain)
    except Exception as e:
        logger.error(f"Domain-specific analysis error: {e}")
        return {"error": str(e)} dict(sentence_clusters)
                
                # Get top terms per cluster
                cluster_terms = {}
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]
                terms = tfidf_vectorizer.get_feature_names_out()
                
                for i in range(n_clusters):
                    cluster_top_terms = [terms[ind] for ind in order_centroids[i, :10]]
                    cluster_terms[i] = cluster_top_terms
                    
                results["cluster_terms"] = cluster_terms
            else:
                results["sentence_clusters"] = {"note": "Insufficient data for clustering"}
                results["cluster_terms"] = {"note": "Insufficient data for clustering"}
        except Exception as cluster_error:
            logger.warning(f"Clustering error: {cluster_error}")
            results["sentence_clusters"] =