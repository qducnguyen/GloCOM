import os
import subprocess
import tempfile
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from utils.file_utils import split_text_word


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary, topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score

def compute_topic_coherence_on_wikipedia(top_word_path, cv_type='C_V'):
    """
    Compute the TC score on the Wikipedia dataset with Palmetto
    """
    pametto_jar = os.path.join("evaluations", "palmetto.jar")
    wiki_bd_dir = os.path.join("evaluations", "wiki_data", "wikipedia_bd")

    if not os.path.exists(pametto_jar):
        raise Exception(f"Pametto jar not found at {pametto_jar}")
    if not os.path.isdir(wiki_bd_dir):
        raise Exception(f"Wikipedia BD folder not found at {wiki_bd_dir}")
    if not os.path.exists(top_word_path):
        raise Exception(f"Top word file not found at {top_word_path}")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
        tmp_filename = tmp_file.name
    
    try: 
        java_command = ["java", "-jar", pametto_jar, wiki_bd_dir, cv_type, top_word_path]

        with open(tmp_filename, 'w') as outfile:
            process = subprocess.run(java_command, stdout=outfile, stderr=subprocess.PIPE, text=True)

        # Check if the Java process exited successfully
        if process.returncode != 0:
            error_msg = process.stderr.strip()
            raise RuntimeError(f"Java process failed with error: {error_msg}")


        cv_scores = []
        with open(tmp_filename, "r") as f:
            for line in f:
                if not line.startswith("202"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1])
                            cv_scores.append(score)
                        except ValueError:
                            pass
        if not cv_scores:
            raise ValueError("No coherence scores were extracted from the output.")

        average_cv_score = sum(cv_scores) / len(cv_scores)

        return cv_scores, average_cv_score

    finally:
        try:
            os.remove(tmp_filename)
        except OSError as e:
            print(f"Warning: Temporary file {tmp_filename} could not be removed. {e}")
