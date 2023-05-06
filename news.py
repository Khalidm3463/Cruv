import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



# Load the dataaset into a DataFrame
news_df = pd.read_csv("news.csv")

# Fill the missing values with an empty string
news_df["content"] = news_df["content"].fillna("")

# Define a function to preprocess the text
def preprocess_text(text):
    """
    Remove unwanted characters and stopwords, and convert the text to lowercase.
    """
    # spacy.prefer_gpu(gpu_id=2)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    cleaned_text = " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    return cleaned_text.lower()

# Apply the preprocess_text function to the "content" column
news_df["content"] = news_df["content"].apply(preprocess_text)






# Define a function to rank the sentences in a document based on their importance
def rank_sentences(text, top_n=3):
    # Split the text into sentences
    sentences = text.split(".")
    
    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "", "", {"error": "No sentences found in text"}
    
    # Vectorize the sentences
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Compute the sentence similarity matrix
    sim_matrix = cosine_similarity(sentence_vectors)
    
    # Sort the sentence similarity matrix in descending order
    sim_matrix_sorted = np.argsort(-sim_matrix)
    
    # Select the top n most important sentences
    top_sentences_idx = sim_matrix_sorted[:top_n]
    
    # Get the actual sentences corresponding to the top n indices
    top_sentences = []
    for idx in top_sentences_idx:
        if idx.size > 0 and idx[0] < len(sentences):
            top_sentences.append(sentences[idx[0]])
    
    # Remove the top n most important sentences from the document
    removed_lines = "\n".join(sentence for sentence in sentences if sentence.strip() not in top_sentences)
    
    # Generate a summary of the top n sentences
    summary = {"top_sentences": top_sentences}
    
    return removed_lines, top_sentences, summary




# Split the dataset into train and test sets
train_set, test_set = train_test_split(news_df, test_size=0.1, random_state=42)

# Rank the sentences in the test set and store the results in a dataframe
results = pd.DataFrame(columns=["Original Content", "New Content", "Removed Lines", "Further Metrics"])
for index, row in test_set.iterrows():
    original_content = row["content"]
    new_content, removed_lines, summary = rank_sentences(original_content)
    if not new_content:
        continue
    new_row = pd.DataFrame({"Original Content": [original_content],
                            "New Content": [new_content],
                            "Removed Lines": [removed_lines],
                            "Further Metrics": [summary]})
    results = pd.concat([results, new_row], ignore_index=True)

# Save the results to a CSV file
results.to_csv("summary.csv", index=False)
