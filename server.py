# # Higher token minimum condition for the metasummary:
# from flask import Flask, request, jsonify
# import requests
# import os

# app = Flask(__name__)

# HUGGINGFACE_URL = "https://m4luzlf8i1z7v700.us-east-1.aws.endpoints.huggingface.cloud"

# HEADERS = {
#     "Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}",
#     "Content-Type": "application/json"
# }

# @app.route('/', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     text = data['text']
#     num_sentences = data.get('num_sentences', 1)  # Default is 10 if not provided

#     min_length_multiplier = 10 if num_sentences == 20 else 3

#     payload = {
#         "inputs": text,
#         "options": {
#             "min_length": num_sentences * min_length_multiplier,
#             "max_length": num_sentences * 50,
#             "encoder_no_repeat_ngram_size": 3
#         }
#     }

#     response = requests.post(HUGGINGFACE_URL, headers=HEADERS, json=payload)
#     result = response.json()

#     sentences = result[0]["summary_text"].split('. ')
#     while len(sentences) < num_sentences:
#         sentences.append(' ')

#     return jsonify({"summary": sentences})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
















# LexRank OG

from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = Flask(__name__)

# Limiting the number of sentences to the availible ones:
@app.route('/', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = data['num_sentences']

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    all_sentences = list(parser.document.sentences)
    summary = summarizer(parser.document, min(num_sentences, len(all_sentences))) # Limit number of sentences to available ones.

    # Sort the summary sentences by their original order in the text
    ordered_summary = sorted(summary, key=all_sentences.index)

    # Filter out duplicate sentences and convert them to strings
    added_sentences = set() 
    sentences = []
    for sentence in ordered_summary:
        str_sentence = str(sentence)
        if str_sentence not in added_sentences:
            sentences.append(str_sentence)
            added_sentences.add(str_sentence)

    return jsonify({"summary": sentences})








# New Word2Vec-based endpoint for Semantic Library
@app.route('/get_related_authors', methods=['POST'])
def get_related_authors():
    data = request.json
    query = data['query']

    author_scores = {}
    for author in AUTHOR_INFO:  # Ensure that AUTHOR_INFO is accessible here
        score = compute_similarity(query, author["id"], model)
        author_scores[author["id"]] = score

    sorted_authors = sorted(author_scores.keys(), key=lambda x: author_scores[x], reverse=True)

    return jsonify(sorted_authors[:3])

def compute_similarity(query, author_name, model):
    query_vector = average_vector(query, model)
    author_vector = average_vector(author_name, model)
    
    similarity = cosine_similarity(query_vector, author_vector)
    return similarity

def average_vector(text, model):
    words = text.split()
    vector = sum([model[word] for word in words if word in model.vocab]) / len(words)
    return vector







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
