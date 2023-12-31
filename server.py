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
















# # Just LexRank OG

# from flask import Flask, request, jsonify
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer

# app = Flask(__name__)

# # Limiting the number of sentences to the availible ones:
# @app.route('/', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     text = data['text']
#     num_sentences = data['num_sentences']

#     parser = PlaintextParser.from_string(text, Tokenizer("english"))
#     summarizer = LexRankSummarizer()

#     all_sentences = list(parser.document.sentences)
#     summary = summarizer(parser.document, min(num_sentences, len(all_sentences))) # Limit number of sentences to available ones.

#     # Sort the summary sentences by their original order in the text
#     ordered_summary = sorted(summary, key=all_sentences.index)

#     # Filter out duplicate sentences and convert them to strings
#     added_sentences = set() 
#     sentences = []
#     for sentence in ordered_summary:
#         str_sentence = str(sentence)
#         if str_sentence not in added_sentences:
#             sentences.append(str_sentence)
#             added_sentences.add(str_sentence)

#     return jsonify({"summary": sentences})
    
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)



import nltk

from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import wordnet as wn

nltk.data.path.append('/app/nltk_data')

app = Flask(__name__)

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


@app.route('/related-authors', methods=['POST'])
def get_related_authors():
    data = request.json
    query = data['query']

    # Extracting author data from the request
    authors_data = data['authors']
    authors_categories = {author['name']: author['category'] for author in authors_data}

    related_authors = get_most_related_authors(query, authors_categories)

    return jsonify({"related_authors": related_authors})

def get_word_similarity(word1, word2):
    w1_synsets = wn.synsets(word1)
    w2_synsets = wn.synsets(word2)

    if not w1_synsets or not w2_synsets:
        return 0  # or some other default similarity score

    w1 = w1_synsets[0]
    w2 = w2_synsets[0]
    return w1.wup_similarity(w2)


def get_most_related_authors(query, authors_categories, num=3):
    query_terms = query.split()
    scores = {}

    for author, categories in authors_categories.items():
        score = sum(get_word_similarity(term, category) 
                    for term in query_terms for category in categories if wn.synsets(term) and wn.synsets(category))
        scores[author] = score
    
    sorted_authors = sorted(scores, key=scores.get, reverse=True)
    return sorted_authors[:num]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
