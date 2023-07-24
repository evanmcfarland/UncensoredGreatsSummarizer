# Higher token minimum condition for the metasummary:
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HUGGINGFACE_URL = "https://m4luzlf8i1z7v700.us-east-1.aws.endpoints.huggingface.cloud"

HEADERS = {
    "Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}",
    "Content-Type": "application/json"
}

@app.route('/', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = data.get('num_sentences', 1)  # Default is 10 if not provided

    min_length_multiplier = 10 if num_sentences == 20 else 3

    payload = {
        "inputs": text,
        "options": {
            "min_length": num_sentences * min_length_multiplier,
            "max_length": num_sentences * 50,
            "encoder_no_repeat_ngram_size": 3
        }
    }

    response = requests.post(HUGGINGFACE_URL, headers=HEADERS, json=payload)
    result = response.json()

    sentences = result[0]["summary_text"].split('. ')
    while len(sentences) < num_sentences:
        sentences.append(' ')

    return jsonify({"summary": sentences})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
















# # LexRank OG

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
