# LED Base Summarizer:

from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1

# Initialize the LED-based summarization model
hf_name = "pszemraj/led-base-book-summary"
summarizer = pipeline(
    "summarization",
    hf_name,
    device=device,
    no_repeat_ngram_size=3,
    repetition_penalty=3.5,
    num_beams=4,
    do_sample=False,
    early_stopping=True
)

@app.route('/', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    num_sentences = data.get('num_sentences', 10)  # Default is 10 if not provided

    # Using the LED-based model to summarize the text
    summary = summarizer(
        text,
        min_length=8,
        max_length=num_sentences*50,  # Assuming an average sentence has ~50 tokens. Adjust accordingly.
        encoder_no_repeat_ngram_size=3
    )

    sentences = summary[0]["generated_text"].split('. ')

    # Ensure the number of sentences matches the required count
    while len(sentences) < num_sentences:
        sentences.append(' ')  # or sentences.append(null);

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
