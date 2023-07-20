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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
