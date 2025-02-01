#pylint: disable=all
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch
import requests
from groq import Groq


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Models:
    def __init__(self):
        self.models = {
            "llama": (
                AutoTokenizer.from_pretrained("bert-base-uncased"),
                AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
            ),
            "custom": (
                AutoTokenizer.from_pretrained(
                    "Credioni/tinyBert_cs530-ta3",
                    #token=self.hf_token
                ),
                AutoModelForSequenceClassification.from_pretrained(
                    "Credioni/tinyBert_cs530-ta3",
                    #token=self.hf_token
                )
            )
        }

    def evaluate(self, input: str, model: str) -> str:
        if model in self.models.keys() and model != "llama":
            tokenizer, model = self.models[model]
            tokenized = tokenizer(input, return_tensors="pt", truncation=True, padding=True)

            if "token_type_ids" in tokenized:
                del tokenized["token_type_ids"]  # Remove token_type_ids for DistilBERT

            with torch.no_grad():
                output = model(**tokenized)

            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence_score = probs[0, predicted_class].item()

            predicted = "positive" if predicted_class == 1 else "negative"
            return predicted, confidence_score
        if model == "llama":
            return self._llama_evaluate(input)
        else:
            return "NONE"

    def _llama_evaluate(self, input: str) -> str:
        """Queries the Groq API for LLaMA 3 responses."""
        chat_completion = GROQ_CLIENT.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Classify the sentiment of this text as positive or negative, answer only positive or negative:\n" + input,
                }
            ],
            #logprobs=True, #not supported yet
            model="llama-3.3-70b-versatile",
        )
        confidence = 1.0

        #print(*chat_completion, sep="\n")

        content = chat_completion.choices[0].message.content.lower()
        pos_neg = "positive" if "positive" in content else "negative"
        return pos_neg, confidence

MODELS = Models()


@cross_origin()
@app.route('/analyze/', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    print(data)

    if not data or 'text' not in data or 'model' not in data:
        return jsonify({'error': 'Missing required fields. Please provide both text and model.'}), 400

    text  = data['text']
    model = data['model'].lower()

    # Validate model selection
    if model not in MODELS.models.keys():
        return jsonify({'error': f'Invalid model specified. Choose either {MODELS.models.keys()}.'}), 400

    try:
        sentiment, confidence = MODELS.evaluate(text, model)

        # Return results
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    #print(MODELS.evaluate("Hello, world!", "llama"))
    app.run(debug=True, port=5000)

