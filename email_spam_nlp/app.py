from flask import Flask, render_template, request
import tensorflow as tf
from transformers import BertTokenizer

app = Flask(__name__)


tokenizer = BertTokenizer.from_pretrained('model/tokenizer')
loaded_model = tf.saved_model.load("model/saved_model")  
infer = loaded_model.signatures["serving_default"]       

def prepare_inputs(text):
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return encoding['input_ids'], encoding['attention_mask']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        try:
            input_ids, attention_mask = prepare_inputs(input_text)

            
            outputs = infer(input_ids=input_ids, attention_mask=attention_mask)

            pred_prob = outputs['output_0'].numpy()[0][0]  
            label = "Spam" if pred_prob > 0.5 else "Not Spam"

            return render_template('index.html', prediction=label, input_text=input_text)

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}", input_text=input_text)

    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
