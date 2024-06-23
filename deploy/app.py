from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from gpt2_pytorch import GPT, GPTConfig  # Make sure your gpt2_pytorch.py is in the same directory or properly imported
import tiktoken

app = Flask(__name__)

# Load model
model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

# Load tokenizer
enc = tiktoken.get_encoding('gpt2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text', '')
    num_return_sequences = data.get('num_return_sequences', 1)
    max_length = data.get('max_length', 30)

    tokens = enc.encode(input_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to('cuda')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            tokens = torch.cat((tokens, next_token), dim=1)

    result_tokens = tokens[0, :max_length].tolist()
    decoded = enc.decode(result_tokens)

    return jsonify({'response': decoded})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
