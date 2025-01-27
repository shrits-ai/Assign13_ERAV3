import torch
import gradio as gr
from transformers import AutoTokenizer
from model_smol2 import LlamaForCausalLM, config_model
import requests
# Instantiate the model
model = LlamaForCausalLM(config_model)

# Correct URL for Google Drive direct download
url = "https://drive.google.com/uc?id=1tyZhudOcZRMaSLkzAWksVyQmbPQ0fi50"

response = requests.get(url)
if response.status_code == 200:
    with open("final_checkpoint.pt", "wb") as f:
        f.write(response.content)
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# Now load the checkpoint
try:
    checkpoint = torch.load("final_checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
except Exception as e:
    print(f"Error loading checkpoint: {e}")

# Load tokenizer (replace with the appropriate tokenizer if you're using a custom one)
# Load the tokenizer
TOKENIZER_PATH = "HuggingFaceTB/cosmo2-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"


# Text generation function
def generate_text(
    prompt, max_length=50, temperature=0.7, top_k=50, repetition_penalty=1.2, n_gram_block=2
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_tokens = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)  # model outputs
            
            # Check if the output is a dictionary with logits
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits'][:, -1, :]
            else:
                # If not, treat the output as a plain tensor
                logits = outputs[:, -1, :]

            # Repetition penalty
            for token_id in set(generated_tokens):
                logits[:, token_id] /= repetition_penalty

            # n-gram blocking
            if len(generated_tokens) >= n_gram_block:
                n_gram = tuple(generated_tokens[-n_gram_block:])
                for token_id in set(generated_tokens):
                    if generated_tokens[-n_gram_block:] == list(n_gram):
                        logits[:, token_id] -= 1e9

            logits /= temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# Gradio UI
def generate_response(prompt, max_length, temperature, top_k, repetition_penalty, n_gram_block):
    return generate_text(prompt, max_length, temperature, top_k, repetition_penalty, n_gram_block)

with gr.Blocks() as demo:
    gr.Markdown("# Smol2 Text Generator")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Input Prompt", placeholder="Enter your text prompt here...")
            max_length = gr.Slider(label="Max Length", minimum=10, maximum=200, value=50)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.1)
            top_k = gr.Slider(label="Top K", minimum=10, maximum=100, value=50, step=1)
            repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.2, step=0.1)
            n_gram_block = gr.Slider(label="N-Gram Blocking", minimum=1, maximum=5, value=2, step=1)
            generate_button = gr.Button("Generate Text")
        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=10)

    generate_button.click(
        generate_response,
        inputs=[prompt_input, max_length, temperature, top_k, repetition_penalty, n_gram_block],
        outputs=[output_text],
    )

demo.launch(share=True)