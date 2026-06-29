import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ===== CONFIG =====
# path to your saved fine-tuned model
MODEL_DIR = "animegpt-model"
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.85
TOP_K = 50
TOP_P = 0.92

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading AnimeGPT on {device}...")

# ===== LOAD MODEL =====
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
model.eval()
print("Model loaded!")


# ===== GENERATION =====
def generate_response(context_lines):
    """
    context_lines: list of plain dialogue strings (most recent last)
    Returns the model's next response as a string.
    """
    context_lines = context_lines[-6:]   # keep last 6 lines as context
    context_str = "\n".join(context_lines)
    prompt = f"<|context|>\n{context_str}\n<|response|>\n"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    text = text.split("\n")[0].strip()   # take only the first line
    return text if text else "..."


def extract_text(content):
    """Gradio 6.x may store message content as a plain string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content)


# ===== GRADIO CHAT LOGIC (new messages format) =====
def chat(user_message, history):
    """
    history: list of {"role": "user"/"assistant", "content": str} dicts (Gradio 5.x/6.x format)
    """
    history = history or []

    context_lines = [extract_text(msg["content"]) for msg in history]
    context_lines.append(user_message)

    response = generate_response(context_lines)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})

    return "", history


def clear_chat():
    return [], []


# ===== EXAMPLE PROMPTS =====
EXAMPLES = [
    "Are you really going to fight him alone?",
    "I never asked to be the chosen one.",
    "Why do you keep pushing everyone away?",
    "We'll never make it in time!",
    "I thought you said you didn't care about winning.",
]

# ===== UI =====
with gr.Blocks(title="AnimeGPT") as demo:

    gr.Markdown("# 🎌 AnimeGPT", elem_id="title")
    gr.Markdown(
        "A GPT-2 model fine-tuned on anime subtitles — Haikyu!!, Fairy Tail, "
        "JoJo's, Food Wars, OreGairu, ARIA and more.",
        elem_id="subtitle"
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        avatar_images=(
            None, "https://api.dicebear.com/7.x/bottts/svg?seed=animegpt"),
        label="",
        show_label=False,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Say something...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    with gr.Row():
        clear_btn = gr.Button("🗑 Clear Chat", size="sm", variant="secondary")

    gr.Markdown("<div id='examples-label'>Try these:</div>")
    with gr.Row():
        for ex in EXAMPLES:
            gr.Button(ex, size="sm").click(
                fn=lambda e=ex: e,
                outputs=msg_box
            )

    # Wire up interactions
    send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
    msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(clear_chat, outputs=[msg_box, chatbot])

    gr.Markdown(
        "<div style='text-align:center; color:#475569; font-size:0.8em; margin-top:12px'>"
        "Built with 🤗 Transformers + Gradio &nbsp;|&nbsp; "
        "<a href='https://github.com/RiteshNipane/AnimeGPT' style='color:#94a3b8'>GitHub</a>"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(
        share=True,   # gives a public URL for 72 hours
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        #chatbot { height: 480px; }
        #title { text-align: center; padding: 10px 0 0 0; }
        #subtitle { text-align: center; color: #94a3b8; margin-top: -8px; }
        #examples-label { color: #94a3b8; font-size: 0.85em; margin-bottom: 4px; }
        footer { display: none !important; }
        """,
    )
