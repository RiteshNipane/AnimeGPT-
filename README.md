# AnimeGPT

A GPT-2 model fine-tuned on anime subtitle dialogue to generate anime-style text responses.

> "Are you really going to fight him alone?"
> **AnimeGPT:** "I don't care. This is my fight."

---

## Anime in the Dataset

Haikyu!!, Fairy Tail, JoJo's Bizarre Adventure (Stardust Crusaders), Food Wars!, My Teen Romantic Comedy SNAFU, ARIA the Animation, Nagi no Asukara, Gargantia on the Verdurous Planet, Kimi ni Todoke, and more.

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/RiteshNipane/AnimeGPT
cd AnimeGPT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the chat UI
python app.py
```

Then open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## How It Works

### Architecture
- Base model: GPT-2 (124M parameters)
- Fine-tuning: Causal language modelling on anime dialogue pairs
- Tokenizer: GPT-2 BPE with added special tokens `<|context|>` and `<|response|>`

### Training Format
Each training example is structured as:

```
<|context|>
Are you really going to fight him alone?
I don't care. It's my fight.
<|response|>
Then I'll go with you. That's what nakama are for.
<|endoftext|>
```

The model is trained to generate a response given recent conversation history, rather than predicting text one character at a time with no context at all.

### Training Setup
| Setting | Value |
|---|---|
| Base model | GPT-2 |
| Dataset | 500 subtitle files (subset of ~3,800 available) |
| Epochs | 2 |
| Batch size | 32 (effective) |
| Learning rate | 5e-5 |
| Hardware | NVIDIA RTX 3050 6GB (laptop) |

---

## Known Limitations

This is a learning project, and the model has real limitations worth being upfront about:

- Responses are tonally consistent but not always topically relevant. The model picked up anime dialogue style well — short, punchy, emotionally charged lines — but doesn't reliably track the actual subject of the conversation. Ask "how are you?" and you might get an answer that sounds like anime dialogue but doesn't directly address the question.
- Why: GPT-2 at 124M parameters is small, and the training data is subtitle lines pulled from a dozen unrelated anime, chopped into context/response pairs. That teaches tone and fluency fast, but coherent multi-turn dialogue needs either much more training data, more epochs, or a larger base model — none of which were the goal here.
- No long-term memory. Each response only looks at the last 6 lines of conversation, and even within that window, relevance isn't guaranteed.

Possible future improvements: train on the full ~3,800-file dataset instead of the 500-file subset, more epochs, or fine-tune `gpt2-medium` instead of base `gpt2` for stronger coherence.

---

## Project Structure

```
AnimeGPT/
├── app.py                  # Gradio chat UI
├── AnimeGPT.ipynb          # Training notebook
├── requirements.txt
├── animegpt-model/         # Saved model (not in repo — see below)
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── README.md
```

Note: the trained model weights are not included in this repo due to file size. Download them from Hugging Face (link coming soon) or train your own using the notebook.

### Dataset Source

The subtitle data used for fine-tuning comes from the [Anime Subtitles dataset on Kaggle](https://www.kaggle.com/datasets/jef1056/anime-subtitles) by jef1056. It contains context/response pairs scraped from subtitles across hundreds of anime series. The dataset itself isn't included in this repo due to size — download it from Kaggle if you want to retrain or extend this model.

---

## Sample Outputs

**Context:**
```
We'll never make it in time!
Just trust me.
```
**AnimeGPT:** `I always do. That's the problem.`

---

**Context:**
```
Why do you keep pushing everyone away?
```
**AnimeGPT:** `Because if I let them in, I'd have something to lose.`

---

## Built With

- [Transformers](https://github.com/huggingface/transformers) — GPT-2 model & fine-tuning
- [Gradio](https://gradio.app) — Chat UI
- [PyTorch](https://pytorch.org) — Training backend

---

## License

MIT