---
title: ISL Fingerspelling To Text
emoji: 🤟
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# 🤟 Indian Sign Language - Fingerspelling to Text

> A two-stage assistive-communication pipeline that converts Indian Sign Language
> fingerspelling into readable text. A MobileNetV2 vision model reads each static
> hand sign; a Transformer language model repairs the noisy letter stream into real
> words - with the two stages genuinely coupled through the vision model's real
> confusion distribution.

**Live demo:** https://huggingface.co/spaces/harshithapethuraj/isl-fingerspelling-to-text
**Notebook:** `PRAICP-1000-Indian-Sign-Language.ipynb`
**Built by:** Harshitha Pethuraj | DataMites AIE Capstone - PRAICP-1000

---

## The Problem

Sign language is the primary mode of communication for millions of people with
hearing and speech impairments. Within sign language, **fingerspelling** - signing
a word letter by letter - is how signers convey names, technical terms, and any
word without a dedicated sign.

The technical challenge is that this is **not a single classification problem.**
Two distinct problems are stacked on top of each other:

1. **Vision:** recognize each static ISL hand sign from an image (24 letters, A-Y;
   J and Z require motion and cannot be captured in a still image).
2. **Language:** a real user fingerspells a *word*, not isolated letters - and the
   vision model will inevitably misread some signs. A per-letter recognizer that is
   correct most of the time still spells whole words wrong. The system must correct
   a noisy letter stream and assemble it into the intended word.

Most ISL projects stop at step 1. This one builds the full pipeline.

---

## What This Project Does

### Stage 1 - Vision (Hand-Sign Recognition)

A convolutional network classifies each static ISL hand sign into one of 24 letters.
Two models are built and compared on identical data:

| Model | Val Accuracy | Train Loss | Params | Champion |
|---|---|---|---|---|
| Custom CNN (from scratch) | 89.8% | 0.477 | 113,304 | No |
| MobileNetV2 (transfer) | **99.9%** | **0.012** | 2,254,616 | **Yes** |

MobileNetV2 carries ImageNet-pretrained features that transfer directly to hand-shape
recognition. The from-scratch CNN is kept as a meaningful lightweight baseline.

### Stage 2 - Language (Spelling Correction)

The recognized letters form a noisy sequence. A character-level sequence-to-sequence
model repairs the noisy letter stream and assembles the intended word. Two
architectures are built and compared:

| Model | Params | Final Loss |
|---|---|---|
| LSTM encoder-decoder | 204,253 | 2.025 |
| Transformer encoder-decoder | **171,421** | **1.271** |

The Transformer's self-attention lets every output character attend to every input
character directly; the LSTM must compress all context into a single hidden vector.
The Transformer wins on both parameter efficiency and loss at this scale.

### Distinctive Contributions

**1. Confusion-aware coupling between the two stages.**
Instead of training Stage 2 on uniform random noise, the CNN's real confusion matrix
is converted into a per-letter error distribution P(predicted | true), and Stage 2
is retrained on noise drawn from that distribution. The corrector learns to fix the
visually-similar-sign errors the vision model actually makes - M/N and U/V type
confusions - rather than imaginary random substitutions. This genuinely couples the
two stages rather than bolting them together.

**2. Beam search with a dictionary language prior.**
A beam-search decoder keeps multiple candidate spellings alive in parallel and applies
a log-prob bonus to candidates that form real dictionary words - the same principle
behind phone autocorrect and speech-to-text rescoring. Measurably beats greedy
decoding under heavy corruption.

**3. Honest scoping.**
The dataset is static images. Rather than overclaiming "video sign-language
recognition," the project is scoped to exactly what the data supports (fingerspelling),
and adds genuine NLP value at the stage where it belongs.

---

## Architecture

User fingerspells a word
        |
        v
[Stage 1 - Vision]
MobileNetV2 classifies each
static ISL hand sign -> letter
        |
        v
Noisy letter stream
(e.g. HELLO -> HLLLO)
        |
        v
[Stage 2 - Language]
Transformer encoder-decoder
trained on CNN real confusion
distribution -> corrected word
        |
        v
Beam search + dictionary prior
-> final word output

---

## Dataset

- Source: DataMites Indian Sign Language fingerspelling (PRAICP-1000)
- Size: 4,972 images across 24 static ISL letters (A-Y)
- Balance: min 116 images (G) to max 259 images (B)
- Excluded: J and Z (require motion, cannot be captured in still images)
- Note: Dataset is proprietary and not included in this repository

---

## Tech Stack

Python - PyTorch - torchvision - OpenCV - scikit-learn - NumPy - pandas -
Matplotlib - Seaborn - Streamlit - Hugging Face Spaces

---

## Project Structure

PRAICP-1000-Indian-Sign-Language/
├── PRAICP-1000-Indian-Sign-Language.ipynb  # full analysis notebook
├── app.py                                  # Streamlit deployment app
├── requirements.txt                        # HF Spaces dependencies
├── README.md                               # this file
├── stage1_champion.pt                      # MobileNetV2 weights (state_dict)
├── stage2_transformer.pt                   # Transformer corrector weights
└── deploy_config.json                      # class order, vocab, word list, config

---

## Deployment

### Hugging Face Spaces (Streamlit)

The app rebuilds both architectures in app.py and loads PyTorch state_dict
weights - immune to version-metadata drift across environments (the PyTorch
equivalent of the weights-only pattern used across this portfolio).

Three tabs:
- Stage 1: upload a hand-sign image, get the predicted letter with top-5
  confidence scores
- Stage 2: type a noisy spelling, get it corrected via beam search with a
  dictionary prior
- About: pipeline explanation and scope

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place stage1_champion.pt, stage2_transformer.pt, and deploy_config.json
in the same directory as app.py before running.

---

## Key Results

- MobileNetV2 achieved 99.9% validation accuracy on 24-class ISL hand-sign
  recognition, with even the weakest letters scoring above F1 0.989 (R: 0.989,
  X: 0.990).
- The Transformer corrector outperformed the LSTM on both parameter count
  (171K vs 204K) and final training loss (1.271 vs 2.025).
- Beam search with a dictionary prior recovers real words under heavy corruption
  where greedy decoding fails entirely.
- Confusion-aware Stage 2 training couples the vision and language stages through
  the CNN's empirical error distribution rather than a uniform noise assumption.

---

## Challenges Faced

- RAM crash from collecting raw pixels into a Python list - fixed by resizing
  to 64x64 before collection and accumulating into NumPy uint8 arrays (1 byte
  per pixel vs 28 bytes per Python int).
- Kernel restart after crash wiped all variables - always run all cells from
  the top after a Colab kernel restart.
- Greedy decoding failed on all 6 qualitative demo words - directly motivated
  beam search (Section 8) and confusion-aware retraining (Section 8.5).
- HF Spaces Streamlit version mismatch - use_container_width unavailable on
  the installed version; fixed by switching to use_column_width.
- No real noisy-to-clean training pairs exist - the entire Stage 2 corpus was
  synthetically engineered by programmatically corrupting clean words to mimic
  Stage-1 recognition errors.

---

## Limitations

- Static fingerspelling only (24 letters, no J/Z); dynamic whole-word gestures
  require video.
- Stage 2 corpus covers common English words; rare or technical words may not
  reconstruct correctly.
- Stages are trained separately; a jointly trained pipeline could propagate
  uncertainty from vision into the language model.
- Dataset collected under controlled plain backgrounds - real-world webcam
  performance may be lower due to clutter, variable lighting, and skin-tone
  diversity gaps.

---

## Future Work

- Train Stage 2 on a large English dictionary (tens of thousands of words) for
  true generalisation beyond a fixed vocabulary.
- Feed CNN per-letter softmax probabilities into Stage 2 as uncertainty weights,
  not just the argmax letter prediction.
- Extend to dynamic signs with a video model (CNN+LSTM or video Transformer)
  to cover J, Z, and whole-word ISL gestures.
- Joint end-to-end training of both stages so vision uncertainty propagates
  directly into the language correction.

---

## About

Built as part of the DataMites AI Engineer certification (PTID-AIE-MAY-26-11194),
Project Code PRAICP-1000.

**Harshitha Pethuraj**
- GitHub: https://github.com/HarshithaPethuraj
- LinkedIn: https://www.linkedin.com/in/harshitha-pethuraj-13738129b/
- HF Spaces: https://huggingface.co/spaces/harshithapethuraj
