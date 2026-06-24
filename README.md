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
