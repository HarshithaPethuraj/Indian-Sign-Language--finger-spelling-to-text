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

# Indian Sign Language - Fingerspelling to Text

A two-stage assistive-communication pipeline that converts Indian Sign Language
fingerspelling into readable text.

- **Stage 1 (Vision):** a MobileNetV2 recognizer classifies a static ISL hand sign
  into one of 24 letters (A-Y; J and Z need motion and are excluded).
- **Stage 2 (Language):** a character-level Transformer encoder-decoder repairs the
  noisy letter stream and assembles the intended word, using beam search with a
  dictionary prior.

The two stages are coupled: Stage 2 is trained on the vision model's real confusion
distribution, so it learns to fix the errors the CNN actually makes.

## Files

- `app.py` - Streamlit app (rebuilds both architectures, loads weights from state_dict)
- `stage1_champion.pt` - Stage 1 recognizer weights (MobileNetV2)
- `stage2_transformer.pt` - Stage 2 Transformer corrector weights
- `deploy_config.json` - class order, vocab, dictionary, and input config

## Notebook

Full analysis: `PRAICP-1000-Indian-Sign-Language.ipynb`
Repo: https://github.com/HarshithaPethuraj/PRAICP-1000-Indian-Sign-Language

Built by Harshitha Pethuraj.
