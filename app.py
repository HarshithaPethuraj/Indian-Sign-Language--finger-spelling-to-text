"""
ISL Fingerspelling -> Text : two-stage assistive pipeline
Stage 1 (Vision)   : CNN / MobileNetV2 recognizes a static ISL hand sign
Stage 2 (Language) : Transformer corrects a noisy letter stream into a real word

Weights are loaded from state_dict .pt files; architectures are rebuilt here,
so this is immune to full-model / version-metadata drift across environments.
"""

import json 
import math
import string

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

# ----------------------------------------------------------------------------
st.set_page_config(page_title="ISL Fingerspelling to Text", page_icon="🤟", layout="wide")
DEVICE = "cpu"  # HF Spaces free tier is CPU; models are tiny


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@st.cache_resource
def load_config():
    with open("deploy_config.json") as f:
        return json.load(f)


CFG = load_config()
IDX_TO_LETTER = CFG["idx_to_letter"]
NUM_LETTERS = CFG["num_letters"]
IMG = CFG["img"]
VOCAB = CFG["vocab"]
MAXLEN = CFG["maxlen"]
WORD_LIST = CFG["word_list"]
CHAMP_NAME = CFG["champ_name"]

STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
PAD_ID, SOS_ID, EOS_ID = STOI["<pad>"], STOI["<sos>"], STOI["<eos>"]
VSZ = len(VOCAB)
DICT = set(WORD_LIST)


# ----------------------------------------------------------------------------
# Stage 1 architectures (rebuilt to match training)
# ----------------------------------------------------------------------------
class SignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mobilenet(num_classes):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m


@st.cache_resource
def load_stage1():
    if CHAMP_NAME == "MobileNetV2":
        model = build_mobilenet(NUM_LETTERS)
    else:
        model = SignCNN(NUM_LETTERS)
    model.load_state_dict(torch.load("stage1_champion.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


EVAL_TF = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])


# ----------------------------------------------------------------------------
# Stage 2 architecture (Transformer corrector)
# ----------------------------------------------------------------------------
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=64):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerCorrector(nn.Module):
    def __init__(self, V, d=64, nhead=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD_ID)
        self.pos = PosEnc(d)
        self.tr = nn.Transformer(d_model=d, nhead=nhead, num_encoder_layers=layers,
                                 num_decoder_layers=layers, dim_feedforward=128,
                                 batch_first=True)
        self.fc = nn.Linear(d, V)

    def forward(self, src, tgt):
        sm = (src == PAD_ID)
        tm = (tgt == PAD_ID)
        L = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(tgt.device).bool()
        s = self.pos(self.emb(src))
        t = self.pos(self.emb(tgt))
        out = self.tr(s, t, tgt_mask=causal, src_key_padding_mask=sm,
                      tgt_key_padding_mask=tm, memory_key_padding_mask=sm)
        return self.fc(out)


@st.cache_resource
def load_stage2():
    model = TransformerCorrector(VSZ)
    model.load_state_dict(torch.load("stage2_transformer.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


# ----------------------------------------------------------------------------
# Stage 2 decoding helpers
# ----------------------------------------------------------------------------
def encode(word):
    return [STOI[c] for c in word if c in STOI]


def pad(ids, n):
    return ids + [PAD_ID] * (n - len(ids))


def beam_decode(model, noisy, beam=5, dict_bonus=2.0):
    """Beam search with a dictionary prior; same logic as the notebook."""
    s = torch.tensor([pad(encode(noisy), MAXLEN)]).to(DEVICE)
    beams = [([SOS_ID], 0.0)]
    for _ in range(MAXLEN):
        new = []
        for seq, lp in beams:
            if seq[-1] == EOS_ID:
                new.append((seq, lp)); continue
            ys = torch.tensor([seq]).to(DEVICE)
            with torch.no_grad():
                logp = F.log_softmax(model(s, ys)[0, -1], dim=-1)
            topv, topi = logp.topk(beam)
            for v, i in zip(topv.tolist(), topi.tolist()):
                new.append((seq + [i], lp + v))
        new.sort(key=lambda x: x[1], reverse=True)
        beams = new[:beam]
        if all(seq[-1] == EOS_ID for seq, _ in beams):
            break

    def to_word(seq):
        return "".join(ITOS[i] for i in seq[1:] if i not in (EOS_ID, PAD_ID, SOS_ID))

    scored = []
    for seq, lp in beams:
        w = to_word(seq)
        scored.append((w, lp + (dict_bonus if w in DICT else 0.0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🤟 Indian Sign Language - Fingerspelling to Text")
st.caption(
    "A two-stage assistive pipeline: a vision model reads each hand sign, "
    "and a Transformer language model repairs the noisy letter stream into real words."
)

tab1, tab2, tab3 = st.tabs(
    ["✋ Stage 1 - Recognize a hand sign", "🔤 Stage 2 - Correct a spelling", "ℹ️ About"]
)

# ---- Stage 1 ----
with tab1:
    st.subheader("Upload a hand-sign image")
    st.write(f"The model recognizes **{NUM_LETTERS} static ISL letters** "
             f"(A-Y, excluding J and Z which need motion).")
    up = st.file_uploader("Hand-sign image", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="Input", use_column_width=True)
        with c2:
            model1 = load_stage1()
            x = EVAL_TF(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model1(x)[0], dim=-1).cpu().numpy()
            top = probs.argsort()[::-1][:5]
            st.markdown(f"### Predicted letter: **{IDX_TO_LETTER[top[0]]}**")
            st.write(f"Confidence: **{probs[top[0]]*100:.1f}%**  ·  recognizer: {CHAMP_NAME}")
            st.write("Top-5:")
            for i in top:
                st.write(f"- {IDX_TO_LETTER[i]} — {probs[i]*100:.1f}%")
                st.progress(float(probs[i]))

# ---- Stage 2 ----
with tab2:
    st.subheader("Type a noisy fingerspelled word")
    st.write("Stage 1 occasionally misreads a sign. Stage 2 repairs the letter "
             "stream and assembles the intended word using beam search with a "
             "dictionary prior.")
    noisy = st.text_input("Noisy spelling (uppercase letters)", value="HELLP").upper()
    noisy = "".join(ch for ch in noisy if ch in string.ascii_uppercase)
    if st.button("Correct it") and noisy:
        model2 = load_stage2()
        fixed = beam_decode(model2, noisy, beam=5)
        c1, c2 = st.columns(2)
        c1.metric("Noisy input", noisy)
        c2.metric("Corrected", fixed)
        if fixed in DICT:
            st.success(f"'{fixed}' is a valid word in the model's dictionary.")
        else:
            st.info(f"Best reconstruction: '{fixed}'.")
    st.caption("Try words from the training corpus: HELLO, FRIEND, WATER, "
               "SCHOOL, THANK, LEARN, FAMILY, MORNING.")

# ---- About ----
with tab3:
    st.markdown(f"""
**How it works**

1. **Stage 1 - Vision.** A {CHAMP_NAME} model classifies a static ISL hand sign into
   one of {NUM_LETTERS} letters.
2. **Stage 2 - Language.** A character-level Transformer encoder-decoder takes the
   noisy letter stream and reconstructs the intended word, using beam search guided
   by a dictionary prior - the same principle behind phone autocorrect.

The two stages are coupled: Stage 2 was trained on the **vision model's real
confusion distribution**, so it learns to fix the errors the CNN actually makes.

**Scope.** The dataset is static images, so this handles fingerspelling, not dynamic
whole-word gestures. J and Z are excluded because they require motion.

Built by Harshitha Pethuraj · [GitHub](https://github.com/HarshithaPethuraj/PRAICP-1000-Indian-Sign-Language)
""")