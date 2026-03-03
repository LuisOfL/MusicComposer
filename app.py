import os
import glob
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =====================
# CONFIG
# =====================
MIDI_FOLDER = "bitmidi_midis"   # Carpeta con tus MIDIs
SEED_MIDI = os.path.join(MIDI_FOLDER, "1674.mid")           # None = usa el primer MIDI
OUTPUT_MIDI = "generated.mid"

SEQ_LEN = 10
EPOCHS = 3       # ajustar según pruebas
BATCH_SIZE = 512  # usa lo máximo que quepa en tu GPU
LR = 0.001
TEMPERATURE = 1.0 # Control de creatividad: 0.7 conservador, 1.2 creativo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# =====================
# MIDI UTILS
# =====================
def midi_to_notes(path):
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        print(f"❌ MIDI inválido: {path}")
        return []

    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            notes.append(note.pitch)
    return notes

def notes_to_midi(notes, out_path):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    time = 0.0
    for pitch in notes:
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=time,
            end=time + 0.3
        )
        inst.notes.append(note)
        time += 0.3
    midi.instruments.append(inst)
    midi.write(out_path)

# =====================
# LOAD MIDIS
# =====================
print("🎹 Cargando MIDIs...")
paths = glob.glob(os.path.join(MIDI_FOLDER, "*.mid")) + \
        glob.glob(os.path.join(MIDI_FOLDER, "*.midi"))

if len(paths) == 0:
    raise ValueError("❌ No se encontraron MIDIs")

print(f"MIDIs encontrados: {len(paths)}")

all_notes = []
for p in paths:
    all_notes.extend(midi_to_notes(p))

print(f"Total de notas: {len(all_notes)}")

if len(all_notes) < SEQ_LEN + 1:
    raise ValueError("❌ No hay suficientes notas para entrenar")

# =====================
# ENCODE
# =====================
unique_notes = sorted(set(all_notes))
note_to_idx = {n: i for i, n in enumerate(unique_notes)}
idx_to_note = {i: n for n, i in note_to_idx.items()}

encoded = np.array([note_to_idx[n] for n in all_notes])

def make_sequences(seq):
    X, y = [], []
    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+SEQ_LEN])
    return np.array(X), np.array(y)

X, y = make_sequences(encoded)
print(f"Secuencias creadas: {len(X)}")

if len(X) == 0:
    raise ValueError("❌ Dataset vacío")

dataset = TensorDataset(
    torch.tensor(X, dtype=torch.long),
    torch.tensor(y, dtype=torch.long)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# MODEL
# =====================
class MidiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])

model = MidiLSTM(len(unique_notes)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# =====================
# TRAIN (Mixed Precision)
# =====================
print("🧠 Entrenando...")
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in tqdm(loader):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # mixed precision
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# =====================
# GENERATE (con sampling)
# =====================
print("🎼 Generando MIDI...")

if SEED_MIDI is None:
    SEED_MIDI = paths[0]

seed_notes = midi_to_notes(SEED_MIDI)
seed_encoded = [note_to_idx[n] for n in seed_notes if n in note_to_idx]

if len(seed_encoded) < SEQ_LEN:
    seed_encoded = encoded[:SEQ_LEN]

generated = seed_encoded[:SEQ_LEN]
model.eval()

for _ in range(500):  # cantidad de notas a generar
    x = torch.tensor(generated[-SEQ_LEN:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits / TEMPERATURE, dim=-1)
        next_note = torch.multinomial(probs, num_samples=1).item()
    generated.append(next_note)

final_notes = [idx_to_note[i] for i in generated]
notes_to_midi(final_notes, OUTPUT_MIDI)

print(f"✅ MIDI generado: {OUTPUT_MIDI}")