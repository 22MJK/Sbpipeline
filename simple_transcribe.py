import torchaudio
import whisper
from speechbrain.inference import VAD
from speechbrain.inference.speaker import EncoderClassifier
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# 1. load audio
wav_path = "/opt/data/majikui/audios/smalltest.wav"
signal, fs = torchaudio.load(wav_path)

# if multi-channel, take the first channel
if signal.shape[0] > 1:
    signal = signal[0:1, :]

# 2. VAD
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
boundaries = vad.get_speech_segments(wav_path)

# 3. speaker embedding (ECAPA-TDNN)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
MIN_DURATION = 1.0

speaker_embeddings = []
valid_segments = []

for seg in boundaries:
    start, end = seg[0].item(), seg[1].item()
    if end - start < MIN_DURATION:
        continue
    segment = signal[:, int(start * fs):int(end * fs)]
    embedding = classifier.encode_batch(segment).squeeze(0)
    speaker_embeddings.append(embedding)
    valid_segments.append((start, end))

# 4. clustering
embeddings_np = np.vstack([emb.cpu().numpy() for emb in speaker_embeddings])
n_speakers = 2
clustering = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
labels = clustering.fit_predict(embeddings_np)

# 5. initialize Whisper
asr_model = whisper.load_model("large")

# 6. transcribe each segment in memory
results = []

for i, ((start, end), label) in enumerate(zip(valid_segments, labels)):
    segment_audio = signal[:, int(start * fs):int(end * fs)].squeeze(0).numpy()
    segment_audio = segment_audio.astype(np.float32)
    if fs != 16000:
        segment_audio = torchaudio.functional.resample(
            torch.from_numpy(segment_audio),orig_freq=fs, new_freq=16000
            ).numpy()
    segment_tensor = torch.from_numpy(segment_audio)
    result = asr_model.transcribe(segment_tensor, language="zh")
    text = result["text"].strip()
    
    results.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "speaker": f"Speaker_{label}",
        "text": text
    })

# 7. output structured text
for r in results:
    print(f"[{r['start']:.2f}s - {r['end']:.2f}s] {r['speaker']}: {r['text']}")

