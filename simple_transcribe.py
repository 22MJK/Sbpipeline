import torchaudio
import whisper
from speechbrain.inference import VAD
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os

# 1. 加载音频
wav_path = "/opt/data/majikui/audios/smalltest.wav"
signal, fs = torchaudio.load(wav_path)

# 2. 语音活动检测（VAD）
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmp_vad")
boundaries = vad.get_speech_segments(wav_path)

# 3. 说话人嵌入（ECAPA-TDNN）
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

# 4. 聚类
embeddings_np = np.vstack([emb.cpu().numpy() for emb in speaker_embeddings])
n_speakers = 2
clustering = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
labels = clustering.fit_predict(embeddings_np)

# 5. 初始化 Whisper
asr_model = whisper.load_model("large")  # 可选：tiny, base, small, medium, large

# 6. 对每个 segment 进行转写并关联说话人
results = []
os.makedirs("segments", exist_ok=True)

for i, ((start, end), label) in enumerate(zip(valid_segments, labels)):
    segment_audio = signal[:, int(start * fs):int(end * fs)]
    seg_path = f"segments/seg_{i:03d}_spk{label}.wav"
    torchaudio.save(seg_path, segment_audio, fs)
    
    transcription = asr_model.transcribe(seg_path, language="zh")  # 可指定语言
    results.append({
        "start": round(start, 2),
        "end": round(end, 2),
        "speaker": f"Speaker_{label}",
        "text": transcription["text"].strip()
    })

# 7. 输出结构化文本
for r in results:
    print(f"[{r['start']:.2f}s - {r['end']:.2f}s] {r['speaker']}: {r['text']}")
