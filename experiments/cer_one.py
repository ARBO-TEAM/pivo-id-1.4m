"""CTC read-back CER for one WAV (debug/compare helper)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import ctc_align  # noqa: E402

ID_CTC = ROOT / "models/hf/hub/models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese/snapshots/e5e699fa5aa5bdce999276a90a29589587b58ac9"
EN_CTC = ROOT / "models/hf/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def read_wav(path):
    import wave
    with wave.open(str(path)) as w:
        n, r, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        d = np.frombuffer(w.readframes(n), dtype="<i2" if sw == 2 else "<i4").astype(np.float32)
        if ch > 1:
            d = d.reshape(-1, ch).mean(1)
        d /= 32768.0 if sw == 2 else 2147483648.0
    return d, r


def main(files, text, lang):
    proc, model = ctc_align.load_ctc(EN_CTC if lang == "en" else ID_CTC)
    want = "".join(c.lower() for c in text if c.isalnum())
    out = []
    for p in files:
        pcm, rate = read_wav(p)
        if rate != 16000:
            pcm = ctc_align.resample_to_16k((np.clip(pcm, -1, 1) * 32767).astype(np.int16), rate)
        else:
            pcm = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
        runs, _ = ctc_align.ctc_runs(proc, model, pcm.astype(np.int16))
        heard = "".join(ch.lower() for ch, _, _ in runs if ch.isalnum())
        prev = list(range(len(want) + 1))
        for i, cx in enumerate(heard, 1):
            cur = [i]
            for j, cy in enumerate(want, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
            prev = cur
        out.append({"file": str(p), "cer": round(prev[-1] / max(1, len(want)), 4), "heard": heard})
    for row in out:
        print(f"{row['cer']:.4f} {row['file']}\n      heard: {row['heard']}")


if __name__ == "__main__":
    lang, text = sys.argv[1], sys.argv[2]
    lines = []
    main_files = sys.argv[3:]
    # write results to a file too (terminal stdout gets filtered by the rtk hook)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(main_files, text, lang)
    Path("experiments/eval/cer_one_out.txt").write_text(buf.getvalue(), encoding="utf-8")
