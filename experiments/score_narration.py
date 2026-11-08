"""Score the hyperframes narration assets against their intended text (CTC read-back)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_id"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

import ctc_align  # noqa: E402
from cer_one import read_wav  # noqa: E402

ID_CTC = ROOT / "models/hf/hub/models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese/snapshots/e5e699fa5aa5bdce999276a90a29589587b58ac9"
EN_CTC = ROOT / "models/hf/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

from render_hyperframes_voices import CLIPS  # noqa: E402


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(directory: Path) -> int:
    procs = {"id": ctc_align.load_ctc(ID_CTC), "en": ctc_align.load_ctc(EN_CTC)}
    out = {}
    for name, (lang, _voice, text) in CLIPS.items():
        path = directory / f"{name}.wav"
        if not path.exists():
            continue
        pcm, rate = read_wav(path)
        p16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
        if rate != 16000:
            p16 = ctc_align.resample_to_16k(p16, rate)
        runs, _ = ctc_align.ctc_runs(procs[lang][0], procs[lang][1], p16)
        heard = "".join(ch.lower() for ch, _, _ in runs if ch.isalnum())
        want = "".join(c.lower() for c in text if c.isalnum())
        cer = ed(heard, want) / max(1, len(want))
        out[name] = {"cer": round(cer, 4), "seconds": round(len(pcm) / rate, 2), "heard": heard}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
