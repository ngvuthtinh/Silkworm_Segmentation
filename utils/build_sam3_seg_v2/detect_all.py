"""Chạy SAM 3 (chỉ gợi ý chữ) trên toàn bộ ảnh yolo_bbox, lưu mọi phát hiện để dựng vùng 'bỏ qua'."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "utils")
import sam3_label_from_yolo as S
from PIL import Image
out_root = Path(sys.argv[1])
model, proc = S.load_sam3_model("models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt",
                                "models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", "cuda", 0)
proc.confidence_threshold = 0.3
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x
for sp in ["valid", "test", "train"]:
    d = out_root / sp; d.mkdir(parents=True, exist_ok=True)
    ims = sorted(Path(f"data/yolo_bbox/{sp}/images").glob("*.jpg"))
    for p in tqdm(ims, desc=sp, mininterval=30):
        dst = d / f"{p.stem}.npz"
        if dst.exists():
            continue
        pil = Image.open(p).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = proc.set_image(pil)
            o = proc.set_text_prompt("silkworm", st)
        if len(o.get("masks", [])):
            m = o["masks"].float().cpu().numpy() > 0.5
            m = m.reshape(m.shape[0], m.shape[-2], m.shape[-1])
            np.savez_compressed(dst, boxes=o["boxes"].float().cpu().numpy(), scores=o["scores"].float().cpu().numpy(),
                                masks=np.packbits(m, axis=-1), shape=np.array(m.shape))
        else:
            np.savez_compressed(dst, boxes=np.zeros((0, 4)), scores=np.zeros(0), masks=np.zeros((0,), np.uint8), shape=np.array([0, 0, 0]))
    print(f"{sp}: xong {len(list(d.glob('*.npz')))}/{len(ims)}", flush=True)
