#!/bin/bash
# Dựng data/sam3_aug20k_v3 từ data/sam3_seg_v3 (mask đã sửa bằng SAM 3). Chạy từ thư mục gốc dự án.
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
SRC=data/sam3_seg_v3; OUT=data/sam3_aug20k_v3
OPTS="--sam3-only --source-root $SRC --paste-scale 0.6 1.1 --paste-counts 1 2 3 --min-inst-area 625 --workers 16"
B=utils/build_sam3_seg_v2/build_real.py

python $B $SRC train $OUT train      625      # ảnh thật làm phần đầu của train, phần còn lại là ảnh ghép
python $B $SRC valid $OUT valid_real 625
python $B $SRC test  $OUT test_real  625
N=$(ls $OUT/train/images | wc -l)
python utils/augment_mixed_10k.py $OPTS --name-prefix aug       --split train --output-dir $OUT/train --goal $((20000 - N)) --seed 42
python utils/augment_mixed_10k.py $OPTS --name-prefix aug_valid --split valid --output-dir $OUT/valid --goal 1000 --seed 101
python utils/augment_mixed_10k.py $OPTS --name-prefix aug_test  --split test  --output-dir $OUT/test  --goal 1000 --seed 101
sed "s#data/sam3_aug20k#$OUT#" data/sam3_aug20k/dataset.yaml > $OUT/dataset.yaml
echo "BUILD_DONE"
