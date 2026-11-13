# multitask_vmunet_arch — biến thể kiến trúc của VM-UNet + Spatial

Kế hoạch: `Reports/Meeting_2026-11-13/PLAN.md`. Baseline: `multitask_vmunet_spatialonly` (chính thức: bản train v3 + ảnh dày, `spatialonly_mix`).

**Quy tắc (quyết định 30/10):** giữ nguyên loss spatial embedding, data, tối ưu, cách gom cụm; mỗi lần chỉ đổi MỘT thứ về kiến trúc.
Mục tiêu: con bị che từ 3 mảnh trở lên (baseline nhận đúng 73.5%, bị tách đôi 18.7%) mà không tăng gộp nhầm (4.7/100) và không tụt `e2e_f1` (0.903).

## Các `--arch`

| `--arch` | Mã | Thay đổi duy nhất so với baseline | Tham số (đo bằng `test_arch.py`) |
|---|---|---|---|
| `base` | — | không đổi (kiểm tra tái tạo baseline; đo nhiễu giữa các seed) | 27.43 M |
| `branch` | 2a | tách decoder: chung tới `layers_up[1]`, rồi 2 tầng decoder cuối + `Final_PatchExpand` nhân đôi. Nhánh chính ra thân/bệnh (2 kênh), nhánh embedding ra offset/σ/seed (5 kênh) | 28.62 M (+1.19 M) |
| `deep3` | 2b | encoder stage 3 sâu hơn: `depths` [2,2,2,2] → [2,2,4,2] | 29.82 M (+2.39 M) |
| `x0skip` | 2c | cộng thêm skip 64×64 (`skip_list[0]`, baseline bỏ phí) vào đầu ra tầng decoder cuối | 27.43 M (+0.00 M) |
| `attn` | 2d | thêm một khối self-attention toàn cục (6 đầu, MLP ×4, mã vị trí sin/cos, khởi tạo = phép đồng nhất) sau `layers_up[1]` (16×16×384) | 29.20 M (+1.77 M) |

Kết quả `attn` (một seed, 13/11): nhận đúng con ≥3 mảnh 71.7% (baseline 73.7 ± 0.3%), ảnh dày 512² 0.865 (baseline 0.884 ± 0.005): không cải thiện. Vùng nhìn hiệu dụng của baseline đã phủ 99% cặp mảnh cùng con (`Reports/Meeting_2026-11-13/results/vmunet/analysis/receptive_field.md`).

`branch`, `attn` và `x0skip` được cài bằng lớp con của `SpatialOnlyVMUNet` (`model.py`), không sửa `models/vmunet`. Đầu ra luôn `[B, 7, H, W]` nên loss, `split`, suy luận và chấm điểm dùng lại nguyên.
Hai nhánh của `branch` khởi tạo bằng sao chép (`deepcopy`) hai tầng decoder cuối đã khởi tạo, nên bắt đầu giống nhau và tách ra khi train.

## File

| File | Việc |
|---|---|
| `config.py` | `ArchConfig` = `SpatialOnlyConfig` + `arch` (`deep3` đổi `depths`) |
| `model.py` | `BranchVMUNet`, `X0SkipVMUNet`, `AttnVMUNet` (+ `GlobalAttnBlock`), `build_model(cfg)`, `count_params` |
| `train.py` | bản sao `spatialonly/train.py` + `--arch`, `--seed` (loss, optimizer, chọn `best.pth` bằng `e2e_f1` giống hệt) |
| `evaluate.py`, `inference.py` | nạp model theo `arch` trong `config.yaml` của run; `run_eval`, `spatial_instances`, `predict` dùng lại từ spatialonly |
| `test_arch.py` | kiểm tra không cần data: kích thước đầu ra, số tham số, khởi tạo offset/σ, gradient chạm mọi tham số |

## Lệnh

```bash
source .venv/bin/activate
python -m experiments.multitask_vmunet_arch.test_arch --gpu 1                          # kiểm tra nhanh (không cần data)
python -m experiments.multitask_vmunet_arch.train --smoke-test --gpu 1 --arch branch --train-mix v3:1 dense_v2:8
# train thật (trong tmux), cùng data với baseline spatialonly_mix:
python -m experiments.multitask_vmunet_arch.train --gpu 1 --arch branch --seed 42 --train-mix v3:1 dense_v2:8 \
    --out-dir Reports/Meeting_2026-11-13/checkpoints/multitask_vmunet_arch_branch
```

Mỗi run phải có `--train-mix v3:1 dense_v2:8` (như baseline); nếu quên thì chạy trên v3 thuần và không so sánh được với `spatialonly_mix`.
`--resume <thư mục run>` yêu cầu truyền lại `--arch` và `--train-mix` đúng như run cũ (khác `arch` thì dừng).

## Chấm điểm và đo nhiễu

- Ngưỡng `seed_thr` chọn trên `v3:valid` cho TỪNG run (quy tắc của `eval_mix_models.py`), rồi chấm các tập khác một lần. Baseline dùng 0.9.
- Nhóm con bị che từ 3 mảnh chỉ có 166 con (sai số khoảng ±3.4 điểm phần trăm): chạy `--arch base` với 2 seed khác để biết nhiễu thật của baseline, rồi mới so các biến thể.
- Mọi run: kiểm tra 2 ảnh `data/test` (báo cả xác suất kênh thân).

## Kết quả (14/11; `v3:test`, một seed mỗi biến thể; số đầy đủ ở `Reports/Meeting_2026-11-13/results/vmunet/analysis/arch_comparison.md`)

Baseline 3 seed (`spatialonly_mix`, `arch_base_s1`, `arch_base_s2`): `e2e_f1` `test` 0.903 ± 0.003, nhận đúng con ≥3 mảnh 73.7 ± 0.3%, `test_real` 0.983 ± 0.004, ảnh dày 512² 0.884 ± 0.005.

| `--arch` | `e2e_f1` test | ≥3 mảnh nhận đúng | `e2e_f1` test_real | `e2e_f1` ảnh dày 512² |
|---|---|---|---|---|
| `branch` | 0.915 | 77.1% | 0.982 | 0.879 |
| `deep3` | 0.914 | 75.9% | 0.982 | 0.880 |
| `attn` | 0.908 | 71.7% | 0.986 | 0.865 |
| `x0skip` | chưa chạy | | | |

Đọc: hai biến thể đầu hơn baseline trên ảnh ghép (vượt nhiễu ±0.003) nhưng không hơn ở ảnh thật và ảnh dày; `attn` không giúp nhóm ≥3 mảnh và kém hơn ở ảnh dày. Vùng nhìn hiệu dụng của baseline đã phủ 99% cặp mảnh cùng con (`receptive_field.md`), nên thiếu ngữ cảnh toàn cục không phải nguyên nhân chính. Mỗi biến thể một seed.
