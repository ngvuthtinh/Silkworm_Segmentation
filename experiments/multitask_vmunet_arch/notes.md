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

`branch` và `x0skip` được cài bằng lớp con của `SpatialOnlyVMUNet` (`model.py`), không sửa `models/vmunet`. Đầu ra luôn `[B, 7, H, W]` nên loss, `split`, suy luận và chấm điểm dùng lại nguyên.
Hai nhánh của `branch` khởi tạo bằng sao chép (`deepcopy`) hai tầng decoder cuối đã khởi tạo, nên bắt đầu giống nhau và tách ra khi train.

## File

| File | Việc |
|---|---|
| `config.py` | `ArchConfig` = `SpatialOnlyConfig` + `arch` (`deep3` đổi `depths`) |
| `model.py` | `BranchVMUNet`, `X0SkipVMUNet`, `build_model(cfg)`, `count_params` |
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
