---
name: meeting-preparation
description: Prepare the content file for the student's next advisor meeting ("báo cáo với cô", every ~2 weeks) in this silkworm segmentation repo — Reports/Meeting_<date>/meeting_preparation.md, in Vietnamese, summarizing what was done since the last meeting, results, analysis, next-step proposals and questions for the advisor. Use it whenever the user mentions preparing a meeting, a report or slides for the advisor/cô, "tuần này báo cáo gì", "chuẩn bị họp", "soạn báo cáo", reviewing what was done since the last meeting, or checking their presentation outline — even if they don't say "skill" or name the file.
---

# Meeting preparation for the advisor

The student meets their advisor (cô) about every two weeks and presents slides. Before each meeting they need one Markdown file holding the *content* of that talk: what was asked last time, what was done, the numbers, why things worked or failed, what to do next, and what to ask. The slides are built from this file, so it must be accurate, easy to say out loud, and honest about what is not done.

Reference examples (read both before writing; match their tone and level of detail):
- `Reports/Meeting_2026-10-30/meeting_preparation.md` — method/experiment meeting (papers → architecture + loss → results → analysis → next limitation).
- `Reports/Meeting_2026-10-17/meeting_preparation.md` — analysis meeting (data fixes, baseline, limitations).

## 1. Find the dates and what was asked

- Previous minutes: newest file in `meetings/` (`YYYY-MM-DD.md`). Its first line gives `Họp tiếp theo` = the date of the meeting being prepared. If no next date is written, ask.
- Output: `Reports/Meeting_<next date>/meeting_preparation.md`. File and folder names are English; the content is Vietnamese (the advisor reads Vietnamese).
- From the minutes take the **Decision Summary** and **Action Items**. They are the backbone: the advisor will check each one.

## 2. Collect evidence of what was done

Look only at work dated after the previous meeting:
- `git log --since=<previous date>` and `git status` (much work may be uncommitted).
- `Reports/Meeting_<next date>/` — `results/README.md` (model → folder → run), `results/<family>/scores/<model>/eval_*.txt` + `run.yaml`, `results/<family>/images*/` (one picture per test image, all models of the family side by side), `results/comparison/mix_models/` (all models + `summary.md`), `literature/`.
- `experiments/*/notes.md` changed since that date (method, loss weights, risks).
- `Reports/experiment_results.xlsx` (run `python utils/export_results_xlsx.py` first so it is current).
- CLAUDE.md §3.5 (research scope and decisions).

Then build a table *action item → status (done / partial / not done) → evidence (file)*. Show this table to the user before writing; it tells both of you where the gaps are.

## 3. Ask the student for what the repo cannot show

The repo does not know which papers were read outside it, ideas being explored, the story they want to tell, or things they said loosely at the last meeting. Ask briefly for:
- their outline for the talk (if they have one, evaluate it: logical order, missing pieces the advisor will ask about, claims the evidence does not support);
- work done outside the repo (papers, other models explored);
- anything in the minutes they want corrected.

Never write "đã làm" for something there is no evidence of and the user has not confirmed. Leave a clearly marked placeholder (`*(Điền nếu có …)*`) instead — an invented accomplishment is worse than a gap, because the advisor will ask about it.

## 4. Write the report

Use this structure (drop sections that do not apply; keep the numbering consistent):

```
# Báo cáo họp DD/MM/YYYY: <chủ đề chính>
> **Mục tiêu 2 tuần qua** (theo biên bản DD/MM): … (+ một câu: tập trung vào gì, vì sao)

## 0. Tóm tắt trong 1 phút        — 3–4 gạch đầu dòng: đã làm / kết quả chính (con số) / điều rút ra / chưa giải được
## 1. Nhắc lại vấn đề             — vấn đề đang giải, một ảnh ví dụ, con số cho thấy nó nghiêm trọng
## 2. Tìm hiểu                    — papers/nguồn đã đọc, cách phân loại, vì sao chọn giải pháp này (và vì sao không chọn cái khác)
## 3–4. Phương pháp (mỗi paper/ý tưởng một mục)
      — trích dẫn (tác giả, năm, tên, arXiv/DOI); "Cách hoạt động" bằng lời đơn giản;
        "Áp dụng vào model": bảng so với baseline: Đầu ra | Loss | Hậu xử lý; tham số chính và chúng được chọn thế nào
## 5. Kết quả                     — bảng Mã | Đổi gì | test (ghép) | test_real (thật) | tìm đúng con bệnh; bảng phân tích riêng cho vấn đề; 1–3 ảnh
## 6. Phân tích                   — vì sao thành công / vì sao cái này tốt hơn cái kia / vẫn còn fail ở đâu
## 7. Limitation tiếp theo + đề xuất — bằng chứng, giả thuyết, bảng hướng làm (Hướng | Chi phí | Vì sao) và thứ tự đề xuất
## 8. Tìm hiểu thêm (tuỳ chọn)
## 9. Hạn chế (nói thẳng với cô)
## 10. Câu hỏi cho cô             — các quyết định cần cô chốt, đánh số
### Phụ lục: tìm file ở đâu      — bảng Nội dung | Đường dẫn (tương đối với thư mục meeting)
```

Writing rules, and why:
- **Numbers must be copied from source files**, not from memory or chat: `eval_*.txt`, `comparison_table.md`, `train_log.csv`. After writing, re-open each source and check every number in the report. Use 3 decimals for scores; `e2e_f1` is the main score.
- **Every claim of cause is labelled.** Measured facts are stated plainly; explanations without an experiment behind them are marked *(giả thuyết)*. The advisor needs to know which is which to decide what to test next.
- **State the comparison conditions**: same data version (`sam3_aug20k_v3` …), thresholds chosen on `valid`, `test` run once, one seed per model (differences below ~0.01 are noise). Note when a run used different data.
- **One change per experiment** is the lab rule (CLAUDE.md §2.5). If an experiment changed architecture and loss together, say so in "Hạn chế".
- **Always include the two dense photos in `data/test`** for every model (CLAUDE.md §3.3) — they are the standing check for Limitation 2.
- **Plain Vietnamese, short sentences**, technical terms in English only where the student would say them on a slide (loss, embedding, watershed). Explain what each metric means the first time it appears in a table.
- **Image links** are relative to the report file and must exist (check with `ls`). Prefer images whose panels carry only names (Ảnh gốc | Ground truth | model).
- Keep it about the length of the reference reports (~150–250 lines). It is a script for 10–15 minutes of slides, not a thesis chapter.

## 5. Finish

- Verify: all numbers re-checked, all paths exist, every action item from the minutes appears somewhere (done, partial, or in "Hạn chế").
- Reply to the user with: the file path, a short evaluation of their outline (if given), the gaps that are theirs to fill (placeholders, Excel manual columns `Mã`/`Đổi gì`/`Ghi chú`), and what you could still compute before the meeting. Start the reply with a one-line "Written for:" naming the audience (the student preparing slides for the advisor).
- Do not commit; the user commits after review.
