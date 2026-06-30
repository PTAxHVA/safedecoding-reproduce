# SafeDecoding — reproduce (nhóm In-inference defense)

Reproduce method **SafeDecoding** (ACL 2024, Xu et al.) — phòng thủ jailbreak ở **tầng decoding**:
lúc model sinh chữ, dùng một "expert an toàn" (LoRA nhỏ) để **kéo xác suất các token từ chối lên**
và **dìm các token độc hại xuống**, nên model quay về câu trả lời an toàn mà **không train lại** model gốc.

Repo này = clone từ repo gốc `uw-nsl/SafeDecoding` + chỉnh lại để **pull về → tạo venv → chạy được ngay**
trên một bộ **sample data nhỏ** (rồi anh cắm data thật của anh vào). README gốc của tác giả giữ ở **`README-goc.md`**.

---

## 1. Eval dùng data nào (đọc cái này trước)

| Mục đích | Bộ data | Ở đâu |
|---|---|---|
| **Sample chạy ngay** (đo DSR) | 20 câu harmful lấy từ AdvBench | **kèm repo**: `datasets/custom_prompts.json` |
| AdvBench-50 (harmful) | 50 câu | kèm repo: `datasets/harmful_behaviors_custom.json` |
| Attack đầy đủ của paper | GCG / AutoDAN / PAIR / DeepInception | **tự tải khi chạy** từ HF `flydust/SafeDecoding-Attackers` |
| Over-refusal / FRR (Utility) | XSTest | **KHÔNG kèm** — dùng bộ benchmark chung của anh |

**Cách chấm điểm:** dùng **DictJudge** = so chuỗi từ chối ("I'm sorry", "I cannot", "As an AI"…) trong câu trả lời.
- **DSR (Defense Success Rate)** = % prompt mà model **từ chối** = `defense_success / tổng`.
- **ASR** (script in ra) = `100 − DSR`.
- DictJudge chạy **offline, KHÔNG cần API**. (Harmful Score 1–5 thì mới cần GPT-4 / Gemini — optional, xem mục 5.)

---

## 2. Cài đặt (cần GPU **CUDA**)

> ⚠️ Code gốc gọi `.cuda()` nên **bắt buộc GPU NVIDIA (CUDA)**. Model 7B fp16 ≈ **16GB VRAM**. Máy Mac / CPU **không chạy được**.

```bash
# B1. Tạo + KÍCH HOẠT môi trường ảo py3.10 TRƯỚC (để mọi thứ cài vào đúng env, không lẫn vào base):
conda create -n SafeDecoding python=3.10 -y
conda activate SafeDecoding

# B2. Cài PyTorch khớp CUDA của máy (ví dụ CUDA 12.1; đổi cu121 cho đúng driver của anh):
#     (bản tham chiếu mình test: torch 2.1.2)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# B3. Cài phần còn lại:
pip install -r requirements.txt

# B4. (optional) login HuggingFace nếu muốn dùng Llama2 (gated) hoặc bộ attack HF.
#     Vicuna thì KHÔNG cần login.
huggingface-cli login
```

---

## 3. Chạy thử sample (offline — không API, không tải data lớn)

> 💡 **Không có GPU CUDA?** Mở **`run_on_colab.ipynb`** bằng Google Colab (Runtime → T4 GPU → Run all) là chạy được sample + ra số DSR, không cần máy có GPU.

```bash
cd exp
# Có phòng thủ SafeDecoding, chấm DSR offline trên 20 prompt sample:
python defense.py --model_name vicuna --attacker custom --defender SafeDecoding --disable_GPT_judge

# So sánh: TẮT phòng thủ (để thấy DSR tụt xuống):
python defense.py --model_name vicuna --attacker custom --defense_off --disable_GPT_judge
```

- Lần đầu sẽ tự tải model `lmsys/vicuna-7b-v1.5` (~13GB) từ HuggingFace.
- Kết quả + log lưu ở `exp_outputs/<defender>_<model>_<attacker>_<n>_<time>/` ; cuối log in ra dòng `ASR: xx.xx%`.
- 20 prompt nên chạy nhanh (vài phút khi model đã tải xong).

---

## 4. Cắm DATA THẬT của anh vào

- **Cách nhanh (khuyên dùng):** thay file `datasets/custom_prompts.json` bằng data của anh, mỗi dòng dạng:
  ```json
  [ { "id": 0, "goal": "<câu hại gốc>", "prompt": "<prompt thực sự đưa vào model, có thể đã bọc attack>" } ]
  ```
  rồi chạy lại lệnh ở mục 3 với `--attacker custom`. (`goal` = nhãn để chấm; `prompt` = cái model đọc.)
- **Dùng bộ attack của paper:** `--attacker GCG` (hoặc `AutoDAN`, `PAIR`, `DeepInception`) → tự tải từ HF.
- **Đổi model:** `--model_name vicuna|llama2|guanaco|falcon|dolphin` — expert LoRA tương ứng đã có sẵn trong `lora_modules/` (khỏi train).

---

## 5. Đo theo benchmark chung (Security / Efficiency / Utility)

| Trục | Đo bằng | Lệnh / ghi chú |
|---|---|---|
| **Security** = DSR | DictJudge (offline) | mục 3, đọc dòng `ASR` → `DSR = 100 − ASR` |
| **Utility** = over-refusal (FRR) | cho input XSTest *benign* xem có bị từ chối nhầm | `--attacker custom` với file XSTest benign, xem tỉ lệ "bị từ chối" |
| **Utility** = chất lượng | Just-Eval / MT-Bench | `--attacker Just-Eval --GPT_API <key>` ; MT-Bench xem `mt_bench/README.md` |
| **Efficiency** = Cost | GPU-giờ inference | mỗi item có `time_cost` (giây) trong file json output |
| Harmful Score 1–5 (optional) | GPT-4 (hoặc Gemini) | bỏ `--disable_GPT_judge`, thêm `--GPT_API <key>` **và `--multi_processing 1`** |

> ⚠️ **Khi dùng LLM-judge (GPT-4/Gemini) nhớ thêm `--multi_processing 1`.** Path chấm song song mặc định (`multi_processing=20`) của repo gốc **KHÔNG truyền `--GPT_API` xuống các worker** (bug gốc ở `safe_eval.py` hàm `single_resp`) → judge sẽ lỗi nếu không có sẵn env `OPENAI_API_KEY`. Đặt `--multi_processing 1` thì key được truyền đúng. *(Việc đo DSR ở mục 3 dùng `--disable_GPT_judge` nên KHÔNG dính lỗi này.)*

> Muốn dùng **Gemini** thay GPT-4 làm judge: sửa class `GPT` trong `utils/model.py` (lib `google-generativeai` đã có trong requirements); `GPTJudge` đã hỗ trợ policy `Google` sẵn.

---

## 6. Đã sửa gì so với repo gốc (để anh diff nhanh)

- **`requirements.txt`**: pin lại version cho khớp `transformers==4.28.1` — `datasets==2.14.6`, `huggingface_hub==0.16.4`,
  `accelerate==0.25.0`, `numpy==1.26.4`, `tokenizers==0.13.3`. (File gốc để trống `datasets`/`accelerate` → pip kéo bản mới
  nhất, lệch version, vỡ `load_dataset`.) PyTorch để anh tự cài đúng CUDA (xem mục 2).
- **`datasets/custom_prompts.json`**: thay bằng **20 câu sample** (id/goal/prompt) để chạy được ngay; file mẫu gốc chỉ có 1 câu.
- **`exp/defense.py`**: sửa 1 dòng `prompt["goal"]` → `prompt.get("goal", user_prompt)` để khỏi **KeyError** khi data
  không có field `goal` (bug nhỏ của path `custom`/`AdvBench`); thêm vài comment đầu file. **Không đụng** thuật toán.
- **`README-goc.md`**: README gốc của tác giả (giữ nguyên).
- Còn lại (thuật toán `utils/safe_decoding.py`, expert LoRA `lora_modules/`, peft bundle `peft/`…) **giữ nguyên**.

---

## 7. Lưu ý nhanh

- **Expert KHÔNG phải model thứ 2:** nó là chính model 7B gốc + một **LoRA nhỏ** (rank 16) gắn qua `peft`
  (`adapter_names=['base','expert']`), nên VRAM chỉ tốn ~1 model 7B. Đây là lý do phải dùng **peft bản bundle** trong repo.
- Tham số mặc định đúng như paper: `--alpha 3 --first_m 2 --top_k 10 --num_common_tokens 5`.
- Repo nặng ~160MB vì có sẵn 5 expert LoRA (`lora_modules/`, mỗi cái ~32MB) — để anh khỏi phải train.
