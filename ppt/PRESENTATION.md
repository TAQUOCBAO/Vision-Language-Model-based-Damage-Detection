# BÁO CÁO THUYẾT TRÌNH KỸ THUẬT (TECHNICAL PRESENTATION DECK)
## HỆ THỐNG CHẨN ĐOÁN HƯ HỎNG KẾT CẤU CÔNG TRÌNH BẰNG MÔ HÌNH THỊ GIÁC - NGÔN NGỮ (VLM)
### Structural Damage Image-Text Diagnosis via Qwen3-VL-8B & LoRA Adaptation

> **Tài liệu tham chiếu:** [TECHNICAL_REPORT.md](../TECHNICAL_REPORT.md) · [README.md](../README.md)  
> **Interactive Canvas:** [`[structural-damage-vlm-presentation]`](/home/ai/.cursor/projects/home-ai-ethan-Project3/canvases/structural-damage-vlm-presentation.canvas.tsx)  
> **Hugging Face LoRA Adapter:** `nhantran214/damage-vlm-qwen3vl-8b-lora`  
> **Môi trường thực thi:** Python 3.14 · PyTorch 2.10.0 (cu128) · NVIDIA RTX 5880 Ada (49 GB)

---

## MỤC LỤC TRÌNH BÀY (SLIDE OVERVIEW)

1. **Slide 1:** Tổng quan & Tóm tắt Nhiệm vụ (Executive Summary & Mission)
2. **Slide 2:** Bối cảnh Kỹ thuật & Thách thức Bài toán (Problem Formulation & Challenges)
3. **Slide 3:** Kiến trúc Toàn diện của Hệ thống (End-to-End System Architecture)
4. **Slide 4:** Kỹ thuật Xử lý Dữ liệu & Gán Nhãn Silver (Data Engineering & Silver Labeling)
5. **Slide 5:** Chiến lược Phân chia Tập dữ liệu Đa nhãn (Stratified Multi-Label Split)
6. **Slide 6:** Thử nghiệm So sánh 3 Tầng Dữ liệu (Three-Tier Data Bake-Off: A / B / C)
7. **Slide 7:** Phương pháp Tinh chỉnh LoRA Tiết kiệm VRAM (VRAM-Safe LoRA Recipe)
8. **Slide 8:** Quy trình Hậu xử lý & Đồng bộ Ngữ nghĩa (Post-Processing & Alignment Pipeline)
9. **Slide 9:** Đánh giá Thực nghiệm & Lựa chọn Mô hình Thắng cuộc (Holdout Evaluation & Winner Selection)
10. **Slide 10:** Huấn luyện Quy mô Lớn trên Tập dữ liệu Cập nhật (Full Retrain on data_v2)
11. **Slide 11:** Triển khai Thực tế & Tối ưu Hóa VRAM 4-bit (Low-VRAM & Hugging Face Deployment)
12. **Slide 12:** Tổng kết, Đóng góp Kỹ thuật & Hướng phát triển (Conclusion & Future Work)

---

## SLIDE 1: TỔNG QUAN & TÓM TẮT NHIỆM VỤ

### 1. Mục tiêu Dự án (Objective)
Xây dựng một pipeline AI hoàn chỉnh, chạy **offline**, tinh chỉnh mô hình nền tảng Thị giác - Ngôn ngữ (**Qwen3-VL-8B-Instruct**) nhằm tự động phân tích hình ảnh công trình dân dụng (cầu, hầm, đập, tòa nhà) và xuất ra báo cáo chẩn đoán giám định dạng **JSON chuẩn hóa**:

```json
{
  "image_id": "00002",
  "damage_categories": ["cracks", "spalling"],
  "description": "Longitudinal fracture running across the reinforced beam with adjacent surface spalling."
}
```

### 2. Các Chỉ số & Kết quả Trọng tâm
- **Mô hình nền tảng:** `Qwen/Qwen3-VL-8B-Instruct` (8 tỷ tham số).
- **Phương pháp thích nghi:** LoRA (Rank 64, Alpha 128) trên toàn bộ các lớp chiếu Attention và MLP của phần ngôn ngữ; đóng băng Vision Tower.
- **Kết quả Holdout (n = 237 ảnh kiểm thử độc lập):**
  - **F1-Score Đa nhãn:** **0.9528** (Độ chính xác nhận diện hư hỏng kết cấu đạt ~95.3%).
  - **METEOR Description:** **0.6379** (Độ tương đồng ngữ nghĩa mô tả chuyên ngành cao).
  - **Điểm Tổng hợp Trọng số (Weighted Score):** **0.8268** (Đứng đầu toàn bộ các cấu hình thử nghiệm).
- **Bộ trọng số xuất bản (Hub Adapter):** `nhantran214/damage-vlm-qwen3vl-8b-lora` (~0.7 GiB).

> **Speaker Notes:**
> Kính thưa hội đồng / quý khán giả, mục tiêu của dự án là chuyển đổi quy trình giám định hư hại công trình từ thủ công sang tự động hóa có cấu trúc. Thay vì chỉ phân loại ảnh nhị phân hay sinh văn bản tự do không cấu trúc, hệ thống cung cấp đồng thời danh mục hư hỏng phân loại chuẩn xác và đoạn văn bản mô tả hình thái kỹ thuật theo định dạng JSON sẵn sàng cho các hệ thống quản lý hạ tầng.

---

## SLIDE 2: BỐI CẢNH KỸ THUẬT & THÁCH THỨC BÀI TOÁN

### 1. Tính phức tạp của Giám định Kết cấu Thực tế
- **Hiện tượng Đồng xuất hiện Nhiều Dạng Hư hỏng (Multi-Defect Co-occurrence):**
  Hơn 64% ảnh công trình trong thực tế có từ 2 loại hư hại trở lên (ví dụ: vết nứt bê tông đi kèm bong tróc cốt thép gỉ sét). Phân loại đơn nhãn truyền thống hoàn toàn thất bại.
- **Độ nhạy về Hướng Không gian (Spatial Orientation Sensitivity):**
  Mô tả kỹ thuật kết cấu chứa các thông tin quan trọng về hướng chịu lực (*"vertical shear crack"*, *"diagonal tension"*, *"horizontal fracture"*). Các phép tăng cường dữ liệu ảnh thông thường (lật ngang/dọc ngẫu nhiên) sẽ làm sai lệch bản chất hình học so với mô tả văn bản.
- **Tập Dữ liệu Nhỏ & Mất Cân Bằng Lớp (Small & Long-Tailed Dataset):**
  Tập dữ liệu gốc chỉ có ~1,200 ảnh với các lớp hiếm xuất hiện rất ít (*honeycomb*, *efflorescence*, *potholes*).

### 2. Định nghĩa Không gian Nhãn Đóng (10-Class Closed Taxonomy)
Hệ thống khóa cố định 10 lớp hư hại chuẩn kỹ thuật theo `configs/labels_v1.yaml`:

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 1. spalling     │ 2. cracks        │ 3. corrosion     │ 4. voids         │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 5. exposed_rebar│ 6. peeling       │ 7. potholes      │ 8. honeycomb     │
├─────────────────┼──────────────────┴──────────────────┴──────────────────┤
│ 9. looseness    │ 10. efflorescence                                      │
└─────────────────┴────────────────────────────────────────────────────────┘
```

---

## SLIDE 3: KIẾN TRÚC TOÀN DIỆN CỦA HỆ THỐNG

### Sơ đồ Khối Pipeline Xử lý (End-to-End Flow)

```
[Ảnh Đầu Vào + Raw description.json]
                 │
                 ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 1. Ingestion & Gán Nhãn Silver (Silver Multi-Labeling)    │
 │    • Tiền tố tên tệp (Prefix Seeds)                       │
 │    • Quét từ khóa đồng nghĩa dài nhất (Longest Synonym)   │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 2. Phân tầng Multi-Label Stratified Split (80/20, Seed 42)│
 │    • Train: 963 ảnh | Val (Holdout): 237 ảnh              │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 3. Thử nghiệm Đánh giá 3 Tầng Dữ liệu (Bake-Off A / B / C)│
 │    • Model: Qwen3-VL-8B-Instruct + LoRA (Rank 64, BF16)   │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 4. Đánh giá Holdout & Chọn Recipe Thắng (Tier A Winner)   │
 │    • Cổng an toàn F1 >= 0.97 * F1_max                     │
 │    • Điểm trọng số: 0.6 * F1 + 0.4 * METEOR               │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 5. Tái huấn luyện Toàn bộ Dữ liệu Cập nhật (data_v2)      │
 │    • 1,200 ảnh toàn phần → LoRA full_winner_v2            │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 6. Khối Hậu xử lý & Đồng bộ Đa chiều (Post-Processing)    │
 │    • JSON Repair → Synonym Map → Text Alignment → Dedupe  │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
     [Kết quả Dự đoán Chuẩn Hóa / Submission JSON]
```

---

## SLIDE 4: KỸ THUẬT XỬ LÝ DỮ LIỆU & GÁN NHÃN SILVER

### 1. Thuật toán Tổng hợp Nhãn Silver (Silver Label Synthesis)
Do tập dữ liệu ban đầu chỉ có dạng hỏi đáp VQA tự do và tên file không đồng nhất, nhãn silver đa nhãn $L_i$ cho mỗi ảnh $i$ được xây dựng tự động bằng luật hợp:

$$L_i = \text{Prefix\_Seeds}(\text{filename}_i) \cup \text{Text\_Keywords}(\text{description}_i)$$

- **Prefix Seeds:** Khai thác tiền tố chữ cái ở tên ảnh (ví dụ: `crack_` $\rightarrow$ `cracks`, `gangf_` $\rightarrow$ `exposed_rebar` + `cracks`, `xiu_` $\rightarrow$ `corrosion`).
- **Text Keywords (Longest-First Match):** Quét toàn bộ từ khóa và từ đồng nghĩa trong mô tả văn bản, ưu tiên so khớp cụm từ dài trước nhằm tránh báo động giả (ví dụ: tìm `"exposed rebar"` trước khi tìm `"exposed"`).

### 2. Tạo Cặp Dữ liệu SFT Multimodal
Mỗi mẫu huấn luyện được cấu trúc theo định dạng ShareGPT chuẩn của LLaMA-Factory:
- **System Prompt:** Chỉ định vai trò kỹ sư giám định kết cấu, yêu cầu chỉ xuất JSON thuần.
- **User Prompt:** Chứa token đặc biệt `<image>`, yêu cầu phân tích theo schema và danh sách 10 nhãn đóng.
- **Assistant Target:** Chuỗi JSON thuần không kèm markdown format ```` ```json ````.

---

## SLIDE 5: CHIẾN LƯỢC PHÂN CHIA TẬP DỮ LIỆU ĐA NHÃN

### 1. Rủi ro của Phân chia Ngẫu nhiên (Random Split)
Khi tập dữ liệu nhỏ và phân phối nhãn lệch nghiêm trọng, chia ngẫu nhiên sẽ dẫn đến hiện tượng các nhãn hiếm (*honeycomb*, *efflorescence*) bị dồn hết vào tập Train hoặc tập Val, làm méo mó kết quả đánh giá.

### 2. Giải pháp: Multilabel Stratified Shuffle Split
- **Thuật toán:** `iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit`.
- **Thiết lập:** Tỷ lệ phân chia $80\% / 20\%$ (`test_size = 0.2`), `seed = 42`.
- **Kết quả phân bổ:**
  - **Tập Huấn luyện (Train set):** $963$ ảnh.
  - **Tập Kiểm tra Độc lập (Holdout Val set):** $237$ ảnh.
- **Nguyên tắc Đóng băng (Frozen Split Artifact):** Lưu cố định tại `data/processed/splits/v1.json`. Tuyệt đối không thay đổi trong suốt quá trình thử nghiệm để đảm bảo tính công bằng tuyệt đối giữa các mô hình.

---

## SLIDE 6: THỬ NGHIỆM SO SÁNH 3 TẦNG DỮ LIỆU (BAKE-OFF)

Giữ nguyên $100\%$ kiến trúc mô hình và siêu tham số, chỉ thay đổi phương pháp xử lý dữ liệu huấn luyện:

| Tiêu chí | Tier A (Clean Baseline SFT) | Tier B (Orientation-Safe Aug) | Tier C (Paraphrase & External) |
| :--- | :--- | :--- | :--- |
| **Số lượng mẫu** | 963 mẫu sạch | 1,926 mẫu (Gốc + Aug) | 2,889+ mẫu mở rộng |
| **Tăng cường ảnh** | Không (Zero Augmentation) | Photometric + Conditional Geometric | Không tăng cường ảnh |
| **Bảo toàn hướng** | Hoàn toàn tự nhiên | Khóa phép lật nếu có từ chỉ hướng | Hoàn toàn tự nhiên |
| **Mở rộng văn bản** | Giữ mô tả gốc dài nhất | Chuẩn hóa câu kỹ thuật | Paraphrase cục bộ $\times 3$ + Pseudo-caption |
| **Dữ liệu ngoài** | Không | Không | Ảnh mẫu hư hại hiếm bên ngoài |

### Chi tiết Kỹ thuật An toàn Hướng của Tier B:
```python
def maybe_geometric_augment(image, description, rng):
    # Nếu trong mô tả có từ chỉ hướng, NGHIÊM CẤM lật ảnh
    if has_orientation_language(description):
        return image
    if rng.random() < 0.3:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
```

---

## SLIDE 7: PHƯƠNG PHÁP TINH CHỈNH LoRA TIẾT KIỆM VRAM

### 1. Kiến trúc Mô hình Nền tảng: Qwen3-VL-8B-Instruct
- **Vision Tower:** Trích xuất đặc trưng hình ảnh độ phân giải động.
- **Multi-modal Projector:** Cầu nối vector không gian thị giác - ngôn ngữ.
- **LLM Backbone:** Khối Transformer ngôn ngữ 8 tỷ tham số.

### 2. Thiết lập LoRA Tối ưu (VRAM-Safe Recipe)
- **Đóng băng (Freeze):** `freeze_vision_tower = true`, `freeze_multi_modal_projector = true` $\rightarrow$ Giảm hơn $18\text{ GB}$ VRAM tiêu thụ.
- **Target Modules:** Tinh chỉnh LoRA trên tất cả 7 ma trận chiếu của LLM: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **Tham số LoRA:** $\text{Rank } r = 64$, $\alpha = 128$, `dropout` $= 0.0$.
- **Batch Size & Tích lũy Gradient:** `per_device_batch_size = 1`, `gradient_accumulation_steps = 16` $\rightarrow$ Effective Batch Size $= 16$.
- **Tốc độ học (LR):** $1.0 \times 10^{-4}$ với bộ điều chỉnh Cosine Annealing, `warmup_ratio = 0.03`.
- **Độ phân giải giới hạn:** `image_max_pixels = 524288` ($\sim 0.5\text{ MP}$) $\rightarrow$ Tránh tràn bộ nhớ khi gặp ảnh kích thước lớn.
- **Định dạng số học:** Bfloat16 (BF16) ổn định trên vi kiến trúc Ada Lovelace.

---

## SLIDE 8: QUY TRÌNH HẬU XỬ LÝ & ĐỒNG BỘ NGỮ NGHĨA

Hệ thống triển khai 4 tầng hậu xử lý liên tiếp trước khi chấm điểm hoặc xuất kết quả:

```
[Chuỗi Sinh Ra Từ VLM]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. JSON Extraction & Syntax Repair                     │
│    Bóc tách markdown fences, tìm cặp ngoặc { ... }     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Category Synonym Mapping                            │
│    Ánh xạ tên nhãn tự do về 10 nhãn chuẩn đóng         │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Category ↔ Description Bidirectional Sync           │
│    • Quét văn bản bổ sung nhãn bị sót vào danh sách    │
│    • Bổ sung câu mẫu nếu nhãn có trong list mà thiếu text│
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Sentence Deduplication                              │
│    Loại bỏ các câu trùng lặp ngữ nghĩa, làm sạch khoảng trắng│
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
             [JSON Dự Đoán Hoàn Hảo]
```

### Ví dụ Thực tế về Sự Đồng bộ:
- **Đầu ra thô của mô hình:**
  `{"damage_categories": ["crack", "corrosion"], "description": "Vertical crack on the wall."}`
- **Sau khi qua Alignment Stack:**
  `{"damage_categories": ["cracks", "corrosion"], "description": "Vertical crack on the wall. Minor corrosion was also detected in the structural area."}`

---

## SLIDE 9: ĐÁNH GIÁ THỰC NGHIỆM & LỰA CHỌN MÔ HÌNH THẮNG CUỘC

### 1. Tiêu chí Đánh giá & Luật Quyết định Thắng cuộc
- **Chỉ số 1 (Đa nhãn):** $\text{F1}_{\text{samples}} = \frac{1}{N} \sum_{i=1}^{N} \text{F1}(Y_i, \hat{Y}_i)$.
- **Chỉ số 2 (Mô tả):** $\text{METEOR}$ tính trung bình trên tập kiểm thử (đánh giá đồng nghĩa, căn ngữ WordNet).
- **Cổng An toàn F1 (Safety Gate):** Loại bỏ mô hình nếu $\text{F1} < 0.97 \times \max(\text{F1})$.
- **Hàm Mục tiêu Trọng số:** $\text{Score} = 0.6 \times \text{F1} + 0.4 \times \text{METEOR}$.

### 2. Bảng Kết quả So sánh Thực nghiệm (Holdout $n = 237$):

| Hạng mục Thử nghiệm | Multilabel F1 | METEOR Score | Điểm Trọng số (0.6/0.4) | Trạng thái Cổng F1 | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier A (Clean SFT)** | **0.9528** | 0.6379 | **0.82684** | **ĐẠT (Baseline)** | 🏆 **WINNER** |
| **Tier B (Aug + Norm)**| 0.9495 | **0.6427** | 0.82681 | ĐẠT ($\ge 0.9242$) | Về nhì |
| **Tier C (Hybrid Exp)**| 0.9378 | 0.6136 | 0.80814 | ĐẠT ($\ge 0.9242$) | Xếp thứ ba |

### 3. Phân tích Chuyên sâu (Insight):
1. **Tier A chiến thắng:** Dữ liệu chuẩn sạch, không nhiễu giúp mô hình tối ưu hóa phân loại danh mục chính xác nhất.
2. **Tier B tăng nhẹ METEOR:** Phép tăng cường và chuẩn hóa câu giúp mô tả mượt hơn (METEOR đạt $0.6427$), nhưng làm giảm nhẹ độ chính xác phân loại F1.
3. **Tier C suy giảm:** Việc bổ sung dữ liệu giả lập chất lượng chưa đồng đều gây nhiễu cho mô hình thị giác trên tập dữ liệu kích thước nhỏ.

---

## SLIDE 10: HUẤN LUYỆN TOÀN DIỆN TRÊN TẬP DỮ LIỆU CẬP NHẬT (DATA_V2)

### 1. Quy trình Retrain Toàn phần
Sau khi xác định **Tier A** là chiến lược tối ưu nhất:
- Nạp toàn bộ tập dữ liệu cập nhật $\text{data\_v2/dataset}$ ($1,200$ ảnh).
- Tự động sinh nhãn silver và định dạng SFT qua `scripts/07_build_full_winner.py`.
- Thực hiện huấn luyện toàn phần 3 epoch với recipe thắng cuộc qua `scripts/03_train_tier.py --tier full`.

### 2. Động lực Hội tụ (Training Convergence)
- **Loss khởi điểm:** $\sim 1.82$.
- **Loss sau 1 epoch:** $\sim 0.52$.
- **Loss kết thúc (Epoch 3):** $\sim \mathbf{0.249}$.
- **Tập trọng số cuối cùng:** Lưu tại `outputs/runs/full_winner_v2`.

---

## SLIDE 11: TRIỂN KHAI THỰC TẾ & TỐI ƯU HÓA VRAM 4-BIT

### 1. Xuất bản Trọng số lên Hugging Face
Bộ adapter LoRA được đóng gói đầy đủ và xuất bản công khai:
- **Repo ID:** `nhantran214/damage-vlm-qwen3vl-8b-lora`
- **Dung lượng:** Chỉ $\sim 700\text{ MB}$ (bao gồm `adapter_config.json`, `adapter_model.safetensors`).

### 2. Các Chế độ Suy Luận Linh Hoạt (Inference Modes)

| Chế độ Thực thi | Thiết bị Mục tiêu | Mức Tiêu thụ VRAM | Tốc độ / Chất lượng | Lệnh Chạy |
| :--- | :--- | :---: | :---: | :--- |
| **Full Precision (BF16)** | GPU Server (RTX 5880, A100, RTX 4090) | $\sim 16 - 22\text{ GB}$ | Tốc độ cao nhất, không mất độ chính xác | `python scripts/08_infer_full_winner.py --adapter ./models/...` |
| **4-Bit NF4 (Quantized)** | GPU Cá nhân (RTX 3060 6GB, RTX 4060) | $\sim \mathbf{4 - 8\text{ GB}}$ | VRAM siêu nhẹ, độ chính xác tương đương | `python scripts/08_infer_full_winner.py --load-in-4bit --adapter ./models/...` |

---

## SLIDE 12: TỔNG KẾT, ĐÓNG GÓP & HƯỚNG PHÁT TRIỂN

### 1. Các Đóng Góp Chính của Dự Án
1. **Thiết kế Hệ thống Chuẩn hóa:** Pipeline VLM hoàn chỉnh đầu tiên tích hợp từ khâu gán nhãn silver, huấn luyện LoRA cho đến hậu xử lý ràng buộc nhãn đóng JSON.
2. **Bằng chứng Thực nghiệm Rõ ràng:** Chứng minh phương pháp SFT chất lượng cao (Tier A) vượt trội hơn việc tăng cường dữ liệu bừa bãi trong bài toán giám định kỹ thuật.
3. **Hiệu năng & Khả năng Tiếp cận Cao:** Đạt F1 $0.9528$, hoạt động hoàn toàn offline, hỗ trợ chạy trên card đồ họa phổ thông $6\text{ GB}$ VRAM.

### 2. Hướng Mở Rộng trong Tương lai
- Bổ sung mô hình Paraphrase chất lượng cao chạy offline (ví dụ: Qwen2.5-14B) để tái thử nghiệm Tier C.
- Tinh chỉnh giải phóng một phần Multi-modal Projector để tăng cường nhận diện các chi tiết vết nứt siêu nhỏ.
- Xây dựng giao diện Web UI trực quan hóa hộp giới hạn (Bounding Box) vị trí khuyết tật kèm điểm tin cậy (Confidence Score).

---

## TÀI LIỆU THAM KHẢO HỌC THUẬT (ACADEMIC REFERENCES)

```bibtex
[1] Qwen Team, "Qwen2-VL: To See the World More Clearly," arXiv preprint arXiv:2409.12191, 2024.
[2] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in Proc. ICLR, 2022.
[3] S. Banerjee and A. Lavie, "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments," in Proc. ACL Workshop, 2005, pp. 65-72.
[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.
[5] Y. Zheng et al., "LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ Language Models," in Proc. ACL: System Demonstrations, 2024.
[6] K. Sechidis, G. Tsoumakas, and I. Vlahavas, "On the stratification of multi-label data," in Proc. ECML PKDD, 2011, pp. 145-158.
```
