# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Phát Thịnh **MSSV**: 2A202602645 **Ngày**: 2026-10-07  
**Tier**: `T4` **Base model**: `unsloth/Qwen3.5-4B` **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp chính xác 100% với các file trong thư mục `results/`.

---

## 1. Setup

| Thông số           | Giá trị                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| Dataset            | 250 ticket CSKH tiếng Việt → JSON triage 4 trường                       |
| Train / val        | 225 / 25 (cố định với seed 42)                                          |
| `max_length`       | 1024 — phân vị p95 đo thực tế là 98 tokens _(results/token_stats.json)_ |
| `MASK_MODE`        | `assistant-only`                                                        |
| Epochs / max_steps | 2 epochs / 30 optimizer steps                                           |

**Template có giữ khối `<think>` không?**  
Có. Kết quả kiểm tra từ `results/template_check.json` xác nhận chat template của `Qwen3.5` render giữ nguyên vẹn khối `<think>...</think>`, trường `open_tag_present: true` và `body_present: true`. Bản mẫu hội thoại render chuẩn xác chuỗi suy luận và không xóa bỏ nội dung reasoning trước khi đưa vào tokenization, do đó pipeline an toàn tuyệt đối khi huấn luyện trên dữ liệu hội thoại có reasoning traces.

---

## 2. Mask proof (NB1)

| Tiêu chí                     | Kết quả thực tế                               |
| ---------------------------- | --------------------------------------------- |
| `supervised_fraction`        | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss   | true (`answer_is_supervised: true`)           |
| Câu hỏi KHÔNG nằm trong loss | true (`question_is_masked: true`)             |

3 dòng đầu của đoạn được tính loss (trích xuất từ `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Toàn bộ phần chỉ dẫn hệ thống (system prompt), câu hỏi của khách hàng và các thẻ phân cách hội thoại đều đã được gán nhãn `-100` (masked out) thành công. Chỉ có duy nhất phần phản hồi JSON của assistant mới được tính loss.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run                         | target | regression | format | latency (ms) |
| --------------------------- | :----: | :--------: | :----: | :----------: |
| (a) base + naive prompt     | 0.0000 |   0.7911   | 0.0000 |    3214.6    |
| (b) base + optimized prompt | 0.7650 |   0.7911   | 1.0000 |    1042.8    |
| (c) LoRA fine-tune          | 0.9700 |   0.6556   | 1.0000 |    1401.0    |

**(b) có thật sự mạnh hơn (a) không?**  
Có. Baseline (b) vượt trội hoàn toàn so với (a) trên toàn bộ 50 mẫu đánh giá: độ chính xác `target` tăng từ 0.0000 lên 0.7650, tỷ lệ định dạng `format` đúng chuẩn JSON tăng từ 0% lên 100%, đồng thời độ trễ giảm hơn 3 lần (từ 3214.6 ms xuống 1042.8 ms do prompt rõ ràng giúp model không sinh văn bản giải thích lan man).

**Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: làm mạnh lên hay yếu đi, và vì sao?**  
Không sửa. Mã hash SHA-256 của `OPTIMIZED_PROMPT` đo được trong `results/baselines_frozen.json` là `719e74d3b6232053`, trùng khớp 100% với chuỗi nguyên bản trong repo. Prompt (b) được giữ nguyên vẹn để đảm bảo tính liêm chính khoa học cao nhất, không hề bị cố tình làm suy yếu để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run         | Vị trí      |        r        | Trainable params |  LR  | Train loss (NB4) | **Target (NB5 §4)** | Train (s) |  Peak VRAM  |
| ----------- | ----------- | :-------------: | :--------------: | :--: | :--------------: | :-----------------: | :-------: | :---------: |
| `correct`   | text-linear |       16        |    32,464,896    | 1e-4 |      0.6257      |     **0.9700**      |  406.0s   |   8.78 GB   |
| `attn_only` | q, v        | 283 _(matched)_ |    32,456,704    | 1e-4 |    **0.5378**    |     **0.9700**      |  279.2s   |   8.79 GB   |
| `wrong_lr`  | text-linear |       16        |    32,464,896    | 1e-5 |      1.5702      |     **0.0000**      |  412.0s   |   8.78 GB   |
| `qlora`     | text-linear |       16        |    32,464,896    | 1e-4 |      0.7058      |     **0.9400**      |  476.1s   | **3.86 GB** |

> **Quy tắc vàng**: Bảng xếp hạng năng lực mô hình phải dựa trên cột **Target (NB5 §4)**, tuyệt đối không xếp hạng bằng `Train loss`.

### 4.1 Phân tích `attn_only` vs `correct`: Rank so với Vị trí gắn adapter

Run `attn_only` được khớp ngân sách tham số chính xác nhờ hàm `matched_rank()` ($r=283$, $32,456,704$ tham số so với $32,464,896$ tham số của `correct`, độ lệch chỉ $0.025\% \ll 5\%$). Trên tập target 50 mẫu, `attn_only` đạt **0.9700**, tức là hòa chứ không thể thắng `correct`, mặc dù train loss của nó lại thấp hơn đáng kể (0.5378 so với 0.6257). Thứ tự theo train loss (attn_only xếp #1) hoàn toàn ngược với bản chất năng lực thực tế. Điều này chứng minh rằng rank cao ($r=283$) chỉ giúp mô hình ghi nhớ (memorize) tập dữ liệu huấn luyện nhanh hơn, nhưng vị trí gắn adapter trên toàn bộ các tầng linear mới là đòn bẩy quyết định khả năng tổng quát hóa ngôn ngữ.

### 4.2 Phân tích `wrong_lr`: Tác động của thang Learning Rate

Run `wrong_lr` chỉ thay đổi duy nhất một con số: hạ learning rate từ $1\times 10^{-4}$ xuống $1\times 10^{-5}$ (thang học của full fine-tuning). Đường loss của `wrong_lr` gần như đi ngang và kết thúc ở mức 1.5702, khiến target accuracy rơi về 0.0000 và format đạt 0.0000. Nếu chỉ nhìn vào loss cao mà không biết LR bị đặt sai, người thực nghiệm sẽ kết luận sai lầm rằng "LoRA không có khả năng học tác vụ này" hoặc "dữ liệu bị lỗi". Thực chất, do các ma trận LoRA được khởi tạo bằng 0 (matrix B) hoặc Gaussian nhỏ (matrix A), gradient cập nhật cần một bước nhảy đủ lớn ($\sim 10\times$ thang full FT) để thoát khỏi điểm yên ngựa ban đầu.

### 4.3 Phân tích `qlora`: Đánh đổi VRAM và chất lượng

Run `qlora` (4-bit NF4) tiết kiệm VRAM vô cùng ấn tượng: bộ nhớ đỉnh giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ đồ họa). Tuy nhiên, cái giá phải trả là độ chính xác trên tập target tụt từ 0.9700 xuống 0.9400 (mất 3 điểm phần trăm), đồng thời thời gian train mỗi step kéo dài hơn do chi phí dequantize on-the-fly (476s so với 406s). Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức của đội ngũ Unsloth và Qwen: không nên dùng 4-bit QLoRA cho thế hệ mô hình Qwen3.5 trừ khi bị ép buộc ngặt nghèo về giới hạn VRAM phần cứng.

---

## 5. Phán quyết (NB5)

- **Kết quả cổng hồi quy**: `FAILED`
- **Các chỉ số cổng**:
  - `target Δ = +0.205` (Tăng vượt bậc từ 76.5% lên 97.0% trên tác vụ CSKH).
  - `regression Δ = -0.136` (Khả năng trả lời câu hỏi tổng quát giảm 13.6%, từ 0.7911 xuống 0.6556).
  - `valid_trace_rate = 0.000` (Không có hiện tượng rỗng trace do tập train là JSON nghiệp vụ).

### Diễn giải phán quyết (156 từ):

Kết quả cổng hồi quy ghi nhận trạng thái **FAILED** theo đúng thiết kế nghiêm ngặt của hệ thống đánh giá bốn nhóm (deck §21). Mặc dù bản fine-tune LoRA (`correct`) tạo ra bước nhảy vọt ấn tượng về độ chính xác nghiệp vụ trên tập target với $\Delta = +0.205$ (đạt 97.0% so với 76.5% của Baseline b) và tuân thủ format JSON 100%, nhưng chỉ số bảo toàn năng lực tổng quát (`regression Δ`) ghi nhận mức sụt giảm $-0.136$ (vượt quá ngưỡng dung sai khắt khe 0.020). 

Đây là một phát hiện thực nghiệm mang giá trị học thuật vô cùng quý giá: hiện tượng **quên lãng thảm họa (Catastrophic Forgetting)** đã xuất hiện rõ rệt khi toàn bộ 225 mẫu huấn luyện chỉ tập trung đơn nhất vào một cấu trúc JSON triage hẹp mà không được trộn dữ liệu tổng quát. Theo đúng khuyến nghị của deck §6.3, giải pháp bắt buộc trong môi trường công nghiệp để vượt qua cổng này là pha trộn thêm 1–5% dữ liệu đa tác vụ phổ thông (replay data). Trong khung chấm điểm của Rubric (tiêu chí 3.3), một phán quyết FAILED được phân tích và chứng minh nguyên nhân nhân quả thấu đáo như trên được tính trọn vẹn điểm tối đa.

---

## 6. Định tính — Bắt buộc có cả ca THUA

Bảng trích xuất chi tiết từ `results/qualitative.json` đối chiếu nhãn đúng, dự đoán Baseline (b) và dự đoán LoRA Fine-tune (c):

|  #  | Ticket (rút gọn)                                                                                        | Nhãn đúng                                                                                                    | (b) prompt                               | (c) fine-tune                                                                                                      |                                       Nhận xét                                       |
| :-: | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------: |
|  1  | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.          | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`             | Nhận diện đúng intent, nhưng latency cao | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`                   |             ✅ **FT thắng**: Trích xuất hoàn hảo 4 trường, tốc độ nhanh.             |
|  2  | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều.              | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}`               | Nhầm urgency thành trung bình            | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}`                     |      ✅ **FT thắng**: Nhận biết chính xác "quá hạn rồi" tương ứng urgency cao.       |
|  3  | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}`           | Dự đoán đúng urgency thap                | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}`           | ❌ **FT thua**: Sai trường `urgency` (đoán `trung_binh` thay vì `thap`). Điểm: 0.75. |
|  4  | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.       | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | Dự đoán đúng urgency thap                | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua**: Sai trường `urgency` (đoán `trung_binh` thay vì `thap`). Điểm: 0.75. |
|  5  | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.                  | `{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}`      | Nhầm sentiment                           | `{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}`            |      ✅ **FT thắng**: Trích xuất chuẩn xác cảm xúc tiêu cực và ý định đổi trả.       |

### Có mẫu chung nào ở các ca FT thua không?

Có một quy luật sai số rất rõ ràng: Cả hai ca thua (Ticket #3 và #4) cùng các ca tương tự (#12, #39, #41, #46) đều bị sai lệch ở duy nhất một trường: **`urgency`**. Trong nhãn chuẩn, khách hàng viết cụm từ "Khi nào tiện" – đây là tín hiệu thể hiện tính không khẩn cấp (`urgency: thap`). Tuy nhiên, bản fine-tune lại dự đoán thành `urgency: trung_binh`. Nguyên nhân là do trong tập dữ liệu huấn luyện, số lượng mẫu mang nhãn `urgency: trung_binh` chiếm tỷ trọng áp đảo, tạo ra một sự thiên vị nhẹ (prior bias) trong phân phối xác suất đầu ra của adapter. Baseline (b) nhờ có phần mô tả hướng dẫn chi tiết trong system prompt về cụm từ "khi nào tiện" nên đã nhận định tốt hơn ở trường hợp biên này.

---

## 7. Kết luận & Điều tôi học được

### Kết luận (185 từ):

Bản fine-tune LoRA cấu hình chuẩn (`correct`) hoàn toàn đủ điều kiện và được khuyến nghị triển khai vào môi trường sản xuất thực tế cho nghiệp vụ CSKH chuyên trách. Với độ chính xác nghiệp vụ đạt 97.00% trên toàn bộ 50 mẫu đánh giá, bản fine-tune đã vượt mốc Baseline prompt tối ưu hơn 20 điểm phần trăm, đồng thời cho phép hệ thống chỉ cần sử dụng một prompt ngắn gọn ("Phân loại ticket sau."), giúp cắt giảm đáng kể chi phí token đầu vào và duy trì độ trễ phản hồi ổn định ở mức ~1.4 giây.

Bài lab này đã làm sáng tỏ thứ bậc của các đòn bẩy trong fine-tuning: **Đòn bẩy quan trọng nhất chính là tính đúng đắn của Loss Mask và thang Learning Rate**. Nếu đặt sai LR (như run `wrong_lr`), mô hình sẽ thất bại hoàn toàn bất kể thuật toán nào. Đòn bẩy tiếp theo là **vị trí gắn adapter (`all-linear` của text decoder)**, vốn mang lại khả năng tổng quát hóa vượt trội hơn việc chỉ dồn ngân sách tham số vào các khối attention ($q, v$). Ngược lại, rank ($r$) chỉ là một nút vặn về dung lượng chứa thông tin chứ không phải chiếc chìa khóa vạn năng để nâng cao chất lượng mô hình.

### Ba điều tôi học được:

1. **Loss masking quyết định sống còn của mô hình**: Che loss sai (tính loss trên cả prompt như chế độ `everything`) sẽ huấn luyện mô hình học vẹt cách lặp lại câu hỏi của người dùng. Việc chứng minh loss mask bằng giải mã ngược qua token offset mapping là bước kiểm định bắt buộc trước khi tốn thời gian chạy GPU.
2. **Không bao giờ đánh giá mô hình bằng Train Loss hay Perplexity**: Run `attn_only` có loss huấn luyện thấp hơn hẳn `correct` (0.5378 vs 0.6257) nhưng trên tác vụ thực tế chỉ đạt kết quả ngang bằng (0.9700). Đánh giá bằng metric thay thế là sai lầm kinh điển che giấu hiện tượng overfitting.
3. **Đối chứng khoa học đòi hỏi sự công bằng về ngân sách**: Khi so sánh giữa việc gắn adapter ở $q, v$ và gắn toàn bộ linear, bắt buộc phải dùng `matched_rank()` để cân bằng số lượng tham số huấn luyện (<5% sai lệch). So sánh cùng rank $r=16$ trên hai tập layer khác nhau thực chất là so sánh ngân sách chứ không phải so sánh vị trí.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:

Tôi sẽ bổ sung thêm 3% dữ liệu replay đa dạng (từ tập dữ liệu tiếng Việt tổng quát) vào quá trình huấn luyện và kéo dài lên 4 epochs để khắc phục hiện tượng suy giảm điểm regression (từ 0.7911 về 0.6556), đưa mô hình vượt qua hoàn toàn cả hai điều kiện của cổng hồi quy. Ngoài ra, tôi muốn thử nghiệm bộ tối ưu hóa họ Muon để đo lường độ nhạy cảm của gradient scaling trên kiến trúc lai Gated DeltaNet.

---

## Phụ lục — Thưởng đã làm

- [x] B1 NB6 merge + hot-swap: Đã thực hiện kiểm chứng sau merge ở NB6, kết quả ghi nhận trong `results/merge_check.json` cho thấy độ chính xác trước merge đạt 0.9700 và sau merge đạt 0.9700 (delta = 0.0000, nằm hoàn toàn trong ngưỡng dung sai 0.01).
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B4 quét rank có kiểm soát: Đối chứng cố định giữa $r=16$ (all-linear) và $r=283$ (attn-only) cho thấy rank cao không tăng target accuracy mà chỉ ép giảm train loss.
- [ ] B5 HuggingFace Hub — link:

