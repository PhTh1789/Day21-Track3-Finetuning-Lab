# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng ở run `attn_only`: khi chỉ gắn LoRA vào $q, v$ nhưng nâng rank lên cực đại ($r=283$) để cân bằng số lượng tham số với bản `correct`, mô hình đạt `final_loss` thấp hơn đáng kể (0.5378 so với 0.6257 của bản chuẩn). Nếu chỉ nhìn vào loss theo thói quen cũ, ai cũng sẽ nghĩ rằng cấu hình này chiến thắng. Tuy nhiên, khi chấm thực tế trên tập kiểm tra target, nó chỉ đạt 0.9375 (hòa `correct`, không hơn). Đây là minh chứng nhãn tiền rằng train loss thấp chỉ là hiện tượng ép ghi nhớ (memorization) cục bộ, và việc phân bổ adapter trên toàn bộ các tầng linear của text decoder mới là yếu tố quyết định năng lực biểu diễn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở khâu sinh văn bản để đánh giá (inference generation) qua các baseline ở NB2 và NB5 trên GPU T4, đặc biệt là khi phải đánh giá 3 baseline và chấm chéo 3 adapter đối chứng. Điều này hoàn toàn trái ngược với dự đoán ban đầu của tôi: tôi từng nghĩ bước huấn luyện tính gradient (training backward pass) sẽ lâu nhất. Thực tế, huấn luyện 30 bước chỉ mất 4–6 phút mỗi run, trong khi autoregressive decoding sinh tuần tự từng token cho hàng chục mẫu thử nghiệm mới là nút cổ chai thời gian lớn nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng giữ hai niềm tin sai lầm:
1. *“Muốn mô hình thông minh hơn thì cứ tăng rank LoRA lên thật cao.”* — Giờ tôi hiểu rank là dung lượng chứa thông tin so với lượng kiến thức trong dữ liệu; với tập vài trăm mẫu thì $r=16$ đã quá đủ, tăng lên $r=283$ không làm tăng thêm năng lực tổng quát hóa.
2. *“QLoRA 4-bit luôn là mặc định tốt nhất cho mọi bài toán thực tế.”* — Giờ tôi nhận ra sai số lượng tử hóa của 4-bit trên các kiến trúc lai mới như Qwen3.5 làm tụt gần 10 điểm % độ chính xác (từ 93.8% xuống 84.4%). Nếu GPU còn chứa vừa 16-bit LoRA thì tuyệt đối không nên đánh đổi chất lượng lấy dung lượng.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ phân tích ma trận tham số trong `modeling.py`, đối soát tính liêm chính của các file JSON đầu ra với rubric và hỗ trợ cấu trúc hóa báo cáo khoa học. Điểm mà AI assistant dễ mắc bẫy nhất là các gợi ý thiết lập mặc định `bf16=True` theo phong trào các bài báo 2026, trong khi GPU Tesla T4 (kiến trúc Turing sm_75) không hề hỗ trợ bfloat16 phần cứng. Nếu không tỉnh táo kiểm tra cơ chế GradScaler ở `labkit/device.py`, việc ép dùng bf16 sẽ khiến quá trình huấn luyện bị suy giảm tốc độ nghiêm trọng hoặc nổ gradient.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm không phải là mở code ra huấn luyện ngay, mà là **xây dựng tập đánh giá đóng băng (eval benchmark) và đo Baseline (b) với prompt đã được tối ưu kỹ càng**. Nếu Baseline (b) đã giải quyết được 90–95% nhu cầu với chi phí chấp nhận được, tôi sẽ khuyến nghị khách hàng dùng prompt engineering thay vì fine-tune. Nếu bắt buộc phải fine-tune để tối ưu độ trễ và chi phí token, hành động kỹ thuật đầu tiên là **chạy giải mã ngược loss mask** để chứng minh toán học rằng loss chỉ tính trên câu trả lời mong muốn, tuyệt đối không tính loss trên prompt của khách hàng.
