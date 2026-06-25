# Lab 21 — Evaluation Report

**Học viên**: Tôn Thành Đạt — 2A202600780
**Ngày nộp**: 25/06/2026
**Submission option**: Option B (HuggingFace Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (T4 profile)
- **Dataset**: Custom Domain Dataset (Âm nhạc lý thuyết tiếng Việt), 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024
- **GPU**: Tesla T4, 15GB VRAM (Google Colab)
- **Training cost**: ~$0 (Sử dụng Free Colab)
- **HF Hub link**: [https://huggingface.co/thanhdatton0/qwen2.5-3b-vi-lab21-r16](https://huggingface.co/thanhdatton0/qwen2.5-3b-vi-lab21-r16)
- **Github repository**: [https://github.com/Tonthanhdat/lab21_2A202600780_TonThanhDat.git](https://github.com/Tonthanhdat/lab21_2A202600780_TonThanhDat.git)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 15,998,976      | 11.94 min  | 5.22 GB   | 1.1209    | 3.0676     |
| 16   | 30,965,760      | 12.85 min  | 4.78 GB   | 1.0797    | 2.9439     |
| 64   | 120,766,464     | 12.72 min  | 7.16 GB   | 1.0252    | 2.7877     |

## 3. Loss Curve Analysis
**W&B Link**: [https://wandb.ai/thanhdatton0-/huggingface?nw=nwuserthanhdatton0](https://wandb.ai/thanhdatton0-/huggingface?nw=nwuserthanhdatton0)

**Quan sát**: Quá trình training diễn ra khá tốt và ổn định. Dựa trên số liệu Eval Loss giảm dần (từ 1.1209 xuống 1.0797 rồi 1.0252 khi tăng rank), không có dấu hiệu overfitting nghiêm trọng trên tập đánh giá. Việc sử dụng learning rate cosine schedule giúp mô hình hội tụ tốt. Đáng chú ý là model học các định dạng cấu trúc câu trả lời của dữ liệu âm nhạc rất nhanh.

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh chất lượng trả lời giữa Base Model và Fine-tuned Model (r=16) trên tập prompt chung chung (không phải domain âm nhạc) để xem sự ảnh hưởng của LoRA adapter:

### Example 1
- **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Output (Base)**: Machine learning là quá trình máy tự biến đổi và cải thiện dựa trên dữ liệu. Khi hiểu machine learning, người học dễ nghe ra vai trò của yếu tố này trong bài nhạc, video hoặc ứng dụng.
- **Output (Fine-tuned)**: Machine learning là quá trình máy tự biến đổi và áp dụng từ kinh nghiệm. Khi hiểu machine learning, người học dễ nghe ra vai trò của yếu tố này trong bài nhạc, video hoặc ứng dụng.
- **Nhận xét**: Cả Base và Fine-tuned model đều bị ảnh hưởng mạnh bởi văn phong âm nhạc của tập train (xuất hiện cụm từ "bài nhạc", "người học dễ nghe ra vai trò"). Mô hình đã học được cấu trúc của file custom dataset nhưng áp dụng sai ngữ cảnh.

### Example 2
- **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Output (Base)**: Để tính số Fibonacci, trước hết hãy tạo một hàm đệ quy với điều kiện n <= 0 trả 0, n == 1 trả 1 và n > 1 gọi lần lượt n-1 và n-2. Sau đó gọi hàm này với tham số n để nhận kết quả.
- **Output (Fine-tuned)**: Nhạc cụ phách dương thường tạo cảm giác rộng, thoáng và giàu màu sắc. Khi áp dụng, hãy tập trung vào hợp âm mở, nhịp và hợp âm chuyển động. Tốt nhất nên nghe ví dụ ngắn để nhận ra nhạc cụ truyền thống của Việt Nam.
- **Nhận xét**: Đây là một ví dụ tuyệt vời về sự "ghi đè kiến thức" (Catastrophic Forgetting) khi fine-tune trên một tập data quá hẹp. Mô hình Base vẫn nhớ cách code Python đệ quy, nhưng mô hình Fine-tuned đã bị "nhạc hóa" hoàn toàn và trả lời lạc đề sang nhạc cụ truyền thống.

### Example 3
- **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Output (Base)**: Nguyên tắc này giúp người dùng dễ hiểu, giao tiếp với sản phẩm và đạt hiệu quả công việc tốt hơn. Tránh lặp lại, tránh tạo cảm giác dày đặc, dùng màu sắc, font và bố cục hợp lý.
- **Output (Fine-tuned)**: Một UI/UX tốt nên rõ ràng, thân thiện, có màu sắc phù hợp, phản ứng nhanh và không gây stress. Khi hiểu 5 nguyên tắc này, người học dễ nghe ra vai trò của yếu tố này trong sản phẩm.
- **Nhận xét**: Mô hình Fine-tuned lại tiếp tục chèn cụm "khi hiểu..., người học dễ nghe ra vai trò" - đây là cấu trúc câu trả lời cố định được lặp đi lặp lại rất nhiều trong file `custom_dataset.json`. Mô hình đã overfit vào format này.

### Example 4
- **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Output (Base)**: LoRA là kỹ thuật bổ sung thông số nhỏ vào LLM, còn QLoRA là cải tiến dựa trên tính chất quan trọng của quan sát. LoRA dễ áp dụng nhưng QLoRA có thể tăng độ chính xác và hiệu năng đáng kể.
- **Output (Fine-tuned)**: LoRA là phương pháp bổ sung thông tin vào mô hình LLM bằng biến đổi kỹ thuật số. QLoRA là phiên bản quantized (đơn hóa) của LoRA, nhờ vậy mô hình nhanh hơn và tiêu thụ ít tài nguyên.
- **Nhận xét**: Cả hai câu trả lời đều khá tốt, tuy nhiên ở đây Fine-tuned model giải thích khái niệm QLoRA dễ hiểu và rõ ràng hơn Base model.

### Example 5
- **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Output (Base)**: Prompt engineering là viết câu lệnh để mô tả dữ liệu và yêu cầu AI thực hiện. RAG là cách sử dụng dữ liệu nguồn (raw data), augementation (thêm dữ liệu) và generation (làm mới) để cải thiện kết quả. Fine-tuning là thay đổi một chi tiết nhỏ của mạng để phù hợp với nhiệm vụ, nhờ vậy mỗi phương pháp có
- **Output (Fine-tuned)**: Prompt engineering là tạo câu lệnh cho mô hình làm việc tốt hơn. RAG là cách sử dụng rerank, aggregation và generation để kết hợp nhiều nguồn dữ liệu. Fine-tuning là sửa đổi mô hình nhỏ để tập trung vào một phạm vi công việc.
- **Nhận xét**: Model fine-tuned sinh ra câu trả lời gọn gàng hơn và không bị cắt ngang giữa chừng như Base model. Dù bị bias về âm nhạc ở những câu khác, nó vẫn giữ lại được một phần kiến thức công nghệ.

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thí nghiệm trên tập dữ liệu Âm nhạc (200 câu), mức rank **r=16 cho kết quả tối ưu nhất (ROI tốt nhất)**. Nó tạo ra sự cân bằng hoàn hảo:
- Thời gian train chỉ xấp xỉ r=8 (12.85 phút so với 11.94 phút), và tiêu tốn VRAM ít nhất trong cả 3 rank (4.78 GB).
- Perplexity (2.94) giảm đáng kể so với r=8 (3.06). 

Hiện tượng **Diminishing Returns (lợi tức giảm dần)** được thấy rất rõ khi tăng rank lên `r=64`. Mặc dù tốn VRAM cao hơn đáng kể (7.16 GB, gấp 1.5 lần so với r=16) do số lượng parameter phải train tăng từ 30 triệu lên 120 triệu, chỉ số Perplexity chỉ giảm xuống còn 2.78 (không cải thiện được bao nhiêu). Điều này chứng tỏ với một dataset nhỏ 200 câu, rank 64 là dư thừa và lãng phí bộ nhớ.

**Recommendation**: Nếu phải deploy production cho một con bot, tôi sẽ chọn **r=16** vì nó tiết kiệm VRAM, tốc độ train nhanh mà vẫn đảm bảo độ chính xác (perplexity) đủ tốt.

## 6. What I Learned
- **Hiện tượng Catastrophic Forgetting (Quên kiến thức cũ)**: Thông qua các ví dụ qualitative (hỏi về Python nhưng model trả lời về nhạc cụ), tôi nhận ra việc fine-tune trên một tập dataset hẹp, nhỏ và lặp cấu trúc liên tục có thể khiến model phá vỡ hoàn toàn những kiến thức chung ban đầu của nó.
- **Nhạy cảm với cấu trúc câu**: Model LLM học lặp cấu trúc rất nhanh. Câu "Khi hiểu..., người học dễ nghe ra..." xuất hiện quá nhiều trong dataset đã làm model overfit và áp dụng bừa bãi vào mọi câu trả lời.
- **Rank Trade-off**: Không phải rank càng cao thì chất lượng càng tỉ lệ thuận. Rank 16 là điểm "ngọt" (sweet spot) lý tưởng nhất về mặt cân bằng giữa chất lượng và tài nguyên GPU.
