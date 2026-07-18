# 01. Project Context

## 1. Project Overview

Dự án xây dựng một hệ thống Agentic Retrieval-Augmented Generation
cho bài toán trả lời câu hỏi pháp luật Việt Nam.

Mục tiêu dài hạn là phục vụ UIT Data Science Challenge 2026 với chủ đề:

> Trả lời câu hỏi pháp luật Việt Nam.

Tại thời điểm bắt đầu dự án, Ban tổ chức chưa công bố đầy đủ:

- quy chế cuộc thi;
- dữ liệu huấn luyện;
- dữ liệu kiểm thử;
- corpus pháp luật chính thức;
- tiêu chí đánh giá;
- định dạng input;
- định dạng output;
- hình thức nộp bài;
- giới hạn tài nguyên;
- quy định sử dụng dữ liệu ngoài;
- quy định sử dụng Internet;
- quy định sử dụng API LLM thương mại.

Do đó, dự án không được thiết kế gắn cứng với một định dạng dữ liệu hoặc
một hình thức đánh giá cụ thể.

---

## 2. Development Strategy

Nhóm không chờ Ban tổ chức công bố dữ liệu mới bắt đầu xây dựng hệ thống.

Trong giai đoạn hiện tại, nhóm xây dựng một hệ thống hoàn chỉnh dựa trên:

- kiến trúc Agentic RAG trong tài liệu của nhóm AIO;
- dataset `th1nhng0/vietnamese-legal-documents`;
- các mô hình pretrained;
- các chiến lược retrieval không cần fine-tune ban đầu.

Khi dữ liệu chính thức của Ban tổ chức được công bố, hệ thống phải có
khả năng thích ứng thông qua:

- dataset adapter;
- schema mapping;
- corpus replacement;
- index rebuilding;
- training pipeline;
- evaluation pipeline;
- output adapter;
- submission formatter.

Core architecture không được phụ thuộc trực tiếp vào dataset AIO.

---

## 3. Current Reference Architecture

Nguồn tham khảo chính là tài liệu xây dựng Agentic RAG của nhóm AIO.

Kiến trúc tham khảo gồm bốn lớp:

1. Ingestion Layer.
2. Retrieval Layer.
3. Agent and Tools Layer.
4. Serving Layer.

Retrieval Layer bao gồm các chiến lược:

- BM25 retrieval;
- dense retrieval;
- hybrid retrieval bằng Reciprocal Rank Fusion;
- cross-encoder reranking;
- graph retrieval.

Agent có nhiệm vụ điều phối các công cụ trên, đánh giá ngữ cảnh và thực
hiện truy xuất lại khi cần.

---

## 4. Current Data Scope

Trong giai đoạn hiện tại, chỉ sử dụng dataset:

`th1nhng0/vietnamese-legal-documents`

Dataset được dùng làm:

- corpus văn bản pháp luật;
- nguồn metadata pháp lý;
- nguồn quan hệ giữa các văn bản;
- nguồn xây BM25 index;
- nguồn xây vector index;
- nguồn xây legal graph;
- nguồn evidence cho answer generation.

Không tích hợp các corpus hoặc QA dataset khác trong baseline hiện tại.

---

## 5. Current System Scope

Phạm vi hiện tại gồm:

- tải và audit dataset;
- chuẩn hóa metadata;
- làm sạch HTML;
- nhận diện cấu trúc pháp lý;
- chunk văn bản;
- xây BM25 index;
- xây vector index;
- xây graph index;
- xây các fixed retrieval baseline;
- xây hybrid retrieval;
- rerank candidate;
- xây context;
- sinh câu trả lời từ evidence;
- kiểm tra citation;
- đóng gói các chức năng thành tools;
- xây Agentic workflow sau cùng;
- cung cấp API và giao diện thử nghiệm.

---

## 6. Out of Scope for Initial Baseline

Những nội dung sau chưa thuộc baseline đầu tiên:

- fine-tune dense retriever;
- fine-tune cross-encoder reranker;
- fine-tune answer generator;
- web search fallback;
- crawling dữ liệu pháp luật mới;
- OCR tài liệu scan;
- tự động cập nhật hiệu lực pháp lý theo thời gian thực;
- autonomous agent không giới hạn vòng lặp;
- multi-agent system;
- tư vấn pháp lý chính thức;
- production deployment quy mô lớn;
- synthetic QA làm ground truth chính thức.

---

## 7. Development Philosophy

Hệ thống được xây theo nguyên tắc:

```text
Data quality
→ Data normalization
→ Legal chunking
→ Indexing
→ Fixed retrieval
→ Reranking
→ Answer generation
→ Verification
→ Tools
→ Agentic orchestration
→ Serving
```

Không được xây Agent trước khi các tool phía dưới hoạt động ổn định.

Agent chỉ có giá trị khi các retrieval strategy đã được kiểm thử độc lập.

---

## 8. Safety and Legal Position

Hệ thống là công cụ hỗ trợ:

- truy xuất văn bản pháp luật;
- tổng hợp evidence;
- giải thích thông tin dựa trên corpus.

Hệ thống không thay thế:

- luật sư;
- chuyên gia pháp lý;
- cơ quan nhà nước;
- nguồn văn bản pháp luật chính thức.

Trạng thái hiệu lực trong dataset chỉ được xem là snapshot tại thời điểm
thu thập dữ liệu.

Hệ thống không được mặc định rằng metadata này luôn phản ánh tình trạng
pháp lý mới nhất.