# Data Statement — UIT DSC 2026 Task 2

> Tạo một bản riêng cho từng release candidate. Thay mọi `[REQUIRED]`; không
> nộp template chưa hoàn thiện.

## 1. Identity

- Team: `[REQUIRED]`
- Release ID: `[REQUIRED]`
- Code commit: `[REQUIRED]`
- Created at (UTC): `[REQUIRED]`
- Competition phase: `[warm-up|public|private]`

## 2. Official data used

| Resource | Organizer filename | SHA-256/revision | Record count | Purpose |
|---|---|---|---:|---|
| Questions | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | inference/evaluation |
| Context corpus | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | retrieval/indexing |
| Training/dev labels | `[NONE or REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` | `[REQUIRED]` |

Xác nhận:

- [ ] Chỉ dùng dữ liệu BTC.
- [ ] Không gán nhãn thủ công.
- [ ] Không dùng external corpus hoặc external augmentation.
- [ ] Không dùng answer của evaluation case làm prediction/input.
- [ ] Không có dữ liệu cá nhân nhạy cảm chưa được ẩn danh trong output/log.

## 3. Processing

- Adapter/version: `[REQUIRED]`
- Cleaning policy/config hash: `[REQUIRED]`
- Parsing/chunking policy/config hash: `[REQUIRED]`
- Rejected/skipped records and reasons: `[REQUIRED]`
- Artifact manifests and checksums: `[REQUIRED]`

Mô tả mọi biến đổi legal text và cách bảo toàn Điều/Khoản/Điểm, số, thời hạn,
phủ định và ngoại lệ: `[REQUIRED]`.

## 4. Split and leakage controls

- Split method/seed: `[REQUIRED]`
- Duplicate detection: `[REQUIRED]`
- Cross-split leakage checks: `[REQUIRED]`
- Retrieval-label provenance: `[REQUIRED]`

## 5. Limitations and risk controls

- Coverage limitations: `[REQUIRED]`
- Known missing/ambiguous fields: `[REQUIRED]`
- Legal-validity limitations: `[REQUIRED]`
- Privacy and misuse mitigations: `[REQUIRED]`

## 6. Reproduction evidence

- Dataset inventory report: `[REQUIRED]`
- Build validation report: `[REQUIRED]`
- Docker image digest: `[REQUIRED]`
- Reproduction command: `[REQUIRED]`
