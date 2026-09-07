# Track B — External Fact-Checking Architecture

> Tài liệu thiết kế hệ thống Track B: kiểm chứng nội dung phát ngôn trong video dài.
> Đây là luồng song song với Track A (deepfake detection nội tại), chạy độc lập và kết hợp ở bước cuối.

---

## 0. Framing — Track B là gì về mặt kỹ thuật?

Track B là một **multi-agent fact-checking system** — toàn bộ pipeline từ B1 đến B5 đều dùng LLM với tool use, không dùng specialized model đơn tác vụ như BERT, T5, hay ClaimBuster.

```
Agent B2: nhận transcript → extract candidate claims
Agent B3: nhận claims    → detect references, decontextualise, build inter-claim DAG
Agent B4: nhận DAG       → gọi tool search, plan queries, judge verdict từng claim
Agent B5: nhận cả hai track → tổng hợp explainable report
```

**Novelty không phải ở việc dùng multi-agent** (AutoGen, CrewAI, LangGraph đã làm) — mà ở **cái task và structure mà agents tạo ra**:

- Agents được giao nhiệm vụ xây dựng **inter-claim dependency graph** — chưa tồn tại trong bất kỳ fact-checking agent system nào
- Cascade verdict theo typed edges là **deterministic rule** áp lên DAG do agents xây — không phải thêm một LLM call
- Decontextualisation và DAG construction xảy ra **trong một agent step** — không tách rời

> *"Track B là một multi-agent fact-checking system trong đó các agents cộng tác để xây dựng và verify một inter-claim dependency graph — cấu trúc chưa tồn tại trong bất kỳ fact-checking agent system nào trước đó."*

---

## 1. Động lực và Gap

Track A (VERA hiện tại) phát hiện deepfake dựa trên **tín hiệu kỹ thuật** — blending artifact, lip-sync, audio jitter. Nhưng có hai loại video nguy hiểm Track A không xử lý được:

- **Genuine video + false claims**: người thật đang phát tán thông tin sai
- **Deepfake video + false claims**: kết hợp nguy hiểm nhất — giả mạo danh tính lẫn nội dung

Track B kiểm chứng **nội dung phát ngôn** từ transcript của video — độc lập với Track A.

**Gap trong literature**: Tất cả hệ thống fact-checking hiện có (LiveFC, AVeriTeC, MAD, Trification) đều verify từng claim độc lập, không theo dõi **chuỗi phụ thuộc logic giữa các claim theo thời gian**. Một speaker có thể đặt tiền đề sai ở phút 2, rồi xây thêm 5 claim lên đó ở phút 8 — các hệ thống hiện tại bỏ sót hoàn toàn.

---

## 2. Claim Taxonomy

### Theo giai đoạn pipeline

| Tên | Giai đoạn | Đặc điểm |
|---|---|---|
| **Candidate claim** | B2 output | Khẳng định thực tế được extract từ transcript, có thể chứa reference mơ hồ ("con số đó", "vì vậy") |
| **Standalone claim** | B3 output | Self-contained — tất cả reference đã được resolve, có thể gửi thẳng cho verifier |

### Theo vị trí trong DAG (sau B3)

| Tên | Đặc điểm | Ví dụ |
|---|---|---|
| **Primitive claim** | Không phụ thuộc claim nào, đã standalone từ B2, root node trong DAG | "GDP tăng 5.8%" |
| **Derived claim** | Có ≥1 reference được resolve bởi B3, có incoming edge trong DAG | "GDP 5.8% chứng tỏ kinh tế phục hồi" |

---

## 3. Kiến trúc — 5 Module

```
Full video
    │
    ▼ Module B1
Full transcript (Whisper ASR trên toàn video)
    │
    ▼ Module B2
Candidate claims (có timestamp)
    │
    ▼ Module B3 (xử lý theo thứ tự thời gian)
Standalone claims + Labeled Inter-claim DAG
    │
    ▼ Module B4 (multi-pass verification)
Per-claim verdicts + source evidence
    │
    ▼ Module B5
VLM final synthesis → Explainable report
```

---

## 4. Module B1 — Full Transcript + Speaker Attribution

**Input**: video file  
**Output**: full transcript với timestamps + speaker labels

Dùng **WhisperX** ([github.com/m-bain/whisperx](https://github.com/m-bain/whisperx)) — tích hợp sẵn Whisper ASR + wav2vec2 forced alignment + pyannote speaker diarization trong 1 pipeline.

- Chạy trên **toàn bộ video** (không dùng per-chunk ASR từ Track A) — tránh boundary artifacts, cần transcript liên tục để resolve reference xa
- wav2vec2 alignment cho word-level timestamps chính xác hơn Whisper gốc — cần thiết khi 2 speaker xen kẽ liên tục (interview, debate)
- MAD2 (dataset eval chính của Track B) cũng dùng WhisperX → kết quả so sánh fair
- Output format: `[(text, start_time, end_time, speaker_id), ...]`

### Speech Segmentation — Layered Approach

Audio là dòng liên tục không có "dấu chấm" hay "xuống dòng" như text. Speech segmentation chuyển nó thành **intervals có nhãn** theo 4 lớp:

| Lớp | Tool | Output | Dùng cho Track B |
|---|---|---|---|
| Speaker | pyannote.audio | "ai nói, từ giây nào đến giây nào" | Attribute claim → speaker |
| Utterance | Whisper segment output | "1 turn nói gì" | Unit input cho Module B2 extract claim |
| Topic | text-level sau khi có transcript | "đang nói về chủ đề nào" | Cluster claims → sub-graph |
| Temporal order | inherent trong audio | "thứ tự xảy ra" | **Hướng cạnh trong DAG (miễn phí)** |

**Whisper đã giải quyết lớp Utterance sẵn**: output mặc định của Whisper là segment-level (không phải full text), mỗi segment ≈ 1 câu, có timestamps. Module B2 xử lý từng Whisper segment thay vì full transcript.

**pyannote.audio giải quyết lớp Speaker**: cần thiết khi video có ≥2 người nói (interview, debate, podcast) — biết "speaker A nói claim X, speaker B phản bác bằng claim Y" là prerequisite để xây cross-speaker dependency edges.

**Lớp Temporal order là contribution quan trọng nhất cho DAG**: trong spoken content, referent luôn nằm trong quá khứ (không thể reference điều chưa nói). Temporal order của audio cho biết hướng edge `A → B` (A được nói trước B) mà không cần bất kỳ discourse parser nào — đây là lý do acyclicity của DAG được đảm bảo tự nhiên.

**Không cần SLM-based speech segmentation** (kiểu arXiv:2501.03711): model này giải quyết acoustic-semantic boundary detection — giá trị chủ yếu cho unsupervised topic segmentation trong audio thuần. Với Track B đã có Whisper + pyannote, thêm SLM tăng độ phức tạp mà không thêm thông tin mới cần thiết.

### Relevant Datasets

**MAD** ([arXiv:2508.12186](https://arxiv.org/abs/2508.12186)) — 600 multi-turn audio dialogues, annotated: speaker turns, checkworthiness, veracity per sentence và per dialogue.

**MAD2** ([arXiv:2606.11420](https://arxiv.org/abs/2606.11420)) — 1,000 two-speaker dialogues, 3,368 check-worthy claims, ~10 giờ audio (MoonCast synthesis), **WhisperX word-level timestamps**. Dataset phù hợp nhất để eval Track B vì 2-speaker structure map trực tiếp vào cross-speaker dependency edges.

---

## 5. Module B2 — Claim Extraction

**Input**: transcript từ WhisperX — `[(text, start_time, end_time, speaker_id), ...]`  
**Output**: danh sách candidate claims, mỗi claim có `(text, timestamp, speaker_id)`

### Nhiệm vụ

Lọc ra các **khẳng định thực tế có thể verify** — bỏ qua:
- Opinion: "I think this is a good time..."
- Câu hỏi, lời chào, transition phrases
- Cảm xúc, dự đoán không có cơ sở

### Phương pháp

Dùng **Qwen3** (hoặc LLM tương đương) với prompt zero-shot, xử lý từng Whisper segment:

```
Given the following transcript segment spoken by {speaker_id} at {start_time}s:
"{text}"

Extract all verifiable factual claims. Each claim must be a statement
that can be true or false. Return as a list.
If no factual claim exists, return empty list.
```

Không dùng ClaimBuster (threshold arbitrary, domain-dependent) — LLM linh hoạt hơn với spoken content.

### Lưu ý

- Một câu dài có thể chứa nhiều claims → B2 tách ra
- `speaker_id` được giữ nguyên theo từng claim → B3 dùng để detect cross-speaker `contradicts` edges
- Candidate claims **chưa self-contained** — vẫn có thể chứa "that figure", "therefore"

---

## 6. Module B3 — Labeled Inter-claim DAG Construction

**Input**: danh sách candidate claims (đã sort theo timestamp)  
**Output**: standalone claims + labeled DAG

Đây là module **novel nhất** trong Track B.

### Nguyên tắc cốt lõi

Khi resolve một reference trong claim B → tìm thấy antecedent là claim A → đồng thời làm hai việc:

1. **Decontextualise**: rewrite claim B để self-contained (thay "con số đó" bằng "GDP 5.8%")
2. **Add DAG edge**: ghi nhận B phụ thuộc A (thêm edge B → A với type tương ứng)

Hai việc này là **một thao tác** — không tách rời.

### Quy trình xử lý — Implementation

**Một LLM call cho toàn bộ B3** (với video ngắn-vừa):

```
Input cho LLM:
  - Toàn bộ transcript (hoặc đoạn đủ dài — xem phần context window bên dưới)
  - Danh sách tất cả candidate claims kèm timestamp

Prompt: "Với mỗi claim có reference mơ hồ, dựa vào transcript:
         1. Xác định nó đang chỉ đến claim nào
         2. Rewrite thành standalone
         3. Ghi edge type tương ứng"

Output: tất cả standalone claims + DAG edges
```

LLM đọc toàn bộ transcript một lần — tự nhận ra luồng ngôn ngữ, resolve tất cả references trong một pass. Không cần loop từng claim.

### Context window và sliding window

**Tại sao không phải luôn luôn đưa full transcript?**

Với video dài, full transcript vượt quá context window của model → dùng sliding window.

**Cách xác định window:**

```
Claim k đang xét (timestamp T)
    │
    ▼
Lấy transcript ngược từ T về quá khứ
cho đến khi đủ context token budget của model
    │
    ▼
Tập claims nằm trong đoạn transcript đó = candidates
    │
    ▼
LLM đọc đoạn transcript + claim k
→ resolve reference trong tập candidates đó
```

**Quan trọng**: không fix khoảng thời gian cố định (T phút) mà fix **số token cố định**. Số phút tương ứng sẽ khác nhau tùy tốc độ nói và mật độ thông tin.

**Lý do linguistically motivated:**

Trong văn nói, reference hầu như là short-range — "con số đó" ở phút 15 rất hiếm chỉ về thứ gì đó ở phút 1. Speaker nói theo luồng tuyến tính, reference thường nằm trong vài phút gần nhất. Một context token budget hợp lý (~4,000-8,000 tokens) đã cover được phần lớn references thực tế trong speech.

**Với video rất dài (30+ phút):**

```
Chunk có overlap:
  Chunk 1: token[0    → 8000]   → LLM call 1
  Chunk 2: token[6000 → 14000]  → LLM call 2  (overlap 2000 tokens)
  Chunk 3: token[12000→ 20000]  → LLM call 3  (overlap 2000 tokens)
```

Overlap đảm bảo reference ở ranh giới chunk không bị miss. Tổng số LLM calls = O(số chunks), không phải O(N²) hay O(N).

### Tại sao temporal order đảm bảo acyclicity

### Tại sao temporal order đảm bảo acyclicity

Speech chỉ có thể reference những gì đã nói trước → edge luôn đi từ claim mới hơn về claim cũ hơn → **không thể có cycle** → DAG được đảm bảo tự nhiên, không cần enforce thủ công.

### Các loại edge (Labeled DAG)

Chỉ giữ hai loại edge — mỗi loại enable một verdict mới mà không có thì không phát hiện được:

| Edge type | Dấu hiệu trong speech | Enable verdict mới |
|---|---|---|
| `presupposes` | "con số đó", "vì vậy", "điều này" | WEAKENED + MISLEADING |
| `contradicts` | "thực ra", "ngược lại", "nhưng thực tế" | CONTRADICTION |

> **Lý do không có `refines`**: claim "cụ thể hơn là Q4 tăng 6.1%" là fact độc lập — verify riêng như primitive claim cho kết quả chính xác hơn là cascade WEAKENED. `refines` không enable verdict mới nào nên bị loại bỏ.

### Tại sao cần rewrite (không chỉ thêm edge)?

Edge ghi nhận **dependency logic**. Rewrite cung cấp **string self-contained** để gửi cho external verifier (B4). B4 không cần traverse graph để hiểu claim — nhận trực tiếp standalone text.

### Novelty so với literature

| | ACL 2024 Decontextualisation | Trification | **VERA Track B** |
|---|---|---|---|
| Domain | Text (news) | Text (written claims) | **Speech (spoken video)** |
| Scope | 1 claim/document | Intra-claim (1 claim → sub-tasks) | **Inter-claim (nhiều claim theo thời gian)** |
| DAG | Không | Intra-claim DAG | **Inter-claim DAG** |
| Acyclicity | N/A | Enforce thủ công | **Tự nhiên từ temporal order** |
| Edge types | N/A | Sub-task edges | **presupposes / contradicts** |

---

## 7. Module B4 — Multi-pass External Verification

**Input**: standalone claims + labeled DAG (chỉ còn `presupposes` và `contradicts`)  
**Output**: per-claim verdict + evidence JSON

---

### Tại sao phải multi-pass (không thể verify song song)?

Verdict của claim cha quyết định có cần verify claim con không. Nếu verify song song, lúc xử lý `atom_03` ta chưa biết `atom_02` đúng hay sai — không biết nên gọi API hay bỏ qua. Vì vậy phải xử lý theo **topological order** (cha trước, con sau).

---

### Hai khái niệm cốt lõi của B4

**Premise validity** — con số/sự kiện trong claim có đúng không?

> "GDP tăng 5.8%" → search Wikipedia/GSO → so sánh với nguồn → TRUE/FALSE

**Inference validity** — dù premise đúng, kết luận rút ra có hợp lý không?

> "GDP 5.8% chứng tỏ kinh tế phục hồi mạnh" → hỏi: "5.8% có đủ để khẳng định phục hồi mạnh?"

Hai chiều này **độc lập** — premise đúng không có nghĩa inference đúng. Flat fact-checkers (LiveFC, AVeriTeC) chỉ check premise. B4 check cả hai → phát hiện được **MISLEADING**.

---

### Pass 1 — Verify tất cả primitive claims

Primitive claims = không có `presupposes` incoming edge. Bao gồm cả các claim chỉ có `contradicts` edge (vì `contradicts` không tạo dependency — hai claim verify độc lập).

Mỗi primitive claim trải qua **3 bước**:

#### Bước 1.1 — Plan search queries

LLM phân tích claim và tạo ra các query cụ thể, không search mù:

```
Claim: "GDP tăng 5.8%"

LLM plan:
  - "Vietnam GDP growth rate 2024 official"
  - "tăng trưởng GDP Việt Nam 2024 Tổng cục Thống kê"
  - "Vietnam annual GDP statistics GSO"
```

#### Bước 1.2 — Retrieve evidence

Gọi external knowledge sources, lấy các đoạn liên quan:

```
Retrieved:
  [GSO] "Tổng sản phẩm trong nước năm 2024 tăng 7.09% so với năm trước"
  [World Bank] "Vietnam GDP growth 2024: 7.1% (estimate)"
  [IMF] "Vietnam 2024 GDP forecast revised to 6.5%"
```

#### Bước 1.3 — LLM judge

LLM nhận claim + toàn bộ evidence → phán xét:

```
Claim: "GDP tăng 5.8%"
Evidence: GSO: 7.09%, World Bank: 7.1%, IMF forecast: 6.5%

LLM reasoning:
  "Tất cả nguồn đáng tin đều cho con số cao hơn 5.8%.
   Con số 5.8% không khớp với bất kỳ nguồn chính thức nào.
   Claim này sai."

→ Verdict: FALSE
→ Evidence: {GSO: 7.09%, World Bank: 7.1%}
→ Source links: [...]
```

**Possible verdicts sau Pass 1:**

| Verdict | Khi nào |
|---|---|
| `TRUE` | Evidence xác nhận rõ ràng, các nguồn đồng thuận |
| `FALSE` | Evidence mâu thuẫn rõ ràng với claim |
| `DISPUTED` | Các nguồn đáng tin cho kết quả xung đột nhau |
| `UNVERIFIABLE` | Không tìm được evidence đủ tin cậy |

---

### Pass 2 — Cascade qua `presupposes` edges

Sau khi Pass 1 có verdict của tất cả primitive claims, xử lý derived claims theo topological order.

**Quy tắc cascade:**

```
Nếu parent = FALSE hoặc DISPUTED:
    → child = WEAKENED
    → Lý do: premise của child không đứng vững
    → Không gọi thêm API nào

Nếu parent = TRUE:
    → child đi vào Pass 3 (verify inference validity)

Nếu parent = UNVERIFIABLE:
    → child = WEAKENED
    → Không thể verify child khi premise chưa xác định
```

**Ví dụ:**

```
atom_02: "GDP tăng 5.8%" = FALSE (Pass 1)

atom_03: "GDP 5.8% chứng tỏ kinh tế phục hồi"
    presupposes atom_02 = FALSE
    → atom_03 = WEAKENED
    → Không cần search thêm

atom_04: "Vì kinh tế phục hồi, tài khóa mở rộng đúng đắn"
    presupposes atom_03 = WEAKENED
    → atom_04 = WEAKENED
    → Cascade tiếp tục xuống cây
```

Toàn bộ cây con bên dưới một FALSE claim bị WEAKENED — không tốn thêm bất kỳ API call nào.

---

### Pass 3 — Inference validity check

Chỉ chạy với derived claims có **tất cả** `presupposes` parents = TRUE.

Đây là bước phân biệt B4 với flat fact-checkers. Không hỏi "con số đúng không?" (đã biết là đúng từ Pass 1) mà hỏi **"kết luận rút ra từ đó có hợp lý không?"**

#### Ví dụ chi tiết

```
atom_02: "GDP tăng 5.8%" = TRUE  (giả sử đúng)
atom_03: "GDP 5.8% chứng tỏ kinh tế đang phục hồi mạnh"
    presupposes atom_02 = TRUE → đi vào Pass 3

Bước 3.1 — Plan:
    "Tôi cần kiểm tra: GDP tăng 5.8% có đủ để
     kết luận kinh tế 'phục hồi mạnh' không?
     Cần tìm: tiêu chí xác định kinh tế phục hồi,
     vai trò của GDP trong đó, chỉ số nào khác cần thiết."

Bước 3.2 — Retrieve:
    [IMF] "Economic recovery requires sustained improvement
           in GDP, employment, and consumer spending
           over multiple consecutive quarters"
    [World Bank] "Single GDP figure is insufficient to
                  declare economic recovery; must consider
                  inflation, trade balance, employment rate"

Bước 3.3 — LLM judge:
    "GDP 5.8% là tín hiệu tích cực nhưng theo IMF và World Bank,
     tuyên bố 'kinh tế phục hồi mạnh' từ một chỉ số GDP duy nhất
     là overstated. Speaker đã dùng số thật để rút kết luận
     quá mức so với những gì con số đó thực sự chứng minh."

→ Verdict: MISLEADING
→ Premise: TRUE (GDP 5.8% đúng)
→ Inference: INVALID (kết luận vượt quá bằng chứng)
```

Flat fact-checker sẽ đánh `atom_03 = TRUE` (vì GDP đúng). B4 phát hiện được `MISLEADING` — đây là contribution chính của Pass 3.

---

### Pass 4 — Contradiction detection

Sau khi tất cả claims đã có verdict, scan toàn bộ `contradicts` edges:

**Quy tắc:**

```
Nếu cả hai endpoints = TRUE:
    → flag CONTRADICTION
    → Speaker phát ngôn hai điều xung đột, cả hai đều được xác nhận

Nếu một hoặc cả hai ≠ TRUE:
    → Không phải CONTRADICTION (một bên sai → chỉ là lỗi, không mâu thuẫn)
```

**Ví dụ:**

```
atom_01: "Thất nghiệp Q1/2024 giảm còn 2.3%"
    → Pass 1 → TRUE (ILO: 2.28% Q1/2024 ✓)

atom_05: "Thất nghiệp vẫn cao ở mức 5.6%" (contradicts atom_01)
    → Pass 1 → TRUE (ILO: Q4/2023: 5.61% ✓ — năm khác)

→ Pass 4: cả hai TRUE, contradicts edge tồn tại
→ flag CONTRADICTION
→ Lý do: speaker dùng hai con số từ hai thời điểm khác nhau
          như thể chúng cùng mô tả tình trạng hiện tại,
          tạo ra hình ảnh mâu thuẫn
```

---

### Tổng quan flow B4

```
┌──────────────────────────────────────────────────────────────┐
│ Input: standalone claims + DAG                               │
│                                                              │
│  PASS 1 ─ Primitive claims (+ claims chỉ có contradicts)    │
│    Bước 1: Plan search queries                               │
│    Bước 2: Retrieve external evidence                        │
│    Bước 3: LLM judge                                         │
│    → TRUE / FALSE / DISPUTED / UNVERIFIABLE                  │
│                        ↓                                     │
│  PASS 2 ─ Cascade qua presupposes edges                     │
│    parent = FALSE/DISPUTED/UNVERIFIABLE → child = WEAKENED   │
│    parent = TRUE → child đi tiếp Pass 3                      │
│                        ↓                                     │
│  PASS 3 ─ Inference validity (chỉ derived còn "sống")       │
│    Plan → Retrieve expert sources → LLM judge inference      │
│    → TRUE / MISLEADING / DISPUTED                            │
│                        ↓                                     │
│  PASS 4 ─ Contradiction scan                                 │
│    contradicts edges: cả hai TRUE → CONTRADICTION            │
│                                                              │
│ Output: per-claim verdict + evidence JSON                    │
└──────────────────────────────────────────────────────────────┘
```

---

### Full verdict space

| Verdict | Giai đoạn phát hiện | Ý nghĩa |
|---|---|---|
| `TRUE` | Pass 1 hoặc Pass 3 | Xác nhận bởi external knowledge |
| `FALSE` | Pass 1 | Bác bỏ bởi external knowledge |
| `DISPUTED` | Pass 1 hoặc Pass 3 | Các nguồn đáng tin xung đột nhau |
| `UNVERIFIABLE` | Pass 1 | Không tìm được evidence đủ tin cậy |
| `WEAKENED` | Pass 2 | Dependency chain chứa FALSE/DISPUTED/UNVERIFIABLE |
| `MISLEADING` | Pass 3 | Premise đúng, inference vượt quá bằng chứng |
| `CONTRADICTION` | Pass 4 | Speaker tự mâu thuẫn, cả hai claim đều được xác nhận |

---

### External knowledge sources

| Nguồn | Dùng cho |
|---|---|
| Wikipedia API | Sự kiện lịch sử, nhân vật, khái niệm |
| Google/Bing Search | Tin tức, báo cáo chính phủ, số liệu gần đây |
| Wolfram Alpha | Tính toán số học, so sánh con số |

B4 chỉ gọi API khi thực sự cần — claims bị WEAKENED bỏ qua hoàn toàn, tiết kiệm đáng kể số lượng external calls với video dài nhiều claim.

---

## 8. Module B5 — VLM Final Synthesis

**Input**: Track A evidence package + Track B evidence package  
**Output**: unified explainable report + structured verdict fields

### Hai track là độc lập

Track A và Track B trả lời hai câu hỏi khác nhau và không có quan hệ nhân quả bắt buộc:

```
Track A: "Video này có bị làm giả không?"     → deepfake detection
Track B: "Nội dung phát ngôn có đúng không?"  → fact-checking
```

VLM **không bị force** phải tìm relationship giữa hai track — nó tổng hợp cả hai để người dùng có đủ thông tin.

### Evidence format cho VLM

**Track A** (từ Module 5 hiện tại):
- Sampled frames (ảnh)
- Anomaly scores per chunk
- Feature values với CRITICAL/NORMAL flags
- Per-chunk reasoning

**Track B** (từ B4):
```json
{
  "claims": [
    {
      "id": "atom_02",
      "text": "GDP tăng 5.8%",
      "timestamp": "4:32",
      "verdict": "FALSE",
      "source": "GSO 2024: GDP thực tế 2.1%",
      "type": "primitive"
    },
    {
      "id": "atom_03",
      "text": "GDP 5.8% chứng tỏ kinh tế phục hồi mạnh",
      "timestamp": "7:23",
      "verdict": "WEAKENED",
      "depends_on": ["atom_02"],
      "edge_type": "presupposes"
    }
  ],
  "dag_summary": {
    "total_claims": 8,
    "false": 2,
    "misleading": 1,
    "weakened": 3,
    "true": 2
  }
}
```

### Output của B5

```
[Video Analysis]
VIDEO: FAKE  |  AUDIO: REAL  |  Confidence: 87%

[Deepfake Analysis — Track A]
→ Chunk 3 (0:08-0:12): blending flicker CRITICAL, iris jitter CRITICAL
→ Giọng nói: jitter/shimmer bình thường

[Fact-checking — Track B]
→ "GDP tăng 5.8%" (4:32): SAI — GSO 2024 ghi nhận 2.1%
→ "Kinh tế đang phục hồi mạnh" (7:23): WEAKENED — xây trên tiền đề sai
→ "Chính sách tài khóa mở rộng là đúng" (9:15): WEAKENED

[VLM Reasoning]
"Video có dấu hiệu face-swap tại đoạn 0:08-0:12. Đồng thời,
nội dung phát ngôn chứa số liệu GDP sai (5.8% thay vì 2.1%)
được dùng làm cơ sở cho các khẳng định tiếp theo về kinh tế
và chính sách. Đây là video deepfake kết hợp thông tin sai lệch."
```

---

## 9. Application Layer (UI)

Đây là **explainability contribution** của hệ thống — không paper nào có DAG visualization tương tác cho spoken fact-checking.

### Tab 1 — Video Analysis

```
[Upload video]
┌──────────────────────────────────┐
│  VIDEO: FAKE  │  AUDIO: REAL     │
│  Confidence: 87%                 │
├──────────────────────────────────┤
│  VLM Reasoning:                  │
│  "Video có dấu hiệu face-swap..."│
└──────────────────────────────────┘
```

### Tab 2 — Track A Metrics

- Anomaly score timeline (chart theo chunks)
- Feature breakdown: blending, iris jitter, audio artifacts
- Top-K annotated frames

### Tab 3 — Track B Claim DAG (Timeline)

```
Timeline (x-axis = thời gian video):

0:00         4:32        7:23        9:15
  │           │            │           │
[✅ atom_01] [❌ atom_02]  [🔶 atom_03] [🔶 atom_04]
              └────────────►           │
                           └───────────►

Click atom_02 → popup:
┌────────────────────────────────┐
│ "GDP tăng 5.8%"                │
│ Verdict: FALSE                 │
│ Timestamp: 4:32                │
│ Source: GSO 2024 (2.1%)        │
│ [Link nguồn]                   │
└────────────────────────────────┘
```

Node colors: TRUE=xanh, FALSE=đỏ, MISLEADING=cam, WEAKENED=xám, DISPUTED=vàng  
Edge labels: presupposes / contradicts / refines

---

## 10. Novelty Summary

### Algorithmic novelty (B3-B4)

1. **Inter-claim dependency graph**: hệ thống đầu tiên nối các claim với nhau theo thời gian và cascade verdict. Không paper nào làm điều này — kể cả Trification (intra-claim only), LiveFC (sentence-by-sentence), AVeriTeC (1 claim/video).

2. **Extension từ text sang speech**: ACL 2024 và Trification đều làm trên written text. Speech có đặc thù: nhiều coreference hơn, không có document structure, reference xa hơn.

3. **Labeled DAG với cascade rules khác nhau theo edge type**: premise validity vs inference validity tách biệt → phát hiện MISLEADING (dùng số thật để rút kết luận lệch).

4. **Acyclicity tự nhiên từ temporal order**: không cần enforce — emerge từ bản chất của speech.

### Systems novelty (B5 + UI)

5. **Interactive DAG timeline visualization**: hệ thống đầu tiên hiển thị inter-claim dependency graph theo timeline video — có thể click từng node để xem verdict và source.

6. **Joint deepfake + fact-checking report**: hệ thống đầu tiên kết hợp signal-level evidence (Track A) và semantic evidence (Track B) trong một explainable output.

---

## 11. Dataset Strategy

Không có dataset nào hiện tại có đồng thời deepfake label lẫn claim truthfulness label — đây là known gap.

### Track A

| Dataset | Role | Ghi chú |
|---|---|---|
| **MAVOS-DD** (test split) | Primary eval | Đã làm, published CSoNet 2026. Single-speaker AV deepfake. |
| **FakeAVCeleb** | Optional — generalization test | AV deepfake có face swap + voice clone; gần nhất với loại fake Track A detect. Không bắt buộc trong thesis. |

### Track B

| Dataset | Role | Có nhãn gì | Eval được gì |
|---|---|---|---|
| **MAD2** ([2606.11420](https://arxiv.org/abs/2606.11420)) | Primary eval | Checkworthiness/câu, veracity/câu, veracity/dialogue | Claim detection (B2) ✅, Individual verification (B4 Pass 1) ✅, DAG/cascade ❌ |
| **MAD** ([2508.12186](https://arxiv.org/abs/2508.12186)) | Reference/comparison | Như MAD2 nhưng ít hơn (600 dialogues) | Dùng để so sánh với MAD2 |
| **AVeriTeC** ([2305.13117](https://arxiv.org/abs/2305.13117)) | Secondary | Claim + web evidence + verdict | Chỉ dùng cho B4 retrieval eval; text-based, không có audio |
| **Mini curated dataset** (~50-100 videos) | DAG/cascade eval | Tự annotate inter-claim dependency | DAG structure ✅, Cascade verdict ✅, WEAKENED/MISLEADING ✅ |

**Lý do cần mini curated dataset**: MAD/MAD2 có nhãn veracity từng claim độc lập nhưng **không annotate inter-claim dependency** — không biết "claim B presupposes claim A". Đây là phần novel nhất của Track B (B3 + B4 Pass 2-3) nên cần tự tạo ground truth cho ~50 videos để eval cascade verdict.

Dataset gap (MAD/MAD2 không có dependency labels) = limitation cần thừa nhận; tạo joint benchmark với dependency annotation là future work.

**LIAR Dataset** — nguồn gốc nhãn của MAD2:
- MAD2 claim kế thừa nhãn TRUE/FALSE từ LIAR (12k câu phát ngôn chính trị, do nhà báo PolitiFact gán nhãn thủ công)
- LIAR public tại: https://huggingface.co/datasets/liar
- Mỗi LIAR entry có PolitiFact URL → truy được **full explanation + evidence sources** mà fact-checker đã dùng
- **Dùng về sau**: nếu muốn eval chất lượng retrieval của B4 trên MAD2, map claim → LIAR ID → scrape PolitiFact article → dùng làm ground-truth evidence. Chưa làm trong thesis scope hiện tại, để dành làm extension.

---

## 12. Infrastructure

- Pipeline chủ yếu **tuần tự** → chỉ cần VRAM cho model lớn nhất tại một thời điểm
- Model lớn nhất: Qwen3-VL-8B ≈ **8-10GB** với 4-bit quantization
- **1× GPU 24GB VRAM đủ** (RTX 3090 / A10G trên VastAI ~$0.25-0.35/hr)
- ASR + VSR có thể chạy song song (~3.5GB combined)
- Track B LLMs (B2/B3/B4): dùng Qwen3-4B (~3-4GB) để tiết kiệm VRAM
