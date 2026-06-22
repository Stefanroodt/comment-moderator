# AI Comment Moderator

Automatic comment moderation for [PropertyTribes](https://www.propertytribes.com/) using the Anthropic Claude API. Supports an appeal flow for rejected comments.

---

## Setup (< 5 minutes)

### Prerequisites
- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd comment-moderator
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run

```bash
uvicorn main:app --reload
```

The API is live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 4. Run tests

```bash
pytest tests/ -v
```

Tests mock the Claude API — no key required, runs in ~2 seconds.

---

## API Reference

### `POST /moderate`

Submit a comment for AI moderation.

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_abc123",
    "comment": "Has anyone had experience with HMO licensing in Manchester? I have a 5-bed property and am unsure about Article 4 directions."
  }'
```

**Response:**
```json
{
  "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "decision": "approved",
  "confidence": 0.97,
  "reasoning": "Genuine HMO licensing question relevant to the PropertyTribes community.",
  "rejection_category": "none",
  "timestamp": "2024-11-15T10:30:00Z"
}
```

Possible decisions: `approved` · `rejected` · `flagged_for_review`

---

### `POST /appeal`

Appeal a rejected comment. Requires the `comment_id` from the original moderation response.

```bash
curl -X POST http://localhost:8000/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "appeal_context": "I am a RICS-qualified surveyor. My comment was professional advice drawn from 15 years of practice, not spam."
  }'
```

**Response:**
```json
{
  "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "original_decision": "rejected",
  "appeal_decision": "approved",
  "reasoning": "The user's professional credentials clarify that this was genuine expert advice, not promotional content. Appeal upheld.",
  "timestamp": "2024-11-15T10:35:00Z"
}
```

Notes:
- Appeals are only allowed for `rejected` comments
- Each comment may only be appealed once
- Final decisions are `approved` or `rejected` only (no further appeals)

---

### `GET /log`

Retrieve all moderation decisions, most recent first.

```bash
curl http://localhost:8000/log
```

Each entry includes: `comment_id`, `user_id`, `comment`, `decision`, `confidence`, `reasoning`, `rejection_category`, `timestamp`, `appealed`, and — if appealed — `appeal_context`, `appeal_decision`, `appeal_reasoning`, `appeal_timestamp`.

---

## Project Structure

```
comment-moderator/
├── main.py          # FastAPI app, route handlers
├── moderator.py     # Claude prompt construction and API calls
├── models.py        # Pydantic request/response/log models
├── storage.py       # Thread-safe in-memory moderation log
├── requirements.txt
├── .env.example
└── tests/
    └── test_api.py  # pytest suite (mocked Claude)
```

---

## Key Design Decisions

### Prompt design
The system prompt embeds specific PropertyTribes context — the forum's topic focus (UK property investment, landlord/tenant law, HMOs), what should be approved versus rejected, and what warrants human review. This is more reliable than a generic moderation prompt because Claude has a concrete reference frame for what's on-topic and what constitutes harm in this specific community.

### Appeal genuineness
The appeal prompt explicitly instructs Claude **not** to simply repeat the original decision, and lists specific questions to consider (does the context clarify intent? does it provide credentials?). This prevents the common failure mode where an appeal system is cosmetically different but functionally identical to the original moderation.

### Three-outcome initial moderation
Using `flagged_for_review` alongside `approved`/`rejected` avoids over-automation. Borderline content (e.g., borderline self-promotion, ambiguous legal claims) routes to a human rather than making a confident wrong call. Appeals, by contrast, are binary — a human has already reviewed the rejection implicitly by the time an appeal is filed.

### JSON extraction robustness
Claude occasionally wraps JSON in markdown fences. The `_extract_json()` function handles both fenced and raw JSON, and all field parsing falls back gracefully — unknown decision values default to `flagged_for_review`, unknown categories default to `none`, invalid confidence values default to `0.5`.

### Separation of concerns
`moderator.py` only knows about AI logic. `main.py` only knows about HTTP. `storage.py` only knows about data. This makes each layer independently testable and swappable — e.g., replacing Claude with another LLM requires changing only `moderator.py`.

---

## Bonus Features Included

- **Rate limiting** — 30 moderate requests/minute and 10 appeal requests/minute per IP (via `slowapi`)
- **Rejection categorisation** — `spam`, `hate_speech`, `misinformation`, `off_topic`, `abusive`, `promotional`
- **Unit tests** — full pytest suite with mocked Claude; tests edge cases, error paths, and the double-appeal guard

---

## Assumptions

- Comments are plain text (no HTML/markdown parsing required)
- User identity is provided by the caller — there is no authentication layer
- "In-memory" means the log resets on server restart, as per the spec
- The `user_id` field is used for rate-limit attribution but not validated against a user database

---

## What I'd Improve With More Time

1. **Persistent storage** — swap the in-memory dict for SQLite or PostgreSQL; add a migration layer
2. **Webhook on flagged content** — POST to a configurable URL when a comment is flagged for human review, so moderators get real-time notifications
3. **Async Claude calls** — use `anthropic.AsyncAnthropic` so the FastAPI event loop isn't blocked during AI inference
4. **Per-user rate limiting** — rate limit by `user_id` in addition to IP for more accurate abuse prevention
5. **Confidence calibration** — log Claude's decisions alongside eventual human override outcomes and use that data to tune prompts over time
6. **Streaming responses** — for very long reasoning, stream the Claude response to reduce perceived latency
7. **Admin endpoint** — `PATCH /log/{comment_id}` for human moderators to override flagged decisions
