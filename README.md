# AI Comment Moderator

Automatic comment moderation for [PropertyTribes](https://www.propertytribes.com/) using the Anthropic Claude API. Supports an appeal flow for rejected comments, per-user rate limiting, webhook notifications for flagged content, and a human admin override endpoint.

---

## Setup (< 5 minutes)

### Prerequisites
- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone and install

```bash
git clone https://github.com/Stefanroodt/comment-moderator.git
cd comment-moderator
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
# Optionally add WEBHOOK_URL for flagged content notifications
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

Rejection categories: `spam` · `hate_speech` · `misinformation` · `off_topic` · `abusive` · `promotional`

Rate limited to **30 requests per user per minute**.

---

### `POST /appeal`

Appeal a rejected comment. Requires the `comment_id` from the original moderation response.

```bash
curl -X POST http://localhost:8000/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "appeal_context": "I am a RICS-qualified surveyor with 15 years of experience. My comment was professional advice, not spam."
  }'
```

**Response:**
```json
{
  "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "original_decision": "rejected",
  "appeal_decision": "approved",
  "reasoning": "The user's professional credentials clarify that this was genuine expert advice. Appeal upheld.",
  "timestamp": "2024-11-15T10:35:00Z"
}
```

Notes:
- Appeals are only allowed for `rejected` comments
- Each comment may only be appealed once
- Final decisions are `approved` or `rejected` only (no further appeals)
- Rate limited to **5 appeals per user per 10 minutes**

---

### `GET /log`

Retrieve all moderation decisions, most recent first. Supports pagination.

```bash
curl "http://localhost:8000/log?page=1&limit=20"
```

Each entry includes: `comment_id`, `user_id`, `comment`, `decision`, `confidence`, `reasoning`, `rejection_category`, `timestamp`, `appealed`, and — if appealed — full appeal details. Admin overrides are also recorded in the log.

---

### `PATCH /log/{comment_id}`

**Admin endpoint.** Override an AI moderation decision with a human judgement.

Typical use: a moderator reviews a `flagged_for_review` comment and makes a final call.

```bash
curl -X PATCH http://localhost:8000/log/3fa85f64-5717-4562-b3fc-2c963f66afa6 \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "approved",
    "note": "Verified RICS credentials — legitimate professional advice."
  }'
```

The original AI decision is preserved alongside the override for a full audit trail.

---

## Project Structure

```
comment-moderator/
├── main.py          # FastAPI app, route handlers
├── moderator.py     # Async Claude prompt construction and API calls
├── models.py        # Pydantic request/response/log models
├── storage.py       # Thread-safe in-memory moderation log
├── rate_limiter.py  # Per-user sliding window rate limiter
├── webhook.py       # Webhook delivery for flagged content
├── requirements.txt
├── .env.example
└── tests/
    └── test_api.py  # pytest suite — 28 tests, mocked Claude
```

---

## Key Design Decisions

### Async Claude calls
All Claude API calls use `AsyncAnthropic`, making them fully non-blocking. The FastAPI event loop remains free to handle other incoming requests while waiting for the AI response — important under any real load, since Claude calls typically take 1–3 seconds each.

### Prompt design
The system prompt embeds specific PropertyTribes context — the forum's topic focus (UK property investment, landlord/tenant law, HMOs), what should be approved versus rejected, and what warrants human review. This is more reliable than a generic moderation prompt because Claude has a concrete reference frame for what's on-topic and what constitutes harm in this specific community.

### Appeal genuineness
The appeal prompt explicitly instructs Claude **not** to simply repeat the original decision, and lists specific questions to consider (does the context clarify intent? does it provide credentials?). This prevents the common failure mode where an appeal system is cosmetically different but functionally identical to the original moderation.

### Three-outcome initial moderation
Using `flagged_for_review` alongside `approved`/`rejected` avoids over-automation. Borderline content routes to a human rather than making a confident wrong call. The admin override endpoint (`PATCH /log/{comment_id}`) closes this loop — moderators can action flagged content and their decision is recorded in the audit log.

### JSON extraction robustness
Claude occasionally wraps JSON in markdown fences. The `_extract_json()` function handles both fenced and raw JSON, and all field parsing falls back gracefully — unknown decision values default to `flagged_for_review`, unknown categories default to `none`, invalid confidence values default to `0.5`.

### Per-user rate limiting
Rate limiting is keyed on `user_id` from the request body (not IP address), which is more accurate in environments where many users share an IP (e.g. corporate NAT). Uses a sliding window algorithm — limits are 30 moderate requests/minute and 5 appeal requests/10 minutes per user. A secondary IP-based limit via `slowapi` acts as a hard ceiling.

### Separation of concerns
`moderator.py` only knows about AI logic. `main.py` only knows about HTTP. `storage.py` only knows about data. This makes each layer independently testable and swappable — e.g., replacing Claude with another LLM requires changing only `moderator.py`.

---

## Bonus Features Included

- **Async AI calls** — `AsyncAnthropic` keeps the event loop non-blocking under load
- **Rate limiting** — per-user sliding window (30 req/min for moderation, 5/10min for appeals) plus IP-based ceiling
- **Rejection categorisation** — `spam`, `hate_speech`, `misinformation`, `off_topic`, `abusive`, `promotional`
- **Webhook on flagged content** — configurable `WEBHOOK_URL` receives a POST when content is flagged; use [webhook.site](https://webhook.site) to test
- **Admin override endpoint** — `PATCH /log/{comment_id}` for human moderators to make final calls on flagged content
- **Paginated log** — `GET /log?page=1&limit=20`
- **Unit tests** — 28 pytest tests covering happy paths, edge cases, rate limiting, and webhook behaviour

---

## Assumptions

- Comments are plain text (no HTML/markdown parsing required)
- User identity is provided by the caller — there is no authentication layer
- "In-memory" means the log resets on server restart, as per the spec
- The `user_id` field is used for rate-limit attribution but not validated against a user database

---

## What I'd Improve With More Time

1. **Persistent storage** — swap the in-memory dict for SQLite or PostgreSQL with a migration layer (Alembic)
2. **Authentication** — API key or JWT on every endpoint; the admin override especially should be protected
3. **Confidence calibration** — log Claude's decisions alongside human override outcomes and use that data to tune prompts over time
4. **Streaming responses** — stream Claude's reasoning for faster perceived latency on long responses
5. **Appeal expiry** — appeals should time out after e.g. 7 days so old rejections can't be reopened indefinitely
6. **Async webhook delivery** — move webhook calls to a background task so they don't add any latency to the main response
