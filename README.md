# AI Comment Moderator

Comment moderation API for [PropertyTribes](https://www.propertytribes.com/) built with FastAPI and Claude. Submit a comment, get back a decision. Rejected users can appeal. Moderators can override. There's a log of everything.

---

## Setup

### Prerequisites
- Anthropic API key
- Python 3.9+ or Docker

### Docker (quickest for dev)

```bash
git clone https://github.com/Stefanroodt/comment-moderator.git
cd comment-moderator
cp .env.example .env   # add your key here
docker compose up
```

This mounts source and runs with `--reload`. For production, remove the volume mount and the reload flag.

### Python

```bash
git clone https://github.com/Stefanroodt/comment-moderator.git
cd comment-moderator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

API runs at `http://localhost:8000`. Swagger UI at `/docs`.

---

## Testing it

The quickest way is just the Swagger UI at `/docs` — fill in the fields and hit Execute.

If you want to see the full flow end to end, run the demo script (requires `jq`):

```bash
brew install jq
chmod +x demo.sh
./demo.sh
```

There's also a Postman collection (`postman_collection.json`) with all 4 endpoints pre-wired and `comment_id` auto-saved between requests.

Unit tests don't need a running server or real API key — Claude is mocked:

```bash
pytest tests/ -v
```

50 tests, runs in about a second.

---

## Endpoints

### POST /moderate

```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_abc123",
    "comment": "Has anyone dealt with Article 4 directions for HMOs in Manchester?",
    "tribe": "HMO Landlords"
  }'
```

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

The `tribe` field is optional. When provided, the moderator applies tribe-specific rules — for example, self-promotion is fine in `Wanted & Recommendations` but rejected in `HMO Landlords`, and `No Money Down (NMD)` gets higher scrutiny by default.

Decisions: `approved`, `rejected`, `flagged_for_review`

Rejection categories: `spam`, `hate_speech`, `misinformation`, `off_topic`, `abusive`, `promotional`

Rate limited to 30 requests per user per minute.

---

### POST /appeal

For rejected comments only. Each comment gets one appeal.

```bash
curl -X POST http://localhost:8000/appeal \
  -H "Content-Type: application/json" \
  -d '{
    "comment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "appeal_context": "I am a RICS-qualified surveyor — this was professional advice, not spam."
  }'
```

The AI re-evaluates the original comment alongside the appeal context. Final decision is `approved` or `rejected` — no further escalation. Rate limited to 5 per user per 10 minutes.

---

### GET /log

```bash
curl "http://localhost:8000/log?page=1&limit=20"
```

Returns all moderation decisions, newest first. If a comment was appealed or overridden by an admin, that's in there too.

---

### GET /stats

```bash
curl "http://localhost:8000/stats"
# or scope to a time window:
curl "http://localhost:8000/stats?since=2024-11-15T00:00:00Z"
```

Returns decision breakdown, average confidence, top 5 rejection categories, appeal overturn rate, and admin override count. `avg_confidence` is `null` if there's no data yet.

---

### PATCH /log/{comment_id}

Admin override — lets a human moderator set the final decision on any comment, typically ones that came back as `flagged_for_review`. The original AI decision is kept in the log alongside it.

```bash
curl -X PATCH http://localhost:8000/log/3fa85f64-5717-4562-b3fc-2c963f66afa6 \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved", "note": "Verified credentials — legitimate advice."}'
```

---

## Structure

```
comment-moderator/
├── main.py                   # routes
├── moderator.py              # Claude prompt + API calls
├── models.py                 # Pydantic models
├── storage.py                # in-memory store (thread-safe)
├── rate_limiter.py           # sliding window per user_id
├── webhook.py                # fires on flagged content
├── demo.sh                   # end-to-end demo script (requires jq)
├── postman_collection.json   # all endpoints pre-wired, comment_id auto-saved
├── Dockerfile
├── docker-compose.yml
└── tests/
    └── test_api.py
```

---

## Design notes

**Why async Claude calls?** FastAPI is async but if you call the Claude API with the regular blocking client it ties up the event loop for the 1-2 seconds each call takes. Used `AsyncAnthropic` to avoid that.

**Prompt design** — the system prompt includes the actual PropertyTribes context: the 40+ tribes, what topics are on-topic, what kinds of comments the community flags as problematic. A generic moderation prompt would misfire constantly on things like "Problem Tenants" discussions which look aggressive out of context but are completely normal on this forum.

**Tribe-aware moderation** — when a `tribe` is supplied, the prompt gets a tribe-specific rules block on top of the general context. Ten tribes have explicit guidance baked in: `No Money Down (NMD)` gets higher scrutiny by default, `Problem Tenants` expects harsh language and approves it, `Wanted & Recommendations` treats self-promotion as the whole point, `Scottish PRS` is reminded it runs under different legislation, and so on. Unknown tribes fall back to the general PropertyTribes guidelines.

**Three outcomes instead of two** — I went with `approved`, `rejected`, and `flagged_for_review` rather than just approve/reject. The flagged state routes borderline content to a human rather than making a confident wrong call. The admin override endpoint closes that loop.

**Rate limiting on `user_id` not IP** — IP limits break in office environments where everyone shares one address. `user_id` is more accurate. The rate limiter is a custom sliding window keyed on `user_id`; there's no per-IP layer because it would contradict this and add noise.

**The appeal prompt** — I had to explicitly tell Claude not to just repeat the original decision. Without that instruction it was basically rubber-stamping rejections. The prompt now asks it to consider whether the context genuinely changes anything.

---

## Assumptions

- Comments are plain text
- No auth layer — user identity is trusted from the request body
- In-memory storage resets on restart (as per the spec)
- `GET /log` and `GET /stats` are unauthenticated — they expose `user_id`s and comment content to anyone who can reach the API. In production these would sit behind an internal network boundary or require an API key.
- Logs do not record comment text or `user_id` at INFO level or above; only `comment_id`, decision, and confidence are emitted to limit PII exposure.

---

## What I'd change with more time

- **Persistent storage** — SQLite would be fine for this scale, Postgres if it needs to grow
- **Auth on the admin endpoint** — right now anyone can override any decision, which is obviously not fine in production
- **Background webhook delivery** — currently it fires inline which adds a small amount of latency. Easy fix with FastAPI's `BackgroundTasks`
- **Appeal expiry** — probably shouldn't allow appeals on 6-month-old rejections
- **Confidence tracking over time** — comparing AI decisions vs human overrides would give useful signal for prompt tuning
- **Redis for horizontal scaling** — the in-memory store and rate-limiter windows aren't shared across processes, so running multiple workers requires an external store
