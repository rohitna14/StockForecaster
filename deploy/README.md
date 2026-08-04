# Deployment

Every component runs on a free tier.

| Component | Host | Free tier |
|---|---|---|
| Postgres | [Neon](https://neon.tech) | 0.5 GB, autosuspends |
| Redis | [Upstash](https://upstash.com) | 10k commands/day |
| API | [Hugging Face Spaces](https://huggingface.co/spaces) (Docker) or [Fly.io](https://fly.io) | 2 vCPU / 16 GB |
| Next.js | [Vercel](https://vercel.com) | Hobby |
| Streamlit | Hugging Face Spaces | 2 vCPU / 16 GB |
| Parquet lake | [Cloudflare R2](https://developers.cloudflare.com/r2/) | 10 GB + free egress |
| Scheduled jobs | GitHub Actions | free for public repos |

## 1. Database (Neon)

```bash
# Create a project, then:
export FORECASTER_DATABASE_URL="postgresql+asyncpg://user:pass@ep-xxx.neon.tech/forecaster"
cd backend && alembic upgrade head
python -m forecaster.cli seed
python -m forecaster.cli ingest --universe demo --hot --years 5
```

Neon autosuspends after inactivity, so the engine is configured with
`pool_pre_ping=True` and a 300s recycle — stale connections after a wake are
expected, not a bug.

## 2. API (Hugging Face Spaces)

Create a **Docker** Space, push this repo, and set these as Space secrets:

```
FORECASTER_DATABASE_URL
FORECASTER_ENVIRONMENT=production
FORECASTER_LOG_FORMAT=json
FORECASTER_API_CORS_ORIGINS=https://your-app.vercel.app
```

`Dockerfile` at the Space root should be `backend/Dockerfile`. The image
deliberately excludes TensorFlow — sequence models are trained offline, not in
a request path.

> Production refuses to start against SQLite or with `debug=True`. That guard is
> in `config.py` and is intentional.

## 3. Frontend (Vercel)

Import the repo, set the root directory to `frontend/`, and add:

```
NEXT_PUBLIC_API_URL=https://your-space.hf.space/api/v1
```

## 4. Streamlit (Hugging Face Spaces)

A second Space, SDK `docker`, using `streamlit_app/Dockerfile`:

```
FORECASTER_API_URL=https://your-api-space.hf.space/api/v1
```

## 5. Scheduled jobs

`.github/workflows/ingest-nightly.yml` runs weekdays after the close. Add
`DATABASE_URL` (and any provider keys) as repository secrets.

---

## Cost-control notes

- **Hot tier is capped** at `FORECASTER_HOT_TIER_MAX_SYMBOLS` (default 800) so a
  runaway ingest cannot exceed the free Postgres allowance.
- **Evaluations are CPU-bound.** The API runs them on a 2-worker thread pool;
  a large model set should go through `POST /runs/async` rather than the
  synchronous endpoint.
- **Provider quotas** are enforced by token buckets and persisted in `api_quota`,
  so a restart does not reset a daily allowance.
