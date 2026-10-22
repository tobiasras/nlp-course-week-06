# NLP course week 06

FastAPI backend on port **8000**. Frontend on port **8001**.

Add course data to `data/`:

- `dtu_courses.jsonl`
- `dtu_courses.warc.gz`


## Run

```bash
docker compose up --build
```

- API docs: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:8001

```bash
curl -s http://localhost:8000/
```

## Tests

```bash
python -m pytest test/backend/test_main.py
```
