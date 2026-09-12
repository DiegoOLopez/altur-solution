# Endpoint checks

`check_endpoint.py` sends dataset calls to `/detect` using the JSON contract expected by the judge and calculates accuracy, balanced accuracy, AUC, Brier score and latency.

The script uses these defaults relative to `backend/`:

- Manifest: `backend/manifest.csv`
- Audio directory: `backend/audio/`
- Split: `val`
- Number of calls: `10`
- Timeout: `30` seconds per call

## Run

From `backend/`:

```bash
python scripts/check_endpoint.py --url http://localhost:8000/detect
```

Examples:

```bash
python scripts/check_endpoint.py --url http://localhost:8000/detect --n 20 --split val
python scripts/check_endpoint.py --url http://localhost:8000/detect --split all --n 0
python scripts/check_endpoint.py --url http://localhost:8000/detect --manifest ../manifest.csv --audio-dir ../audio --out results.json
```

The script uses only the Python standard library. It does not start the backend; start the API separately before running the check.
