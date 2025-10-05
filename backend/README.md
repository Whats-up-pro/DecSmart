# Backend for CFG Visualization Tool

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node helper dependencies (first run only):
```bash
cd node_helper && npm ci
```

3. Run the server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/cfg` - Build CFG from Solidity code using Node helper

### Example request

```bash
curl -X POST http://localhost:5000/api/v1/cfg \
  -H "Content-Type: application/json" \
  -d '{"code":"pragma solidity ^0.8.0; contract C { function f() public { if (true) { } } }"}'
```

## Architecture

- Flask app factory in `app.py`; API routes live in `api/routes.py`.
- CFG generation is delegated to a Node script under `node_helper/` that uses `@solidity-parser/parser`.
- This keeps frontend parsing logic optional while enabling server-side generation for consistency and sharing.
