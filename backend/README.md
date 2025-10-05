# Backend for CFG Visualization Tool

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/parse` - Parse Solidity code (currently reserved for future use)
