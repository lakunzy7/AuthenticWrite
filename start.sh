#!/usr/bin/env bash
# AuthenticWrite — start backend and frontend together.
# Logs go to logs/backend.log and logs/frontend.log.
# Stops cleanly with Ctrl+C (kills both children) or run ./stop.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend/authenticwrite"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Sanity checks ---------------------------------------------------------
if [[ ! -x "$BACKEND_DIR/venv/bin/python" ]]; then
  echo "ERROR: backend venv missing. Run: cd backend && python3 -m venv venv && pip install -r requirements.txt" >&2
  exit 1
fi
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "ERROR: frontend node_modules missing. Run: cd frontend/authenticwrite && npm install" >&2
  exit 1
fi

# --- Refuse to start if ports already in use -------------------------------
for port in 5000 7000; do
  if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
    echo "ERROR: port $port already in use. Run ./stop.sh first or kill the process." >&2
    exit 1
  fi
done

# --- Start backend ---------------------------------------------------------
echo "Starting backend (Flask + RoBERTa detector)..."
cd "$BACKEND_DIR"
nohup ./venv/bin/python app.py >"$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$PID_DIR/backend.pid"
echo "  backend PID $BACKEND_PID — log: $LOG_DIR/backend.log"

# --- Start frontend --------------------------------------------------------
echo "Starting frontend (React on port 7000)..."
cd "$FRONTEND_DIR"
BROWSER=none nohup npm start >"$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$PID_DIR/frontend.pid"
echo "  frontend PID $FRONTEND_PID — log: $LOG_DIR/frontend.log"

cd "$ROOT_DIR"

# --- Wait for backend to bind ---------------------------------------------
echo
echo "Waiting for backend to load model and bind port 5000 (first run can take 1-2 minutes)..."
for i in $(seq 1 120); do
  if curl -sf -o /dev/null --max-time 1 http://localhost:5000/ 2>/dev/null; then
    echo "  backend ready."
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: backend died. See $LOG_DIR/backend.log" >&2
    exit 1
  fi
  sleep 1
done

# --- Wait for frontend ----------------------------------------------------
echo "Waiting for frontend to compile and bind port 7000..."
for i in $(seq 1 120); do
  if curl -sf -o /dev/null --max-time 1 http://localhost:7000/ 2>/dev/null; then
    echo "  frontend ready."
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "ERROR: frontend died. See $LOG_DIR/frontend.log" >&2
    exit 1
  fi
  sleep 1
done

echo
echo "AuthenticWrite is running:"
echo "  Frontend  http://localhost:7000"
echo "  Backend   http://localhost:5000"
echo
echo "Stop with:  ./stop.sh"
echo "Tail logs:  tail -f logs/backend.log logs/frontend.log"
