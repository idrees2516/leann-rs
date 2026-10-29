#!/usr/bin/env bash
# leann-rs REST API demo — a five-minute tour of the server.
#
#   ./rest-demo.sh [port]
#
# Requires: the `leann` binary on PATH (cargo build --release -p leann-cli).
set -euo pipefail

PORT="${1:-8199}"
KEY="demo-key"
DIR="$(mktemp -d)"
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

export LEANN_EMBEDDER=local   # offline embedder; unset + OPENAI_API_KEY = OpenAI

echo ">> starting server on :$PORT (data in $DIR)"
leann serve --port "$PORT" --data-dir "$DIR/data" --api-key "$KEY" &
SERVER_PID=$!
until curl -sf "http://127.0.0.1:$PORT/health" -H "x-api-key: $KEY" >/dev/null 2>&1; do sleep 0.2; done

H1="x-api-key: $KEY"
H2="content-type: application/json"

echo ">> create a collection"
curl -s -H "$H1" -H "$H2" -X POST "http://127.0.0.1:$PORT/collections" -d '{"name":"demo"}'

echo ">> add some texts"
curl -s -H "$H1" -H "$H2" -X POST "http://127.0.0.1:$PORT/collections/demo/texts" -d '{
  "texts": [
    "LEANN stores no raw embeddings: PQ codes compress vectors about 96x.",
    "HNSW graphs give sub-linear nearest neighbor search over chunks.",
    "The IBM Granite Guardian audits answers for groundedness.",
    "QR code video memory packs the corpus into a portable MP4."
  ],
  "metadatas": [{"topic": "pq"}, {"topic": "graph"}, {"topic": "guard"}, {"topic": "visual"}]
}'

echo ">> hybrid search (vector + BM25 + RRF)"
curl -s -H "$H1" -H "$H2" -X POST "http://127.0.0.1:$PORT/collections/demo/hybrid" \
     -d '{"query": "how are vectors compressed", "k": 2}' | head -c 400
echo

echo ">> build + query the QR-video memory over HTTP"
curl -s -H "$H1" -H "$H2" -X POST "http://127.0.0.1:$PORT/collections/demo/visual" -d '{"format":"gif"}'
echo
curl -s -H "$H1" -H "$H2" -X POST "http://127.0.0.1:$PORT/collections/demo/visual-search" \
     -d '{"query": "guardian groundedness", "k": 1}' | head -c 400
echo

echo ">> export as CSV"
curl -s -H "$H1" "http://127.0.0.1:$PORT/collections/demo/export?format=csv" -o "$DIR/demo.csv"
head -2 "$DIR/demo.csv"

echo ">> open the playground: http://127.0.0.1:$PORT/  (API key: $KEY)"
echo ">> press Ctrl-C to stop"
wait
