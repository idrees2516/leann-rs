# Examples

| Example | What it shows |
|---|---|
| [`rest-demo.sh`](rest-demo.sh) | Full REST API tour: start the server, create a collection, add texts, hybrid search, build + query the QR-video memory, CSV export. |
| [`../scripts/e2e_self.sh`](../scripts/e2e_self.sh) | Real-data end-to-end: the repository ingests **its own source code** (356 chunks), searches it, builds a lossless MP4 memory, retrieves text from video frames, and drives the server with auth/rate-limit checks. |
| `quickstart` (`cargo run --release -p leann-core --example quickstart`) | Rust API quickstart: build → search → hybrid → save. |
| [`../npm`](../npm) | TypeScript bindings with a full `.d.ts`; see the [README](../npm/README.md) for `addFiles` / `hybridSearch` / `chat` / `localGrounding` / `exportCorpus` / `importJson`. |
