// Native module loader for leann-node
// The binary is built with `napi build --platform [--release]` or
// `cargo build -p leann-node --release` (then copy target/release/libleann_node.so
// to leann.<platform>.node next to this file).

const { existsSync } = require('fs')
const { join } = require('path')

const candidates = [
  join(__dirname, 'leann.linux-x64-gnu.node'),
  join(__dirname, 'leann.linux-x64-musl.node'),
  join(__dirname, 'leann.darwin-arm64.node'),
  join(__dirname, 'leann.darwin-x64.node'),
  join(__dirname, 'leann.win32-x64-msvc.node'),
  join(__dirname, 'leann.node'),
  // cargo build output (dev convenience)
  join(__dirname, '..', 'target', 'release', 'libleann_node.so'),
]

for (const p of candidates) {
  if (existsSync(p)) {
    module.exports = require(p)
    break
  }
}

if (!module.exports) {
  module.exports = new Proxy(
    {},
    {
      get() {
        throw new Error(
          'leann native binary not found. Build it with `napi build --platform --release` or `cargo build -p leann-node --release`.',
        )
      },
    },
  )
}
