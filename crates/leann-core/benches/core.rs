//! Criterion benchmarks for the LEANN hot paths.
//!
//! `cargo bench -p leann-core`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;

use leann_core::cache::{EmbedSource, EmbeddingProvider};
use leann_core::graph::{GraphBuildConfig, HubPruneConfig, SliceOracle};
use leann_core::index::{IndexConfig, LeannIndex, SearchOptions, SearchDefaults};
use leann_core::metric::{normalize, Metric};
use leann_core::simd::{dot, l2_sq, dot_portable, l2_sq_portable};
use leann_core::pq::{PqCodec, PqConfig};
use leann_core::search::{two_level_search, SearchParams};

fn bench_provider(dim: usize) -> Arc<dyn EmbeddingProvider> {
    struct P(usize);
    impl EmbeddingProvider for P {
        fn name(&self) -> &str {
            "bench"
        }
        fn dim(&self) -> usize {
            self.0
        }
        fn embed_batch(&self, texts: &[&str]) -> leann_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0f32; self.0];
                    let mut h = 5381u64;
                    for b in t.bytes() {
                        h = h.wrapping_mul(33) ^ b as u64;
                        v[(h as usize) % self.0] += 1.0;
                    }
                    normalize(&mut v);
                    v
                })
                .collect())
        }
    }
    Arc::new(P(dim))
}

fn corpus(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let p = bench_provider(dim);
    let topics = [
        "machine learning gradient descent neural networks training",
        "rust ownership borrow checker memory safety performance",
        "vector index approximate nearest neighbor search embeddings",
        "espresso extraction temperature roast grinding coffee",
        "planetary orbits telescopes gravity astronomy starlight",
        "pasta recipe tomato basil garlic olive oil cooking",
        "quarterly revenue forecast margins sales business",
        "mountain hiking trails forest wilderness camping gear",
    ];
    let texts: Vec<String> = (0..n)
        .map(|i| format!("doc{i} {}", topics[i % topics.len()]))
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    p.embed_batch(&refs).unwrap()
}

fn bench_pq(c: &mut Criterion) {
    let dim = 768;
    let data = corpus(5000, dim);
    let cfg = PqConfig {
        m: 32,
        train_iters: 6,
        ..Default::default()
    };
    let mut group = c.benchmark_group("pq");
    group.sample_size(10);

    group.bench_function("train_5k_x768_m32", |b| {
        b.iter(|| PqCodec::train(&data, &cfg, Metric::Cosine).unwrap())
    });

    let codec = PqCodec::train(&data, &cfg, Metric::Cosine).unwrap();
    group.bench_function("encode_one", |b| b.iter(|| codec.encode(&data[0])));
    let codes = codec.encode_batch(&data);
    group.bench_function("adc_table", |b| b.iter(|| codec.adc_table(&data[0])));
    let table = codec.adc_table(&data[0]);
    group.bench_function("adc_distance", |b| {
        b.iter(|| codec.adc_distance(&table, &codes[0]))
    });
    group.finish();
}

fn bench_graph(c: &mut Criterion) {
    let n = 5000;
    let dim = 256;
    let data = corpus(n, dim);
    let oracle = SliceOracle {
        vecs: &data,
        metric: Metric::Cosine,
    };
    let cfg = GraphBuildConfig {
        m: 32,
        m0: 64,
        ef_construction: 200,
        seed: 1,
    };
    let mut group = c.benchmark_group("graph");
    group.sample_size(10);
    group.bench_function("build_5k_m32", |b| {
        b.iter(|| leann_core::graph::HnswGraph::build_local(n, &oracle, &cfg))
    });
    let mut g = leann_core::graph::HnswGraph::build_local(n, &oracle, &cfg);
    group.bench_function("hub_prune_5k", |b| {
        b.iter(|| {
            let mut gg = g.clone();
            gg.hub_prune(&HubPruneConfig::default(), 32, &oracle);
            gg.edge_count()
        })
    });
    g.hub_prune(&HubPruneConfig::default(), 32, &oracle);
    group.bench_function("search_layer_ef128", |b| {
        b.iter(|| g.search_layer(42, &[g.entry], 0, 128, &oracle))
    });
    group.finish();
}

fn bench_two_level(c: &mut Criterion) {
    for n in [2000usize, 10000] {
        let dim = 256;
        let data = corpus(n, dim);
        let texts: Vec<String> = (0..n).map(|i| format!("doc{i} payload text {i}")).collect();
        let source = EmbedSource::new(bench_provider(dim), Metric::Cosine, n);
        let oracle = SliceOracle {
            vecs: &data,
            metric: Metric::Cosine,
        };
        let cfg = GraphBuildConfig {
            m: 16,
            m0: 32,
            ef_construction: 100,
            seed: 1,
        };
        let mut graph = leann_core::graph::HnswGraph::build_local(n, &oracle, &cfg);
        graph.hub_prune(&HubPruneConfig::default(), 16, &oracle);
        let codec = PqCodec::train(
            &data,
            &PqConfig {
                m: 16,
                train_iters: 6,
                ..Default::default()
            },
            Metric::Cosine,
        )
        .unwrap();
        let codes: Vec<u8> = codec.encode_batch(&data).into_iter().flatten().collect();

        let mut group = c.benchmark_group("two_level_search");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n", n), &n, |b, _| {
            let params = SearchParams {
                k: 10,
                ef: 96,
                alpha: 0.08,
                ..Default::default()
            };
            b.iter(|| {
                two_level_search(
                    &graph,
                    &codec,
                    &codes,
                    &source,
                    &texts,
                    &data[7],
                    &params,
                )
                .unwrap()
            })
        });
        group.finish();
    }
}

fn bench_index_e2e(c: &mut Criterion) {
    let n = 5000;
    let dim = 256;
    let texts: Vec<String> = (0..n)
        .map(|i| format!("doc{i} about {}", ["vectors", "rust", "coffee"][i % 3]))
        .collect();
    let mut group = c.benchmark_group("index");
    group.sample_size(10);

    let mk_cfg = || IndexConfig {
        metric: Metric::Cosine,
        pq: PqConfig {
            m: 16,
            train_iters: 4,
            ..Default::default()
        },
        graph: GraphBuildConfig {
            m: 12,
            m0: 24,
            ef_construction: 64,
            seed: 1,
        },
        hub: HubPruneConfig::default(),
        search: SearchDefaults::default(),
        cache_capacity: 8192,
    };
    group.bench_function("build_5k", |b| {
        b.iter(|| LeannIndex::build(&texts, None, bench_provider(dim), mk_cfg()).unwrap())
    });
    let idx = LeannIndex::build(&texts, None, bench_provider(dim), mk_cfg()).unwrap();
    group.bench_function("hybrid_search_5k", |b| {
        b.iter(|| {
            idx.hybrid_search("rust vectors", &leann_core::index::HybridOptions::default())
                .unwrap()
        })
    });
    group.bench_function("vector_search_5k", |b| {
        b.iter(|| idx.search("rust vectors", &SearchOptions::default()).unwrap())
    });
    group.finish();
}

fn bench_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_kernels");
    for dim in [384usize, 768, 1536] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.91).cos()).collect();
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bb, _| {
            bb.iter(|| dot(&a, &b))
        });
        group.bench_with_input(BenchmarkId::new("dot_portable", dim), &dim, |bb, _| {
            bb.iter(|| dot_portable(&a, &b))
        });
        group.bench_with_input(BenchmarkId::new("l2_sq", dim), &dim, |bb, _| {
            bb.iter(|| l2_sq(&a, &b))
        });
        group.bench_with_input(BenchmarkId::new("l2_sq_portable", dim), &dim, |bb, _| {
            bb.iter(|| l2_sq_portable(&a, &b))
        });
    }
    group.finish();
}

fn bench_scratch_and_batch(c: &mut Criterion) {
    use leann_core::search::{two_level_search_tls, two_level_search_with_scratch, SearchScratch};
    let n = 10000;
    let dim = 256;
    let data = corpus(n, dim);
    let texts: Vec<String> = (0..n).map(|i| format!("doc{i} payload {i} about vectors")).collect();
    let source = EmbedSource::new(bench_provider(dim), Metric::Cosine, n);
    let oracle = SliceOracle { vecs: &data, metric: Metric::Cosine };
    let cfg = GraphBuildConfig { m: 16, m0: 32, ef_construction: 100, seed: 1 };
    let mut graph = leann_core::graph::HnswGraph::build_local(n, &oracle, &cfg);
    graph.hub_prune(&HubPruneConfig::default(), 16, &oracle);
    let codec = PqCodec::train(&data, &PqConfig { m: 16, train_iters: 6, ..Default::default() }, Metric::Cosine).unwrap();
    let codes = codec.encode_batch_flat(&data);

    // warm the cache so the benchmark measures the graph+PQ hot path
    let all: Vec<u32> = (0..n as u32).collect();
    let _ = source.batch_by_id(&all, &texts);

    let params = SearchParams { k: 10, ef: 96, alpha: 0.08, ..Default::default() };

    let mut group = c.benchmark_group("search_hot_path");
    group.sample_size(30);
    group.bench_function("fresh_scratch_10k_warm", |b| {
        b.iter(|| {
            two_level_search(&graph, &codec, &codes, &source, &texts, &data[7], &params).unwrap()
        })
    });
    group.bench_function("reused_scratch_10k_warm", |b| {
        let mut s = SearchScratch::new();
        b.iter(|| {
            two_level_search_with_scratch(&graph, &codec, &codes, &source, &texts, &data[7], &params, &mut s).unwrap()
        })
    });
    group.bench_function("tls_scratch_10k_warm", |b| {
        b.iter(|| {
            two_level_search_tls(&graph, &codec, &codes, &source, &texts, &data[7], &params).unwrap()
        })
    });
    group.finish();

    // parallel batch speedup
    let mut group = c.benchmark_group("batch_parallel");
    group.sample_size(20);
    let queries: Vec<Vec<f32>> = (0..64).map(|i| data[(i * 97 + 11) % n].clone()).collect();
    group.bench_function("batch_64_sequential", |b| {
        b.iter(|| {
            queries
                .iter()
                .map(|q| two_level_search_tls(&graph, &codec, &codes, &source, &texts, q, &params).unwrap())
                .count()
        })
    });
    group.bench_function("batch_64_parallel", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|q| two_level_search_tls(&graph, &codec, &codes, &source, &texts, q, &params).unwrap())
                .count()
        })
    });
    group.finish();
}

fn bench_dedup_crypto(c: &mut Criterion) {
    let mut group = c.benchmark_group("dedup_crypto");
    group.sample_size(20);
    let text = "The quarterly report shows revenue growth driven by enterprise sales in the EMEA region with strong retention";
    group.bench_function("simhash64_80_tokens", |b| {
        b.iter(|| leann_core::dedup::simhash64(black_box(text)))
    });
    group.bench_function("deduper_register_1k", |b| {
        b.iter(|| {
            let mut d = leann_core::Deduper::new(leann_core::DedupMode::Near);
            for i in 0..1000 {
                d.register(black_box(&format!("{text} variant {i}")));
            }
            d.len()
        })
    });
    let pt = vec![42u8; 64 * 1024];
    group.bench_function("argon2_derive", |b| {
        b.iter(|| leann_core::crypto::derive_key(black_box("passphrase"), black_box(&[7u8; 16])).unwrap())
    });
    let key = leann_core::crypto::derive_key("passphrase", &[7u8; 16]).unwrap();
    group.bench_function("seal_64kib", |b| {
        b.iter(|| leann_core::crypto::seal(black_box(&key), black_box(&pt)).unwrap())
    });
    let blob = leann_core::crypto::seal(&key, &pt).unwrap();
    group.bench_function("unseal_64kib", |b| {
        b.iter(|| leann_core::crypto::unseal(black_box(&key), black_box(&blob)).unwrap())
    });
    group.finish();
}

// black_box: criterion re-exports it
use criterion::black_box;

criterion_group!(
    benches,
    bench_simd,
    bench_pq,
    bench_graph,
    bench_two_level,
    bench_index_e2e,
    bench_scratch_and_batch,
    bench_dedup_crypto
);
criterion_main!(benches);
