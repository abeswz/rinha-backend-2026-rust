# Remote Test Analysis + Optimization Roadmap

**Updated**: 2026-05-30  
**Baseline remote** (clean redesign, commit `c8141fa`): score **3588** / p99 **99.44ms** / FP **23** / FN **0**  
**Baseline local**: score 6000 / p99 0.24ms / FP 0 / FN 0

---

## 1. Decomposing the score gap

```
Current:  p99_score=1002  +  detection_score=2586  =  3588
Max:      p99_score=3000  +  detection_score=3000  =  6000
Gap:                -1998                    -414      -2412
```

Two independent problems. Fix independently.

### 1.1 p99 = 99ms → p99_score = 1002

Formula: `K × log10(T_max / max(p99, p99_MIN))` where K=1000, T_max=1000ms, p99_MIN=1ms.

```
log10(1000 / 99.44) = log10(10.056) = 1.002  →  p99_score = 1002  ✓
```

Score potential at each target:

| p99 target | p99_score | gain vs current |
|---|---|---|
| 99ms (current) | 1002 | — |
| 10ms | 2000 | +998 |
| 1ms | 3000 | +1998 |

Every 10× speedup = +1000 points. The log scale means the first 10× (99→10ms) is worth as much as the next 10× (10→1ms).

**Root cause**: Mac Mini 2014 — dual-core i5 2.6GHz, 4 logical threads. Each instance gets 0.45 CPU = ~45% of one logical core. Under 250 VU burst, throughput ceiling is hit and requests queue. Local machine is much faster, so queueing never appears locally.

### 1.2 FP = 23 → detection_score = 2586

```
E = 1×FP + 3×FN + 5×Err = 1×23 = 23
ε = 23/54100 = 4.26e-4  (below ε_MIN=0.001, so rate_component saturates at 3000)
absolute_penalty = -300 × log10(1+23) = -300 × 1.3802 = -414.06
detection_score  = 3000 - 414 = 2586  ✓
```

Rate is already perfect — ε < ε_MIN means FP rate is so low it doesn't affect the rate term. The entire -414 loss comes from the absolute penalty on E=23.

Fix FP=0 → detection_score=3000 → **+414 points guaranteed**.

**Root cause**: Remote test dataset differs from local (`edge_case_count=645` remote vs `797` local). For 23 cases, brute-force exact KNN gives fraud_count≤1 (approved), but our IVF search returns wrong neighbors → fraud_count≥3 (denied = FP). This is IVF recall failure: suboptimal centroids cause the true nearest neighbors to fall outside the probed clusters.

---

## 2. Optimization roadmap

Prioritized by impact × implementation cost.

---

### 2.1 Fix FP via better centroids [HIGH IMPACT, MEDIUM COST]

**Impact**: Guaranteed +414 detection score → ~4000 total (holding p99 fixed).

**Mechanism**: `bin/build_index.rs` uses KMeans++ initialized from a 50k sample of the 3M dataset. With K=4096, 50k init vectors = ~12 candidates per cluster on average — poor coverage of cluster geometry. Centroids land suboptimally. Clusters in low-density regions (edge cases, unusual feature combinations) are impure, causing IVF to probe the wrong clusters for those 23 test points.

**Solution**: Rebuild index with `tools/build_ivf.py` (already written, produces IVF2 format):
- MiniBatchKMeans from sklearn (full 3M dataset input, batch size 10k)
- N_INIT=3 (multiple restarts, picks best inertia)
- Output: `resources/ivf_index.bin` (IVF2 format, uncompressed)

Then adapt runtime to read IVF2:
- `data.rs`: change magic check `IVF1` → `IVF2`, adjust block padding format (`i16::MAX` instead of `0`)
- IVF2 is uncompressed, so remove flate2 decompression step (simplifies loading)
- `include_bytes!` path: point to `resources/ivf_index.bin` instead of gzip file

**Validation**: run `tools/eval_recall.py` before/after. Must show recall improvement at NPROBE=5 and NPROBE=24.

---

### 2.2 Reduce p99 via i8 block scan [HIGH IMPACT, MEDIUM COST]

**Impact**: 2× throughput in block scan → more requests processed per second → less queueing under 250 VU → lower p99. Estimated 30–50% p99 reduction (from 99ms toward 50ms). Combined with 2.1, possibly toward 30ms.

**Mechanism**: Current block scan: i16 vectors (scale=10000), AVX2 loads 16 i16 per register = 8 effective f32-equivalent lanes per iter. All 14 features fit in i8 with scale=100 (range [-1.0,1.0] → [-100,100]). dim13 (merchant_avg_amount/10000) has std=0.015 → only ~5 distinct i8 values, but it's the lowest-variance feature and has minimal discriminative power.

i8 AVX2: `_mm256_cvtepi8_epi16` (extends 16×i8 → 16×i16) + `_mm256_madd_epi16` (multiply-accumulate 16 pairs) = 32 i8 per register → **2× throughput**.

**Changes**:
- `tools/build_ivf.py`: change quantize scale 10000→100, dtype int16→int8
- `data.rs`: change `AVec<i16>` to `AVec<i8>` for block storage
- `knn.rs`: rewrite `scan_blocks_avx2`:
  ```rust
  // Load 32 i8 (2 full vectors of dim=14 with 2 padding bytes)
  let v = _mm_loadu_si64(block_ptr);          // load 8 i8
  let v16 = _mm256_cvtepi8_epi16(v128);       // extend to 16 i16
  // accumulate with _mm256_madd_epi16
  ```
- Rebuild index with i8 quantization (can combine with 2.1)

**Risk**: Overflow in squared distance. With i8 values up to ±100, squared diff up to 40000 per dim, 14 dims → max sum 560000. Fits in i32 (max ~2B). Accumulate in i32 via `_mm256_add_epi32` after `_mm256_madd_epi16`.

**Validation**: `eval_recall.py` must show identical or better FP/FN before shipping.

---

### 2.3 Vectorize centroid selection [MEDIUM IMPACT, LOW COST]

**Impact**: ~20–25% reduction in fast path (NPROBE=5 cases). Estimated ~5–10ms p99 improvement on top of 2.2.

**Mechanism**: `top_n_centroids_fast` uses insertion sort over K=4096 entries to find top-NPROBE nearest centroids:

```rust
let pos = top[..nprobe].partition_point(|&(b, _)| b <= bits);
top[pos..nprobe].rotate_right(1);  // O(K × NPROBE) ≈ 98k ops
```

Replace with partial sort via `select_nth_unstable` (PDQ = O(K) expected ≈ 4096 ops):

```rust
let mut indexed: [(u32, u16); 4096] = ...;  // stack array, no alloc
// fill with (dist_bits, centroid_idx)
indexed.select_nth_unstable_by_key(nprobe - 1, |&(b, _)| b);
indexed[..nprobe].sort_unstable_by_key(|&(b, _)| b);
```

No heap allocation. Benchmark before committing — PDQ is O(N) expected but has higher constant than insertion sort for very small N. At K=4096 and NPROBE=5..24, PDQ wins.

---

### 2.4 Worker thread tuning [MEDIUM IMPACT, NO COST]

**Context**: 0.45 CPU on dual-core 2.6GHz = ~45% of one logical core. Each tokio worker thread competes for this slice. With 4 workers, context switching overhead adds up — OS scheduler gives each worker ~0.45/4 = 11% of one core.

**Hypothesis**: 2 workers may outperform 4 at 0.45 CPU. Less context switch, same max throughput (throughput is CPU-bound, not parallelism-bound at this scale).

**Test**: Two builds — `worker_threads(2)` vs `worker_threads(4)` — same image, benchmark remotely. No code change beyond runtime config.

---

## 3. What NOT to do

| Idea | Why not |
|---|---|
| Raise fraud threshold (fraud_count ≥ 4) | Converts FP→FN at 3× penalty weight. E goes up, not down |
| HNSW | N=3M, 32-NN graph ≈ 1.2GB RAM. Blows 168MB limit |
| PQ (Product Quantization) | D=14 doesn't split into meaningful sub-vectors. Marginal gain, precision loss |
| More API instances | Budget fixed at 350MB / 0.95 CPU. Can't add meaningful third instance |
| Blocking on FP during NPROBE=24 | Already doing this. The 23 FP are from wrong neighbor search, not threshold |

---

## 4. Priority order

```
1. Rebuild index with sklearn (2.1)     → +414 detection score (guaranteed)
2. Validate recall with eval_recall.py  → safety gate before shipping
3. i8 block scan (2.2)                  → ~2x block scan throughput
4. select_nth_unstable (2.3)            → ~20% fast path speedup
5. Worker thread A/B test (2.4)         → test 2 vs 4 workers remotely
```

Steps 1+2 are prerequisites for 3 (index rebuild needed for both). Steps 3+4 are independent of each other and can be done in parallel.

---

## 5. Score history

| Commit | Architecture | p99 remote | FP | FN | Score |
|---|---|---|---|---|---|
| `87ae940` | Rust LB + spawn_blocking(2) + LightGBM model | 210ms | 23 | 0 | 3263 |
| `a586fa8` | nginx + IVF pure | 84ms | — | — | 4071 |
| `c8141fa` | HAProxy TCP + inline IVF + 4 workers | **99ms** | **23** | **0** | **3588** |

**p99 improved** from 210ms to 99ms (+1 point in log scale) — spawn_blocking bottleneck eliminated. LightGBM model removed (was causing I-cache pollution). The FP count (23) is identical to 87ae940, suggesting FP comes from dataset characteristics, not model interference — it's an IVF recall problem present in both versions.

**Next target**: Fix IVF recall (path to FP=0) + reduce p99 to 10ms → projected score ~5000.

## Para guia
Conversa inicial foi esta e resposta:
para melhorar ainda mais isso, aqui está o conversamos, sua mensagem:           
  Estado atual — números concretos                                                
                                                                                  
    N = 3,000,000 vetores  D = 14  K = 4096 centroids                             
    Avg cluster size: 3M / 4096 ≈ 732 vetores = 92 blocos (8-vec/bloco)           
                                                                                  
    Custo por request (estimativa de SIMD ops):                                   
                                                                                  
    ┌───────────────────────────┬─────────────────┬──────────────────┐            
    │           Fase            │ Fast (NPROBE=5) │ Full (NPROBE=24) │            
    ├───────────────────────────┼─────────────────┼──────────────────┤            
    │ Centroid scan (AVX2)      │ ~41K ops        │ ~41K ops         │            
    ├───────────────────────────┼─────────────────┼──────────────────┤            
    │ Top-N selection (escalar) │ ~20K ops        │ ~20K ops         │            
    ├───────────────────────────┼─────────────────┼──────────────────┤            
    │ Block scan (AVX2)         │ ~18K ops        │ ~88K ops         │            
    ├───────────────────────────┼─────────────────┼──────────────────┤            
    │ Total                     │ ~79K ops        │ ~149K ops        │            
    └───────────────────────────┴─────────────────┴──────────────────┘            
                                                                                  
    A diferença entre os dois caminhos é quase 2x. A pergunta central que não     
    sabemos responder ainda: que % das requests cai no caminho full?              
                                                                                  
    ---                                                                           
    Os 3 caminhos que você identificou — análise                                  
                                                                                  
    Caminho 1: Como buildamos o índice (qualidade dos centroids)                  
                                                                                  
    Problema crítico do builder atual (bin/build_index.rs):                       
                                                                                  
    // Inicializa KMeans++ com sample de apenas 50k de 3M                         
    let mut centroids = kmeans_plus_plus_init(&vecs, K, INIT_SAMPLE);             
    // 25 Lloyd iterations sobre TODOS os 3M                                      
    for iter in 0..LLOYD_ITERS { ... }                                            
                                                                                  
    A inicialização ruim com seed LCG fixo (0xdeadbeef) pode gerar centróides     
    subótimos. Centróides ruins → clusters impuros (mistura fraud/legit) → mais   
    requests caindo em fraud_count==2||3 → mais NPROBE=24.                        
                                                                                  
    O builder Python (build_ivf.py) usa MiniBatchKMeans do sklearn com N_INIT=3 — 
    gera centróides de qualidade superior, mas produz formato IVF2 que o runtime  
  não                                                                             
     lê.                                                                          
                                                                                  
    Oportunidade real: Atualizar o runtime para ler IVF2 (ou adaptar o Python     
    builder para escrever IVF1). Estimar ganho: 10-30% menos requests no full     
  probe                                                                           
    path.                                                                         
                                                                                  
    ---                                                                           
    Caminho 2: Como carregamos em memória                                         
                                                                                  
    Isso já está otimizado ao máximo:                                             
                                                                                  
    - include_bytes! — índice embedded no binário, sem I/O em runtime             
    - AVec<_, ConstAlign<32>> — alinhamento 32-byte para AVX2                     
    - fill_avec / fill_vec — popula direto na memória final, zero cópia           
    intermediária                                                                 
    - GzDecoder na startup apenas — nenhum I/O em hot path                        
                                                                                  
    Não tem o que fazer aqui. Está ideal.                                         
                                                                                  
    ---                                                                           
    Caminho 3: Algoritmo de busca vetorial                                        
                                                                                  
    3a. Top-N centroid selection — gargalo oculto                                 
                                                                                  
    fn top_n_centroids_fast(dists: &[f32; K], nprobe: usize) -> [u16;             
  FULL_NPROBE] {                                                                  
        // insertion sort sobre 4096 elementos para encontrar top-24              
        // O(K * NPROBE) = 4096 * 24 ≈ 98K operações, com rotate_right            
                                                                                  
    Isso é escalar puro. Para o fast path (NPROBE=5) o centroid scan + top-N      
    representa mais de 75% do custo total. Alternativas:                          
                                                                                  
    - PDQ / nth_element: O(K) average, ~4K ops                                    
    - SIMD top-N: comparar 8 distâncias por vez, manter heap de 24 com AVX2       
                                                                                  
    3b. Quantização i8 — viável para este dataset                                 
                                                                                  
    Todas as 14 features têm range [-1.0, 1.0] — todas cabem em i8 com scale=100: 
                                                                                  
    dim 0: [0.001, 1.0]  ✓                                                        
    dim 5: [-1.0, 0.5]   ✓                                                        
    dim13: [0.002, 0.05]  ← problema: só 5 valores distintos com scale=100        
                                                                                  
    dim13 tem std=0.0146 — quase sem variância, baixo poder discriminativo. Com   
  i8                                                                              
    você perde precisão exatamente onde ela menos importa. Gain: AVX2 processa 32 
   i8                                                                             
     vs 16 i16 = 2x throughput no block scan.                                     
                                                                                  
    Risco: impacto em FP/FN precisa ser medido empiricamente antes de usar.       
                                                                                  
    3c. K tuning — sweet spot                                                     
                                                                                  
    K=4096: centroid scan = 512 SIMD iters, avg_cluster = 92 blocos               
    K=8192: centroid scan = 1024 SIMD iters, avg_cluster = 46 blocos              
                                                                                  
    Com K=8192 e NPROBE=10, é possível ter recall equivalente a K=4096 +          
  NPROBE=24                                                                       
    com metade do block scan. Mas centroid scan dobra. Precisa medir recall ×     
  speed                                                                           
    por configuração.                                                             
                                                                                  
    ---                                                                           
    Minha recomendação — ordem de prioridade                                      
                                                                                  
    1. Instrumentar a ambiguity rate (antes de qualquer coisa)                    
    Adicionar um contador atômico que mede % de requests que entram no full       
  probe.                                                                          
    Se for < 5%, o full path não importa. Se for > 30%, é o principal problema.   
                                                                                  
    2. Build melhor com Python/sklearn → IVF2 + adaptar runtime                   
    Maior potencial de ganho sem mudar o algoritmo de busca. O sklearn            
    MiniBatchKMeans com N_INIT=3 produz centróides muito melhores que 25 Lloyd    
    iterations sobre seed ruim.                                                   
                                                                                  
    3. Vectorizar top-N selection                                                 
    Mudar top_n_centroids_fast de insertion sort para partial sort                
    (select_nth_unstable do Rust, que usa PDQ pattern = O(K) expected). Uma linha 
   de                                                                             
     mudança, ganho de ~20% no fast path.                                         
                                                                                  
    4. K=8192 benchmark                                                           
    Testar se K maior com NPROBE menor tem melhor tradeoff. Mudança no builder e  
  no                                                                              
    compile-time const K.                                                         
                                                                                  
    5. i8 quantization (só depois de medir impacto em accuracy)                   
                                                                                  
    ---                                                                           
    O que NÃO vale a pena                                                         
                                                                                  
    - HNSW: N=3M, D=14 — seria 1.2GB de grafo. Inviável com 168MB de RAM.         
    - Melhorar carregamento de memória: já no máximo.                             
    - HTTP/transport: correto, não é o gargalo.                                   
                                                                                  
    O maior leverage está na qualidade dos centróides (build) e na seleção do     
  top-N                                                                           
    (search). Concordo com sua análise dos 3 caminhos, mas o segundo (memória) já 
    está resolvido.                                                               
                                                                                  
  minha resposta:                                                                 
  Concordo com o que explorou, e aqui tenho alguns guias e também o resultado do  
                                                                                  
    teste remoto.                                                                 
                                                                                  
                                                                                  
                                                                                  
    Sobre sua recomendação, eu não faria o Instrumentar a ambiguity rate, acho    
  que                                                                             
    não temos necessidade disso, acho que estamos solidos com o caso.             
                                                                                  
                                                                                  
                                                                                  
    Gosto muito disso que levantou no "Build melhor com Python/sklearn → IVF2 +   
                                                                                  
    adaptar runtime", me parece que esta adaptação pode melhorar e muito como     
                                                                                  
    tratamos o indice e utilizamos o que o python fez de melhor.                  
                                                                                  
                                                                                  
                                                                                  
    Com isso cobrimos melhorias para como buildamos o índice, e como lemos ele,   
                                                                                  
    correto?                                                                      
                                                                                  
                                                                                  
                                                                                  
    Melhoria no algoritmo de busca vetorial me parece necessaria, além disso e    
                                                                                  
    muito importante a quantizacao dos vetores para i8.                           
                                                                                  
                                                                                  
                                                                                  
    Sobre o full patch, seria interassante nos proteger usando o máximo que       
                                                                                  
    podemos, sem impactar performance, para tentar garantir que a maioria dos     
  casos                                                                           
     não caia no full path, assim tentamos ter a maior cobertura possível,        
  chegando                                                                        
     proximo do limite de impacto de performance, para não afetar.
