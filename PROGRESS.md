# Progresso do Projeto

> **Instrução:** Atualizar este arquivo após cada ciclo de teste remoto ou mudança arquitetural significativa.  
> Registrar: o que mudou, por quê, e o resultado (score/p99/FP/FN).

---

## Estado Atual

**Branch principal:** `main`  
**Última atualização:** 2026-05-23  
**Arquitetura ativa:** Legit fast-path (m2cgen) + IVF fallback + observabilidade runtime

### Pipeline por request

```
JSON → vetorize → model::predict(q)
                      │
              p ≤ 0.20 │ Legit     → approved:true  (inline, ~0.7µs)
                       │ Fraud/Uncertain → spawn_blocking → IVF → score 0-5 → response
```

### Resultado local mais recente

```
p99: 0.23ms | final_score: 6000/6000 | FP: 0 | FN: 0
```

---

## Histórico de Testes Remotos

| Data       | Commit imagem | Dataset edge_cases | p99      | FP | FN | Score    | Observação |
|------------|---------------|--------------------|----------|----|----|---------:|------------|
| 2026-05-?  | `f2a276a`     | 645                | 127.17ms | 23 | 0  | 3481.56  | Pure IVF + custom Rust LB |
| 2026-05-?  | `bb5b85f`     | 797                | 366.90ms | 0  | 0  | 3435.45  | AVX2 scan, nginx, slow |
| 2026-05-?  | `a586fa8`     | 797                | 84.83ms  | 0  | 0  | **4071.46** | **Melhor anterior** — worker_threads=2 max_blocking=2 |
| 2026-05-?  | `33baf8f`     | 797                | 105.39ms | 0  | 0  | 3977.21  | multi_thread flavor |
| 2026-05-23 | `c839874`     | 645                | 239.97ms | 51 | 0  | 3105.05  | m2cgen Fraud fast-path ativo — FPs por overfitting no dataset local |
| 2026-05-23 | `1afacab`     | 645                | 98.91ms  | 23 | 0  | 3590.69  | Legit fast-path apenas — melhoria vs `f2a276a` no mesmo dataset |

> **Nota sobre datasets:** O servidor de testes usa dois datasets alternados.  
> Dataset 645 (`edge_case_count=645`): IVF produz 23 FPs intrínsecos, score máximo ~3590.  
> Dataset 797 (`edge_case_count=797`): IVF zera FPs, score máximo observado 4071.

---

## Observabilidade Runtime (2026-05-23)

Adicionado módulo `src/metrics.rs` com 7 contadores atômicos (`AtomicU64`, `Relaxed`) e endpoint `GET /metrics`:

| Contador | Descrição |
|----------|-----------|
| `request_total` | Total de POST /fraud-score processados |
| `fast_path_count` | Requests resolvidos pelo fast-path Legit |
| `ivf_count` | Requests despachados para IVF (spawn_blocking) |
| `fast_probe_count` | Chamadas IVF com nprobe=5 |
| `full_probe_count` | Chamadas IVF com nprobe=24 (ambíguo: score 2 ou 3) |
| `fast_probe_total_us` | µs acumulados em fast probe |
| `full_probe_total_us` | µs acumulados em full probe |

`knn5_ivf` instrumentado com `Instant::now()` — sem alocações, sem overhead observável.

**Exp 3 — Safety zone LOW (0.20 → 0.25):** UNSAFE.  
1 vetor fraud com score 0.2386 cai na zona (0.20, 0.25] (em 3M referências).  
**NÃO elevar LOW para 0.25 — introduz 1 FN real.**

**Ferramentas offline adicionadas:**
- `tools/eval_recall.py` — benchmark recall brute-force vs IVF (nprobe=5 e 24)
- `tools/threshold_analysis.py` — conta vetores fraud na safety zone do threshold

---

## Evolução Arquitetural

### Fase 1 — Baseline (nginx + pure IVF)
- nginx como LB, IVF para todas as requisições
- p99 alto (~366ms) por nginx overhead + IVF sync

### Fase 2 — Custom Rust LB + AVX2 IVF
- Substituiu nginx por TCP LB em Rust (`bin/lb.rs`)
- AVX2+FMA para scan de centroides, POPCNT para contagem de labels
- Busca adaptativa dois estágios: NPROBE=5 rápido → NPROBE=24 para ambíguos
- Melhor score: 4071 com `a586fa8`

### Fase 3 — m2cgen Legit fast-path (atual)
- LightGBM treinado em `resources/references.json.gz` (3M vetores, 14 dims)
- Exportado via m2cgen como if-else Rust (`src/fraud/model_gen.rs`, 1.65MB, 19.581 branches)
- Thresholds: LOW=0.20 (Legit fast-path), HIGH=0.95 (só para testes do modelo)
- **Fraud fast-path desativado** — modelo overfit dataset local → 51 FPs remotos
- Apenas Legit usa fast-path; o restante vai para IVF
- ~44.5% dos requests resolvidos inline em ~0.7µs

### Por que o fast-path de Fraud foi removido
Teste remoto com `c839874` (m2cgen Fraud+Legit fast-path) retornou FP=51 no dataset 645.  
O modelo foi treinado nos dados locais (`references.json.gz`) e overfitou: o dataset remoto tem distribuição ligeiramente diferente.  
IVF prova 0 FPs em dataset 797. No dataset 645, 23 FPs são intrínsecos ao IVF neste dataset.

---

## Próximos Passos Possíveis

- [ ] Testar em dataset 797 com código atual → esperado score > 4071
- [ ] Investigar os 23 FPs do IVF no dataset 645 (threshold KNN ou NPROBE)
- [ ] Coletar dados reais via GET /metrics em produção para validar H1/H2/H3
- [ ] Rodar `tools/eval_recall.py` para medir recall IVF vs brute-force nos dados de teste
- [ ] Se `full_probe_count / fast_probe_count > 50%` → Exp 4a (reduzir FULL_NPROBE 24→12)
- [ ] Se wall-time >> CPU-time em full-probe → Exp 4c (max_blocking 2→4)
- [x] Exp 3 — LOW 0.20→0.25: UNSAFE (1 FN) — Exp 4b bloqueado
- [ ] Avaliar reativação do Fraud fast-path com threshold mais conservador (ex: HIGH=0.999)
- [ ] Re-treinar modelo com dados aumentados para reduzir sensibilidade a distribuição

---

## Referências Rápidas

```bash
make bench         # build + docker up + k6 load test + score
make score         # só mostra score do último results.json
make submission    # build image + force-push branch submission
cargo test         # 38 testes unitários
uv run tools/train_model.py         # re-treina LightGBM + exporta model_gen.rs
uv run tools/eval_recall.py         # benchmark recall IVF vs brute-force
uv run tools/threshold_analysis.py  # safety zone análise do threshold LOW
curl --unix-socket $SOCK http://localhost/metrics  # métricas runtime
```
