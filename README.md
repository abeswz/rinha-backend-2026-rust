# Fraud Detection API

API de detecção de fraude em tempo real em Rust. Recebe uma transação, compara contra 3 milhões de vetores de referência e retorna uma pontuação de risco com decisão de aprovação.

---

## Arquitetura

```
Client (HTTP :9999)
    └── Rust TCP LB  (bin/lb.rs, 0.05 vCPU)
            ├── api1 (Unix socket /run/sock/api1.sock)
            └── api2 (Unix socket /run/sock/api2.sock)

Por request (cada instância):
    JSON → vetorize → model::predict(q)
                           │
                   p ≤ 0.20│ Legit      → approved:true  (~0.7µs, inline)
                           │ else       → spawn_blocking → IVF KNN → score → response
```

**Sem Axum. Sem nginx. Sem framework HTTP.** Parser HTTP/1.1 custom com keep-alive. Respostas são `&[u8]` estáticos pré-baked (6 níveis).

---

## Pipeline de fraude

### 1. Vetorização

A transação JSON é convertida num vetor de 14 dimensões:

| Dim | Descrição | Fórmula |
|-----|-----------|---------|
| 0 | Valor | `amount / 10000` |
| 1 | Parcelas | `installments / 12` |
| 2 | Razão valor/média cliente | `(amount / avg_amount) / 10` |
| 3 | Hora do dia | `hour / 23` |
| 4 | Dia da semana | `weekday / 6` |
| 5 | Minutos desde última tx | `minutes / 1440` ou `-1.0` se ausente |
| 6 | Distância da última tx (km) | `km / 1000` ou `-1.0` se ausente |
| 7 | Distância de casa (km) | `km_from_home / 1000` |
| 8 | Transações nas últimas 24h | `tx_count_24h / 20` |
| 9 | Terminal online | `1.0` / `0.0` |
| 10 | Cartão presente | `1.0` / `0.0` |
| 11 | Comerciante desconhecido | `1.0` / `0.0` |
| 12 | Risco do MCC | mapa fixo por categoria |
| 13 | Ticket médio do comerciante | `merchant_avg_amount / 10000` |

### 2. Modelo LightGBM (fast-path Legit)

`src/fraud/model_gen.rs` — código Rust gerado por m2cgen a partir de LightGBM treinado nos 3M vetores de referência. 1.65MB, 19.581 branches, ~0.7µs por chamada.

- `p ≤ 0.20` → Legit → resposta imediata (sem IVF)
- `p > 0.20` → IVF

### 3. IVF KNN (busca adaptativa)

`src/fraud/knn.rs` — índice IVF2 i16 com K=4096 clusters. Busca em dois estágios:

| Estágio | nprobe | Quando retorna |
|---------|--------|----------------|
| Rápido  | 5      | votos ≤ 1 ou ≥ 4 |
| Lento   | 24     | ambíguo (2–3 votos) |

Top-5 vizinhos → conta labels fraud → score 0–5 → resposta estática.

| Score | approved | fraud_score |
|-------|----------|-------------|
| 0–2   | true     | 0.0–0.4     |
| 3–5   | false    | 0.6–1.0     |

Scan de centroides com AVX2+FMA. POPCNT para contagem de labels.

---

## Estrutura

```
src/
  main.rs           # Runtime Tokio (2 worker + 2 blocking threads), Unix socket
  env.rs            # SOCK env var
  fraud/
    mod.rs
    data.rs         # Carrega referencias.json.gz + ivf_index.bin no startup
    json.rs         # Parser JSON posicional zero-alloc
    vector.rs       # Transaction → [f32; 14]
    model.rs        # predict() — thresholds LOW/HIGH
    model_gen.rs    # Gerado: m2cgen LightGBM if-else chains
    knn.rs          # IVF KNN adaptativo AVX2
  net/
    mod.rs
    http.rs         # Parser HTTP/1.1 + roteador + handler
    response.rs     # Respostas estáticas pré-baked

bin/
  lb.rs             # Load balancer TCP round-robin
  build_index.rs    # Constrói ivf_index.bin

tools/
  train_model.py    # Treina LightGBM + exporta model_gen.rs via m2cgen
  build_ivf.py      # Constrói ivf_index.bin (MiniBatchKMeans K=4096)

resources/
  references.json.gz   # 3M vetores de referência com labels
  ivf_index.bin        # Índice IVF2 i16 K=4096 (~84MB)
  model.onnx           # Modelo ONNX (não usado em runtime — referência)
  normalization.json   # Constantes de normalização
  mcc_risk.json        # Risco por MCC
```

---

## Restrições de recursos

| Serviço | CPU     | RAM   |
|---------|---------|-------|
| api1    | 0.475   | 172MB |
| api2    | 0.475   | 172MB |
| lb      | 0.05    | 6MB   |
| **Total** | **1.0** | **350MB** |

---

## Comandos

```bash
make bench      # docker up + k6 load test + score
make score      # exibe score do último results.json
make smoke      # smoke test (5 requests)
make publish    # build + push imagem para GHCR
make submission # build + force-push branch submission

cargo test      # testes unitários (34)
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release

# Re-treinar modelo (Python, requer uv)
uv run tools/train_model.py
```

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SOCK`   | `/tmp/fraud-api.sock` | Caminho do Unix socket |

---

## API

### `GET /ready`

Health check. Retorna `200 OK` com body `OK`.

### `POST /fraud-score`

**Request:**
```json
{
  "id": "tx-001",
  "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
  "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009"] },
  "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
  "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.71 },
  "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.86 }
}
```

`last_transaction` pode ser `null` — dims 5 e 6 recebem `-1.0` como sentinela.

**Response:**
```json
{ "approved": true, "fraud_score": 0.2 }
```

---

## Performance

Ver `PROGRESS.md` para histórico completo de testes remotos e evolução arquitetural.

Resultados locais (máquina dev, sem limites de container):

```
p99: 0.23ms | final_score: 6000/6000 | FP: 0 | FN: 0
```
