# Fraud Detection API

API de detecção de fraude em tempo real construída em Rust. Recebe uma transação, compara contra 3 milhões de vetores de referência e retorna uma pontuação de risco com decisão de aprovação.

---

## Arquitetura

```
                  ┌─────────────────────────────────────────────┐
                  │              Docker Compose                  │
                  │                                             │
  Client ─────►  │  nginx:9999 ──least_conn──► api1:3000       │
                  │               │             api2:3000       │
                  │               └─────────────────┘           │
                  │                      │                       │
                  │              spawn_blocking                  │
                  │                      │                       │
                  │              ┌───────▼──────────┐            │
                  │              │  knn_adaptive    │            │
                  │              │  Stage 1 (fast)  │            │
                  │              │  nprobe = 5      │            │
                  │              └───────┬──────────┘            │
                  │                      │                       │
                  │              ambiguous vote?                  │
                  │               /          \                   │
                  │             no            yes                │
                  │              │             │                  │
                  │           return    ┌──────▼──────────┐      │
                  │                    │  Stage 2 (slow)  │      │
                  │                    │  nprobe = 24     │      │
                  │                    └──────┬───────────┘      │
                  │                           │                   │
                  │                        return                 │
                  └─────────────────────────────────────────────┘
```

---

## Como funciona

### Visão geral

O sistema usa **IVF KNN** (Inverted File Index + K-Nearest Neighbors) para classificar transações:

1. A transação chega como JSON
2. É convertida num vetor de 14 dimensões
3. As 5 transações mais próximas são buscadas via busca adaptativa em dois estágios
4. A proporção de fraudes entre as 5 vizinhas vira o `fraud_score`
5. Se `fraud_score >= 0.6`, a transação é negada

### Busca adaptativa (dois estágios)

Em vez de sempre sondar `N` clusters, a busca usa dois estágios:

| Estágio | nprobe | Quando retorna |
|---------|--------|----------------|
| Rápido  | 5      | votos ≤ 1 (claramente legítimo) ou votos ≥ 4 (claramente fraude) |
| Lento   | 24     | casos ambíguos (2–3 votos de fraude em 5) |

A maioria das requisições retorna no estágio rápido (nprobe=5 ≈ 0,3% dos clusters). Apenas os casos ambíguos sobem para o estágio lento (nprobe=24 ≈ 1,4% dos clusters), melhorando acurácia sem penalizar latência no caso médio.

### O vetor de 14 dimensões

| Dim | Descrição | Fórmula |
|-----|-----------|---------|
| 0 | Valor da transação | `amount / 10000` |
| 1 | Número de parcelas | `installments / 12` |
| 2 | Razão valor vs. média do cliente | `(amount / avg_amount) / 10` |
| 3 | Hora do dia | `hour / 23` |
| 4 | Dia da semana | `weekday / 6` |
| 5 | Minutos desde última transação | `minutes / 1440` ou `-1.0` se ausente |
| 6 | Distância da última transação (km) | `km / 1000` ou `-1.0` se ausente |
| 7 | Distância de casa (km) | `km_from_home / 1000` |
| 8 | Transações nas últimas 24h | `tx_count_24h / 20` |
| 9 | Terminal online | `1.0` se online, `0.0` se não |
| 10 | Cartão presente | `1.0` se presente, `0.0` se não |
| 11 | Comerciante desconhecido | `1.0` se novo, `0.0` se conhecido |
| 12 | Risco do MCC | Mapa fixo por categoria |
| 13 | Ticket médio do comerciante | `merchant_avg_amount / 10000` |

### Risco por MCC

| MCC | Categoria | Risco |
|-----|-----------|-------|
| 5411 | Supermercados | 0.15 |
| 5812 | Restaurantes | 0.30 |
| 5912 | Farmácias | 0.20 |
| 5944 | Joalherias | 0.45 |
| 7801 | Loteria / apostas | 0.80 |
| 7802 | Corridas de cavalo | 0.75 |
| 7995 | Cassinos / jogos | 0.85 |
| 4511 | Companhias aéreas | 0.35 |
| 5311 | Lojas de departamento | 0.25 |
| 5999 / Outros | — | 0.50 |

### Índice IVF

Os 3 milhões de vetores são agrupados em **K=1732 clusters** via MiniBatchKMeans. Armazenado em `resources/ivf_index.bin` como `f16` (~90MB em RAM). O índice é carregado inteiro no startup — nenhuma I/O por requisição.

A busca usa SIMD (AVX2 + F16C) para conversão f16→f32 e cálculo de distância, e `select_nth_unstable` (O(K) vs O(K log K)) para selecionar os clusters mais próximos.

### Runtime

- **2 worker threads** Tokio: accept/parse/serialize
- **8 blocking threads**: buscas IVF paralelas via `spawn_blocking`
- **mimalloc**: alocador global de alta performance
- **500 queries de warmup** no startup para primar caches de CPU

---

## Estrutura do projeto

```
src/
  lib.rs              # AppState, módulos públicos
  main.rs             # Binário (runtime Tokio + mimalloc)
  config.rs           # Config via variáveis de ambiente
  domain/
    transaction.rs    # Transaction, Customer, Merchant...
    fraud.rs          # FraudDecision
  service/
    vectorizer.rs     # Transaction → [f32; 14]
  repository/
    ivf.rs            # IvfIndex: knn() + knn_adaptive()
    reference.rs      # ReferenceRepository
  usecase/
    score_fraud.rs    # Vetoriza → knn_adaptive → decisão
  web/
    dto.rs            # DTOs request/response
    handlers.rs       # Axum handlers (spawn_blocking)
    router.rs         # GET /ready, POST /fraud-score
tools/
  build_ivf.py        # Gera ivf_index.bin (MiniBatchKMeans K=1732)
resources/
  ivf_index.bin       # Índice IVF f16 (~90MB) — gerado, não versionado
  mcc_risk.json       # Risco por MCC
  normalization.json  # Constantes de normalização
```

---

## Como usar

### Makefile

| Comando | O que faz |
|---------|-----------|
| `make up` | Build da imagem + `docker compose up -d` + aguarda `/ready` |
| `make down` | Para o docker compose |
| `make dev` | Roda instância local na porta 9999 (sem Docker) |
| `make smoke` | Smoke test k6 (5 requests) |
| `make load` | Load test k6 (54k transações, 120s) |
| `make publish` | Build + push da imagem para GHCR |
| `make submission` | Cria branch `submission` com 3 arquivos, força push |

### Fluxo Docker

```bash
make up      # build + nginx:9999 → api1+api2
make smoke   # valida resposta
make load    # load test completo
make down
```

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `PORT` | `3000` | Porta HTTP |
| `IVF_PATH` | `resources/ivf_index.bin` | Índice IVF |
| `IVF_NPROBE` | `24` | nprobe do estágio lento (deve ser ≥ 5) |
| `MCC_PATH` | `resources/mcc_risk.json` | Mapa de risco MCC |
| `NORM_PATH` | `resources/normalization.json` | Constantes de normalização |

---

## API

### `GET /ready`

Health check. Retorna `200 OK` com body `ok`.

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

**Response:**
```json
{ "approved": true, "fraud_score": 0.2 }
```

`last_transaction` pode ser `null` — dimensões 5 e 6 recebem `-1.0` como sentinela.

---

## Testes

```bash
cargo test                    # todos os testes
cargo test --test integration # integração (requer ivf_index.bin)
cargo test --test regression  # regressão (requer ivf_index.bin)
```

---

## Restrições de recursos (por instância)

| Recurso | Limite |
|---------|--------|
| CPU | 0.475 vCPU |
| RAM | 170 MB |
| nginx | 0.05 vCPU / 10 MB |
| **Total** | **1 CPU / 350 MB** |
