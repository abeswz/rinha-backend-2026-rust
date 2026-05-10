# Fraud Detection API

API de detecção de fraude em tempo real construída em Rust. Recebe uma transação, compara contra 3 milhões de vetores de referência e retorna uma pontuação de risco com decisão de aprovação.

---

## Como funciona

### Visão geral

O sistema usa **KNN brute-force** (K-Nearest Neighbors) para classificar transações. Em vez de um modelo de ML complexo, a abordagem é direta:

1. A transação chega como JSON
2. É convertida num vetor de 14 dimensões
3. As 5 transações mais próximas são buscadas em 3M de referências
4. A proporção de fraudes entre as 5 vizinhas vira o `fraud_score`
5. Se `fraud_score >= 0.6`, a transação é negada

### O vetor de 14 dimensões

Cada transação é normalizada para o intervalo `[0.0, 1.0]` (ou `-1.0` como sentinela quando o dado não existe):

| Dim | Descrição | Fórmula |
|-----|-----------|---------|
| 0 | Valor da transação | `amount / 10000` |
| 1 | Número de parcelas | `installments / 12` |
| 2 | Razão valor vs. média do cliente | `(amount / avg_amount) / 10` |
| 3 | Hora do dia | `hour / 23` |
| 4 | Dia da semana | `weekday / 6` (Seg=0, Dom=6) |
| 5 | Minutos desde última transação | `minutes / 1440` ou `-1.0` se ausente |
| 6 | Distância da última transação (km) | `km / 1000` ou `-1.0` se ausente |
| 7 | Distância de casa (km) | `km_from_home / 1000` |
| 8 | Transações nas últimas 24h | `tx_count_24h / 20` |
| 9 | Terminal online | `1.0` se online, `0.0` se não |
| 10 | Cartão presente | `1.0` se presente, `0.0` se não |
| 11 | Comerciante desconhecido | `1.0` se novo, `0.0` se conhecido |
| 12 | Risco do MCC | Mapa fixo por categoria de comerciante |
| 13 | Ticket médio do comerciante | `merchant_avg_amount / 10000` |

### Risco por MCC (categoria do comerciante)

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
| 5999 | Outros | 0.50 |
| Desconhecido | — | 0.50 |

### Decisão final

```
fraud_score = fraudes_entre_5_vizinhas / 5
approved    = fraud_score < 0.6
```

### Armazenamento dos vetores de referência

Os 3 milhões de vetores são armazenados em `resources/refs.bin` como `f16` (float de 16 bits). Isso reduz o tamanho de ~168MB (f32) para ~83MB, cabendo na restrição de 170MB por instância. O arquivo é carregado inteiro na memória no startup — nenhuma I/O em tempo de requisição.

---

## Estrutura do projeto

```
src/
  lib.rs              # AppState e módulos públicos (usado pelos testes)
  main.rs             # Binário principal (HTTP server)
  config.rs           # Config via variáveis de ambiente
  error.rs            # AppError com IntoResponse
  domain/
    transaction.rs    # Structs de domínio (Transaction, Customer, Merchant...)
    fraud.rs          # FraudVector([f32;14]), FraudDecision
  service/
    vectorizer.rs     # Vetorização: Transaction → FraudVector
  repository/
    reference.rs      # Leitura do refs.bin + KNN brute-force
  usecase/
    score_fraud.rs    # Orquestração: vetoriza → KNN → decisão
  web/
    dto.rs            # DTOs de request/response (serde)
    handlers.rs       # Handlers Axum
    router.rs         # Rotas: GET /ready, POST /fraud-score
bin/
  preprocess.rs       # Converte references.json.gz → refs.bin
resources/
  references.json.gz  # Dataset de referência (3M transações rotuladas)
  refs.bin            # Vetores pré-processados em f16 (~83MB)
  mcc_risk.json       # Mapa de risco por MCC
  normalization.json  # Constantes de normalização
test/
  smoke.js            # Smoke test k6 (1 VU, 5 iterações)
  test.js             # Load test k6 com scoring de detecção
  test-data.json      # 54.100 transações com gabarito (44% fraude)
```

---

## Como usar

### Pré-requisitos

- Rust 1.80+
- Docker + Docker Compose (para rodar via contêiner)
- k6 (para testes de carga)

### Makefile

Todos os fluxos comuns estão cobertos pelo `Makefile`:

| Comando | O que faz |
|---------|-----------|
| `make up` | Build da imagem Docker + `docker compose up -d` + aguarda `/ready` |
| `make down` | Para o docker compose |
| `make dev` | Compila e roda uma instância local direto na porta 9999 (sem Docker) |
| `make smoke` | Executa o smoke test k6 (5 requests, valida JSON/campos) |
| `make load` | Executa o load test k6 completo (54k transações, 120s de rampa) |
| `make build` | `cargo build --release` |
| `make preprocess` | Gera `resources/refs.bin` a partir de `references.json.gz` |
| `make doc` | Abre a documentação Rust gerada pelo `cargo doc` no browser |
| `make clean` | Remove artefatos de build |

#### Fluxo com Docker (stack completa)

```bash
make up      # build + sobe nginx:9999 → api1:3000 + api2:3000
make smoke   # valida que está respondendo
make load    # executa o load test completo
make down    # para tudo
```

#### Fluxo local sem Docker

```bash
make dev     # em um terminal (instância única na porta 9999)
make smoke   # em outro terminal
```

`make up` e `make dev` rodam `make preprocess` automaticamente se `resources/refs.bin` não existir.

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `PORT` | `3000` | Porta HTTP |
| `REFS_PATH` | `resources/refs.bin` | Vetores de referência |
| `MCC_PATH` | `resources/mcc_risk.json` | Mapa de risco MCC |
| `NORM_PATH` | `resources/normalization.json` | Constantes de normalização |

---

## API

### `GET /ready`

Health check.

**Resposta:** `200 OK` com body `ok`

---

### `POST /fraud-score`

Avalia uma transação.

**Request:**
```json
{
  "id": "tx-001",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-009", "MERC-001"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.71
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.86
  }
}
```

`last_transaction` pode ser `null` quando não há histórico — as dimensões 5 e 6 do vetor recebem `-1.0` como sentinela.

**Resposta:**
```json
{
  "approved": true,
  "fraud_score": 0.2
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `approved` | bool | `true` se `fraud_score < 0.6` |
| `fraud_score` | float | 0.0 a 1.0 (proporção de vizinhos fraudulentos) |

**Erros:**
- `422 Unprocessable Entity` — campo obrigatório ausente ou timestamp inválido

---

## Testes

### Testes unitários e de integração (Rust)

```bash
# Todos os testes
cargo test

# Apenas testes de integração
cargo test --test integration

# Apenas testes de regressão
cargo test --test regression
```

Os testes de integração e regressão carregam o `refs.bin` real (3M vetores). O primeiro run demora ~4s para carregar; os testes em si são rápidos.

### Smoke test (k6)

Verifica se o servidor está respondendo corretamente com 5 iterações sequenciais.

```bash
# Com o servidor rodando na porta 9999
k6 run test/smoke.js
```

Valida: status 200, body JSON válido, campos `approved` (boolean) e `fraud_score` (number) presentes.

### Load test com scoring (k6)

Teste de carga completo com 54.100 transações reais e gabarito de detecção. Rampa de 0 a 900 req/s em 120 segundos.

```bash
# Com docker compose rodando
k6 run test/test.js
```

Gera `test/results.json` com o breakdown de detecção e a pontuação final.

#### Como o scoring funciona

O teste classifica cada resposta em quatro categorias:

| Categoria | Sigla | Descrição |
|-----------|-------|-----------|
| True Positive | TP | Fraude corretamente bloqueada |
| True Negative | TN | Legítima corretamente aprovada |
| False Positive | FP | Legítima incorretamente bloqueada |
| False Negative | FN | Fraude incorretamente aprovada |

**Pesos dos erros:**
- FP = 1 ponto de penalidade
- FN = 3 pontos (fraude não detectada é pior)
- Erro HTTP = 5 pontos

**Fórmula de pontuação:**

```
E             = (FP × 1) + (FN × 3) + (erros_http × 5)
epsilon       = E / N
score_p99     = 1000 × log10(1000 / p99_ms)
score_detecção = 1000 × log10(1 / epsilon) - 300 × log10(1 + E)
score_final   = score_p99 + score_detecção
```

Cortes automáticos aplicados:
- `p99 > 2000ms` → `score_p99 = -3000`
- `taxa de falhas > 15%` → `score_detecção = -3000`

Dataset de teste: **54.100 transações** (44% fraude, 56% legítima, 1.5% casos-limite).

---

## Restrições de recursos (por instância)

| Recurso | Limite |
|---------|--------|
| CPU | 0.475 vCPU |
| RAM | 170 MB |
| nginx (proxy) | 0.05 vCPU / 10 MB |
| **Total** | **1 CPU / 350 MB** |

O `refs.bin` ocupa ~83MB em RAM. O restante (~87MB) cobre o binário, stack Tokio, buffers de conexão e overhead do OS.
