BINARY   := target/release/fraud-detection
REFS_BIN := resources/refs.bin
PORT     := 9999
READY_URL := http://localhost:$(PORT)/ready

.PHONY: all build preprocess up down dev smoke load test clean help

all: help

help:
	@echo "Targets:"
	@echo "  make up       Build image + start docker compose (nginx:9999 → api1+api2)"
	@echo "  make down     Stop docker compose"
	@echo "  make dev      Run single local instance on port 9999 (no Docker)"
	@echo "  make smoke    Run k6 smoke test (5 requests)"
	@echo "  make load     Run k6 load test (54k transactions, 120s ramp)"
	@echo "  make build    cargo build --release"
	@echo "  make preprocess  Generate resources/refs.bin from references.json.gz"
	@echo "  make clean    Remove build artifacts"

# ── Build ──────────────────────────────────────────────────────────────────

build:
	cargo build --release

$(BINARY): build

preprocess: $(REFS_BIN)

$(REFS_BIN):
	@echo "refs.bin not found — running preprocessor..."
	cargo run --bin preprocess --release

# ── Docker Compose (nginx:9999 → api1:3000 + api2:3000) ───────────────────

up: $(REFS_BIN)
	docker compose up --build -d
	@echo "Waiting for $(READY_URL)..."
	@until curl -sf $(READY_URL) > /dev/null 2>&1; do \
		printf '.'; sleep 1; \
	done
	@echo " ready."

down:
	docker compose down

# ── Local dev (single instance on port 9999, no Docker) ───────────────────

dev: $(BINARY) $(REFS_BIN)
	PORT=$(PORT) \
	REFS_PATH=$(REFS_BIN) \
	MCC_PATH=resources/mcc_risk.json \
	NORM_PATH=resources/normalization.json \
	$(BINARY)

# ── k6 tests (servidor deve estar rodando em localhost:9999) ───────────────

smoke:
	k6 run test/smoke.js

load:
	k6 run test/test.js

# ── Misc ───────────────────────────────────────────────────────────────────

clean:
	cargo clean
