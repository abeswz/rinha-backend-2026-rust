BINARY         := target/release/fraud-detection
PROFILING_BIN  := target/profiling/fraud-detection
IVF_BIN        := resources/ivf_index.bin
PORT           := 9999
READY_URL      := http://localhost:$(PORT)/ready

.PHONY: all build ivf up down dev smoke load test clean doc help profile

all: help

help:
	@echo "Targets:"
	@echo "  make up    Build image + start docker compose (nginx:9999 → api1+api2)"
	@echo "  make down  Stop docker compose"
	@echo "  make dev   Run single local instance on port 9999 (no Docker)"
	@echo "  make ivf   Generate resources/ivf_index.bin via Python (3-8 min)"
	@echo "  make smoke Run k6 smoke test (5 requests)"
	@echo "  make load  Run k6 load test (54k transactions, 120s ramp)"
	@echo "  make build   cargo build --release"
	@echo "  make profile Build profiling binary + run under samply (open Firefox Profiler)"
	@echo "  make doc     Open rustdoc in browser (cargo doc --open)"
	@echo "  make clean   Remove build artifacts"

# ── Build ──────────────────────────────────────────────────────────────────

build:
	cargo build --release

profile-build:
	cargo build --profile profiling

$(PROFILING_BIN): profile-build

$(BINARY): build

ivf: $(IVF_BIN)

$(IVF_BIN):
	@echo "ivf_index.bin not found — running build_ivf.py (3-8 min)..."
	uv run tools/build_ivf.py

# ── Docker Compose (nginx:9999 → api1:3000 + api2:3000) ───────────────────

up:
	docker compose up --build -d
	@echo "Waiting for $(READY_URL)..."
	@until curl -sf $(READY_URL) > /dev/null 2>&1; do \
		printf '.'; sleep 1; \
	done
	@echo " ready."

down:
	docker compose down

# ── Local dev (single instance on port 9999, no Docker) ───────────────────

dev: $(BINARY) $(IVF_BIN)
	PORT=$(PORT) \
	IVF_PATH=$(IVF_BIN) \
	IVF_NPROBE=4 \
	MCC_PATH=resources/mcc_risk.json \
	NORM_PATH=resources/normalization.json \
	$(BINARY)

# ── k6 tests (servidor deve estar rodando em localhost:9999) ───────────────

smoke:
	k6 run test/smoke.js

load:
	k6 run test/test.js

# ── Misc ───────────────────────────────────────────────────────────────────

profile: $(PROFILING_BIN) $(IVF_BIN)
	@echo "Starting fraud-detection under samply. Run 'make load' in another terminal."
	PORT=$(PORT) \
	IVF_PATH=$(IVF_BIN) \
	MCC_PATH=resources/mcc_risk.json \
	NORM_PATH=resources/normalization.json \
	samply record $(PROFILING_BIN)

doc:
	cargo doc --open

clean:
	cargo clean
	rm -f $(IVF_BIN)
