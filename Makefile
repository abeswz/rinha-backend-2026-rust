BINARY        := target/release/fraud-detection
GIT_SHA       := $(shell git rev-parse --short HEAD)
IMAGE         := ghcr.io/abeswz/fraud-detection-rinha-backend-2026:$(GIT_SHA)
PORT          := 9999
READY_TIMEOUT := 300

.PHONY: build clean down smoke load score bench profile publish submission

build:
	cargo build --release

clean:
	cargo clean

down:
	docker compose --compatibility down

bench: down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready > /dev/null 2>&1; do \
		printf '.'; sleep 1; \
		i=$$((i+1)); \
		if [ $$i -ge $(READY_TIMEOUT) ]; then echo " timeout"; exit 1; fi; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

smoke:
	k6 run test/smoke.js

load:
	k6 run test/test.js

score:
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors) p99_score:\(.scoring.p99_score.value) det_score:\(.scoring.detection_score.value)"' test/results.json

profile: build
	SOCK=/tmp/profile.sock samply record $(BINARY)

publish:
	docker build --network=host -t $(IMAGE) .
	docker push $(IMAGE)

submission: clean info.json
	$(MAKE) publish
	@ORIG=$$(git rev-parse --abbrev-ref HEAD); \
	sed 's|build: \.|image: $(IMAGE)|' docker-compose.yml > /tmp/sub-compose.yml; \
	cp info.json /tmp/sub-info.json; \
	git checkout --orphan submission-tmp; \
	git rm -rf . > /dev/null 2>&1; \
	cp /tmp/sub-compose.yml docker-compose.yml; \
	cp /tmp/sub-info.json info.json; \
	git add docker-compose.yml info.json; \
	git commit -m "submission: docker-compose.yml, info.json"; \
	git branch -D submission 2>/dev/null || true; \
	git branch -m submission-tmp submission; \
	git push origin submission --force; \
	git checkout $$ORIG; \
	echo "done"
