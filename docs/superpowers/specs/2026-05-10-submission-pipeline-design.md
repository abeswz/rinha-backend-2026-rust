# Submission Pipeline Design

**Date:** 2026-05-10
**Topic:** Rinha de Backend 2026 — submission pipeline

---

## Goal

Prepare the project for official submission to Rinha de Backend 2026:
- Build and publish Docker image to GHCR
- Create isolated `submission` branch with only required files
- Add `info.json` to both branches
- Enable preview test via GitHub issue

---

## Approach

Makefile targets. Zero extra infrastructure. Builds on existing `Makefile`.

---

## Components

### 1. `make publish`

Builds multi-stage Docker image and pushes to GitHub Container Registry.

```makefile
IMAGE = ghcr.io/abeswz/fraud-detection-rinha-backend-2026:latest

publish:
    docker build -t $(IMAGE) .
    docker push $(IMAGE)
```

Pre-requisite (one-time):
```bash
docker login ghcr.io -u abeswz -p <GITHUB_PAT>
```
PAT scope required: `write:packages`.

After first push: set package visibility to public in GitHub → Packages → Settings.

### 2. `make submission`

Creates/resets `submission` branch as orphan with exactly 3 files:

| File | Source |
|------|--------|
| `docker-compose.yml` | Generated — uses `image:` instead of `build:` |
| `nginx.conf` | Copied from `main` |
| `info.json` | New file |

The submission `docker-compose.yml` replaces `build: .` with:
```yaml
image: ghcr.io/abeswz/fraud-detection-rinha-backend-2026:latest
```

All resource limits remain identical to `main` docker-compose.yml:
- nginx: 0.05 CPU, 10MB
- api1/api2: 0.475 CPU, 170MB each

Branch is force-pushed (orphan, no source code, no git history from main).

### 3. `info.json`

Added to `main` branch first, then included in `submission` branch by the Makefile target.

```json
{
    "participants": ["abesnow"],
    "social": ["https://github.com/abeswz", "https://www.linkedin.com/in/anves"],
    "source-code-repo": "https://github.com/abeswz/fraud-detection-rinha-backend-2026",
    "stack": ["rust", "nginx"],
    "open_to_work": false
}
```

---

## Submission Flow

1. `make publish` — build + push image to GHCR
2. `make submission` — create/update `submission` branch
3. Open PR on rinha repo adding `participants/abeswz.json`
4. Open issue on rinha repo for preview test with body: `rinha/test abeswz-rust`

---

## Constraints

- `submission` branch must contain no source code
- `docker-compose.yml` in `submission` must use pre-built image (not `build:`)
- Image must be publicly accessible on GHCR before test runs
- `info.json` must exist in both `main` and `submission`
