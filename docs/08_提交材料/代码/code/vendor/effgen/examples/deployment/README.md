# effGen Deployment Guide

## Quick Start (Docker Compose)

```bash
# Clone the repo and navigate to deployment examples
cd examples/deployment

# Set your API key (optional — protects the endpoint)
export EFFGEN_API_KEY=my-secret-key

# Start the stack
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| effGen API | 8000 | REST + WebSocket agent endpoint |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards (login: admin/admin) |

## Configuration

Environment variables for the API server:

| Variable | Default | Description |
|----------|---------|-------------|
| `EFFGEN_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Model to load |
| `EFFGEN_API_KEY` | `changeme` | API authentication key |
| `EFFGEN_RATE_LIMIT` | `60` | Requests per minute |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU(s) to use |

## GPU Support

The compose file includes NVIDIA GPU reservations. Ensure you have:
- NVIDIA Container Toolkit installed
- Docker >= 19.03

## Production Notes

- Use a reverse proxy (nginx/caddy) for TLS termination
- Set strong `EFFGEN_API_KEY` and `GRAFANA_PASSWORD` values
- Mount persistent volumes for model weights to avoid re-downloading
- Monitor `/metrics` endpoint with Prometheus for latency, token usage, and errors
