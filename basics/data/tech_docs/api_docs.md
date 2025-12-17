# NebulaDB API Documentation

## Introduction
NebulaDB is a high-performance distributed key-value store designed for interstellar latency.

## Authentication
All requests must include the `X-Nebula-Token` header.
```bash
curl -H "X-Nebula-Token: <your_token>" https://api.nebuladb.com/v1/stats
```

## Endpoints

### GET /v1/clusters
Retrieves a list of all active clusters.

**Parameters:**
- `region` (optional): Filter by region code (e.g., `us-east-1`, `mars-north-2`).

**Response:**
```json
{
  "clusters": [
    {"id": "c-123", "status": "active", "region": "us-east-1"},
    {"id": "c-456", "status": "provisioning", "region": "mars-north-2"}
  ]
}
```

### POST /v1/clusters
Creates a new cluster.

**Body:**
- `name` (required): Name of the cluster.
- `node_count` (required): Number of nodes (min: 3, max: 100).

## Error Codes
- `400`: Invalid parameters (e.g., node_count < 3).
- `401`: unauthorized.
- `429`: Rate limit exceeded (100 req/sec per token).
