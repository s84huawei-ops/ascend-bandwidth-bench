## 0. Preassumption

Setup UCM+vLLM inside of docker container with `--net=host` (in docker run command)

## 1. Enable metrics on the vLLM/UCM side (if not already)

In your UCM config, metrics are on by default, but explicitly:
```yaml
enable_metrics: true
```
Then hit your vLLM container to confirm it's actually exporting UCM series:
```bash
curl http://<vllm-ip>:<vllm-port>/metrics | grep '^ucm:'
```
Metrics only appear after at least one inference request has run through — an idle server shows nothing.

## 2. Networking — connect dokcer container with your current UCM setup

Prometheus and Grafana themselves are new, separate containers you spin up on `ucm-monitoring`:

*You can pull docker images from here:*
*- https://quay.io/repository/ceph/grafana?tab=tags*
*- https://quay.io/repository/prometheus/prometheus?tab=tags*

create file `prometheus.yml`:
```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    static_configs:
      - targets:
          - "<your-vllm-container-name-or-host-ip>:<port>"      # Usually it is 172.17.0.1
```

```bash
docker network create ucm-monitoring
docker volume create prometheus-data
docker volume create grafana-data

docker run -d --name prometheus --restart unless-stopped \
  --network ucm-monitoring -p 9090:9090 \
  -v "/path/to/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v prometheus-data:/prometheus \
  quay.io/prometheus/prometheus:latest

docker run -d --name grafana --restart unless-stopped \
  --network ucm-monitoring -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  quay.io/ceph/grafana:9.5.21   # Change it to whatever version you want
```

Check `http://<host>:9090/targets` — the `vllm` job should show **UP**.

## 3. Grafana setup

- Log in at `http://<host>:3000` (admin/admin first time).
- Add a Prometheus data source with URL `http://prometheus:9090` (this hostname resolves because Grafana and Prometheus share the `ucm-monitoring` network).
- Import UCM's prebuilt dashboards from the repo: `examples/metrics/grafana_vllm.json`, `grafana_store.json`, and `grafana_connector.json` (for bandwidth measurement `grafana_store.json` is the most interesting).

**Note:** If grafana displays "No Data" instead of plots, check if the `model_name` field in upper bar is set to None. If so, change it to model path that you have passed to vllm. 

## 4. Which metrics actually show HBM↔Host↔SSD transfer

Since we're specifically after HBM-Host and Host-SSD movement, the relevant catalog sections are:

- **HBM ↔ Host (device copies):** Cache Store histograms — `ucm:cache_h2d_submit_ms`, `ucm:cache_h2d_sync_ms` (host→device), and `ucm:cache_d2h_duration_ms`, `ucm:cache_dump_mkbuf_duration_ms` (device→host). `ucm:cache_load_bandwidth_gbps` / `ucm:cache_dump_bandwidth_gbps` give effective per-task bandwidth for the whole Cache-stage task.
- **Host ↔ SSD:** if your backend is the Posix store, use `ucm:posix_s2h_bytes_total` (SSD→host reads) and `ucm:posix_h2s_bytes_total` (host→SSD writes), plus the matching `ucm:posix_s2h_bandwidth_gbps` / `ucm:posix_h2s_bandwidth_gbps` histograms for per-task speed. For aggregate throughput rather than per-task speed, use `rate()` on the `_bytes_total` counters divided by 1e9, not the bandwidth histograms.

A caveat from the docs worth keeping in mind: these Posix byte counters count nominal task bytes at completion (including aborted/error paths), so for true physical SSD throughput you'd want to cross-check with filesystem/device-level observation (e.g. `iostat`) alongside these.

If you're not using the Posix backend (e.g. Mooncake or YuanRong instead), the equivalent host↔SSD-tier metrics live under those sections instead (`mooncake_h2d_bytes_total`/`mooncake_d2h_bytes_total`, or the YuanRong `*_ssd_*` gauges) — let me know which backend you've configured and I can point you at the exact ones.
