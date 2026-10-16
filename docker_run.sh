docker volume create \
  --driver local \
  --opt type=none \
  --opt device=/data0/ss84435910/ucm_docker/ascend-bandwidth-bench/prometheus-data \
  --opt o=bind \
  prometheus-data

docker volume create \
  --driver local \
  --opt type=none \
  --opt device=/data0/ss84435910/ucm_docker/ascend-bandwidth-bench/grafana-data \
  --opt o=bind \
  grafana-data

docker run -d --name prometheus --restart unless-stopped \
  --network ucm-monitoring -p 9090:9090 \
  -v "/data0/ss84435910/ucm_docker/ascend-bandwidth-bench/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v prometheus-data:/prometheus \
  quay.io/prometheus/prometheus:latest

docker run -d --name grafana --restart unless-stopped \
  --network ucm-monitoring -p 3000:3000 \
  -v grafana-data:/var/lib/grafana \
  quay.io/ceph/grafana:9.5.21   # Change it to whatever version you want