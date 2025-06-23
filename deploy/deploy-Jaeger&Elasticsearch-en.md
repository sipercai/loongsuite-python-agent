# Jaeger + Elasticsearch Deployment Guide

### K8S Deployment (kubectl)

1. Create namespace

```yaml
kubectl create namespace jaeger-system
```

2. Check available storage classes for setting the storageClassName of PersistentVolumeClaim for ElasticSearch.

```yaml
kubectl get sc
```

3. Create a jaeger-es-config.yaml file and add the following content

> Here we use the storage class: alicloud-disk-essd

```yaml
# ConfigMap for Jaeger config
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
  namespace: jaeger-system
data:
  config.yaml: |
    service:
      extensions: [jaeger_storage, jaeger_query, healthcheckv2]
      pipelines:
        traces: 
          receivers: [otlp]
          processors: [batch]
          exporters: [jaeger_storage_exporter]
      telemetry:
        resource:
          service.name: jaeger
        logs:
          level: debug

    extensions:
      healthcheckv2:
        use_v2: true
        http:

      jaeger_query:
        storage:
          traces: some_storage
          traces_archive: another_storage

      jaeger_storage:
        backends:
          some_storage:
            elasticsearch:
              server_urls:
                - http://elasticsearch:9200
              indices:
                index_prefix: "jaeger-main"
                spans:
                  date_layout: "2006-01-02"
                  rollover_frequency: "day"
                  shards: 5
                  replicas: 1
                services:
                  date_layout: "2006-01-02"
                  rollover_frequency: "day"
                  shards: 5
                  replicas: 1
                dependencies:
                  date_layout: "2006-01-02"
                  rollover_frequency: "day"
                  shards: 5
                  replicas: 1
                sampling:
                  date_layout: "2006-01-02"
                  rollover_frequency: "day"
                  shards: 5
                  replicas: 1
          another_storage:
            elasticsearch:
              server_urls:
                - http://elasticsearch:9200
              indices:
                index_prefix: "jaeger-archive"

    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: "0.0.0.0:4317"
          http:
            endpoint: "0.0.0.0:4318"
            
    processors:
      batch:

    exporters:
      jaeger_storage_exporter:
        trace_storage: some_storage
---
# Elasticsearch Deployment and Service
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: elasticsearch-data
  namespace: jaeger-system
spec:
  storageClassName: alicloud-disk-essd  # Use ESSD cloud disk
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: jaeger-system
spec:
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      initContainers:
      - name: fix-permissions
        image: busybox
        command: ["sh", "-c", "chown -R 1000:1000 /usr/share/elasticsearch/data"]
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:7.5.2
        env:
        - name: discovery.type
          value: single-node
        - name: ES_JAVA_OPTS
          value: "-Xms512m -Xmx512m"
        - name: xpack.security.enabled
          value: "false"
        ports:
        - containerPort: 9200
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
        livenessProbe:          
          httpGet:
            path: /_cluster/health
            port: 9200
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: elasticsearch-data
---
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch
  namespace: jaeger-system
spec:
  ports:
  - port: 9200
    targetPort: 9200
  selector:
    app: elasticsearch
---
# Jaeger Deployment and Service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: jaeger-system
spec:
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/jaeger:latest
        args:
        - --config=/jaeger/config.yaml
        env:
        - name: LOG_LEVEL
          value: debug
        ports:
        - containerPort: 16686
        - containerPort: 4317
        - containerPort: 4318
        volumeMounts:
        - name: config
          mountPath: /jaeger/config.yaml
          subPath: config.yaml
      volumes:
      - name: config
        configMap:
          name: jaeger-config
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger
  namespace: jaeger-system
spec:
  ports:
  - name: ui
    port: 16686
    targetPort: 16686
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  selector:
    app: jaeger

```

4. Deploy

```yaml
kubectl apply -f jaeger-es-config.yaml
```

5. Check deployment status

* Basic checks

```plaintext
# Check Pod status
kubectl get pods -n jaeger-system

# Check Service status
kubectl get svc -n jaeger-system

# Check ConfigMap
kubectl get configmap -n jaeger-system

# Check PVC status
kubectl get pvc -n jaeger-system
```

* After reporting data, check ES indexes

```yaml
# kubectl exec -it $(kubectl get pod -l app=elasticsearch -n jaeger-system -o jsonpath='{.items[0].metadata.name}') -n jaeger-system -- curl localhost:9200/_cat/indices
Defaulted container "elasticsearch" out of: elasticsearch, fix-permissions (init)
yellow open jaeger-main-jaeger-service-2025-05-08 cWBu77OHQ4eljaOSAtf7wg 5 1   25 0 48.1kb 48.1kb
```

### K8S Deployment (helm)

[https://github.com/jaegertracing/helm-charts/tree/v2](https://github.com/jaegertracing/helm-charts/tree/v2)

1. Create namespace

```yaml
kubectl create namespace jaeger-system
```

2. Add helm repositories

```yaml
# Add Elastic repository
helm repo add elastic https://helm.elastic.co

# Add Jaeger repository
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts

# Update repositories
helm repo update
```

3. Create values.yaml for Elasticsearch (elasticsearch-values.yaml):

* First confirm the storage class, then configure volumeClaimTemplate.storageClassName, here we use alicloud-disk-essd

```yaml
clusterName: "elasticsearch"
nodeGroup: "master"
replicas: 1

# Resource configuration
resources:
  requests:
    cpu: "100m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "512Mi"

# Storage configuration, need to replace ${StorageClassName}
volumeClaimTemplate:
  accessModes: [ "ReadWriteOnce" ]
  storageClassName: "${StorageClassName}"
  resources:
    requests:
      storage: 20Gi

# ES configuration
esConfig:
  elasticsearch.yml: |
    xpack.security.enabled: false
    discovery.type: single-node

# JVM configuration
esJavaOpts: "-Xmx512m -Xms512m"
```

4. Create values.yaml for Jaeger (jaeger-values.yaml):

```yaml
provisionDataStore:
  cassandra: false
  elasticsearch: false  # Install ES separately

storage:
  type: elasticsearch
  options:
    es:
      server-urls: http://elasticsearch-master:9200
      index-prefix: jaeger-main

# OTLP configuration
collector:
  enabled: true
  service:
    otlp:
      grpc:
        port: 4317
      http:
        port: 4318

query:
  enabled: true
  service:
    type: ClusterIP
  ingress:
    enabled: true
    ingressClassName: mse
    hosts:
      - jaeger.example.com

ingress:
  enabled: true
  ingressClassName: mse
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "0"
  hosts:
    - host: jaeger-collector.example.com
      paths:
        - path: /
          pathType: Prefix
          port: 4318

agent:
  enabled: false 

sampling:
  type: probabilistic
  param: 1.0
```

5. Install ES

```yaml
helm install elasticsearch elastic/elasticsearch \
  --namespace jaeger-system \
  -f elasticsearch-values.yaml
```

6. Install Jaeger

```yaml
helm install jaeger jaegertracing/jaeger \
  --namespace jaeger-system \
  -f jaeger-values.yaml
```

7. Verify installation

```yaml
# Check if all pods are running normally
kubectl get pods -n jaeger-system

# Check services
kubectl get svc -n jaeger-system

# Check ingress
kubectl get ingress -n jaeger-system
```

### Docker Compose Deployment

1. Create Jaeger collector configuration

> Based on official yaml configuration (Change: ES address changed from localhost to elasticsearch): [https://github.com/jaegertracing/jaeger/blob/v2.5.0/cmd/jaeger/config-elasticsearch.yaml](https://github.com/jaegertracing/jaeger/blob/v2.5.0/cmd/jaeger/config-elasticsearch.yaml)

```plaintext
service:
  extensions: [jaeger_storage, jaeger_query, healthcheckv2]
  pipelines:
    traces: 
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger_storage_exporter]
  telemetry:
    resource:
      service.name: jaeger

    logs:
      level: debug

extensions:
  healthcheckv2:
    use_v2: true
    http:

  jaeger_query:
    storage:
      traces: some_storage
      traces_archive: another_storage

  jaeger_storage:
    backends:
      some_storage:
        elasticsearch:
          server_urls:
            - http://elasticsearch:9200
          indices:
            index_prefix: "jaeger-main"
            spans:
              date_layout: "2006-01-02"
              rollover_frequency: "day"
              shards: 5
              replicas: 1
            services:
              date_layout: "2006-01-02"
              rollover_frequency: "day"
              shards: 5
              replicas: 1
            dependencies:
              date_layout: "2006-01-02"
              rollover_frequency: "day"
              shards: 5
              replicas: 1
            sampling:
              date_layout: "2006-01-02"
              rollover_frequency: "day"
              shards: 5
              replicas: 1
      another_storage:
        elasticsearch:
          server_urls:
            - http://elasticsearch:9200
          indices:
            index_prefix: "jaeger-archive"

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:
        endpoint: "0.0.0.0:4318"
        
processors:
  batch:

exporters:
  jaeger_storage_exporter:
    trace_storage: some_storage
```

2. Create docker-compose.yaml (with demo application)

```yaml
# To run a specific version of Jaeger, use environment variable, e.g.:
#     JAEGER_VERSION=2.0.0 HOTROD_VERSION=1.63.0 docker compose up

services:

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.5.2
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m  # Control memory usage
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    networks:
      - jaeger-example
    volumes:
      - esdata:/usr/share/elasticsearch/data  # Data persistence
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q '\"status\":\"green\"'"]
      interval: 10s
      timeout: 10s
      retries: 5
      
  jaeger:
    image: ${REGISTRY:-}jaegertracing/jaeger:${JAEGER_VERSION:-latest}
    ports:
      - "16686:16686"
      - "4317:4317"
      - "4318:4318"
    volumes:
      - ./config.yaml:/jaeger/config.yaml
    command:
      - --config=/jaeger/config.yaml
    environment:
      - LOG_LEVEL=debug
    networks:
      - jaeger-example
    depends_on:
      elasticsearch:
        condition: service_healthy  # Wait for ES health check to pass
    
  hotrod:
    image: ${REGISTRY:-}jaegertracing/example-hotrod:${HOTROD_VERSION:-latest}
    # To run the latest trunk build, find the tag at Docker Hub and use the line below
    # https://hub.docker.com/r/jaegertracing/example-hotrod-snapshot/tags
    #image: jaegertracing/example-hotrod-snapshot:0ab8f2fcb12ff0d10830c1ee3bb52b745522db6c
    ports:
      - "8080:8080"
      - "8083:8083"
    command: ["all"]
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
    networks:
      - jaeger-example
    depends_on:
      - jaeger

networks:
  jaeger-example:

volumes:
  esdata:
    driver: local
```

3. Start all services

```plaintext
# Start
docker compose up -d

# View logs
docker compose logs -f

# Destroy
docker compose down
```

4. After generating test data, check ES indexes

```yaml
# View all indexes
curl -X GET "localhost:9200/_cat/indices?v"
# curl -X GET "localhost:9200/_cat/indices?v"
health status index                                 uuid                   pri rep docs.count docs.deleted store.size pri.store.size
yellow open   jaeger-main-jaeger-service-2025-05-07 Da5bpM4oSrGnn1C3cQMFJw   5   1         11            0     20.6kb         20.6kb
yellow open   jaeger-main-jaeger-span-2025-05-07    PkPgQl1-QtmGNpWlCEvioQ   5   1       2648            0    464.8kb        464.8kb



# View jaeger related indexes
curl -X GET "localhost:9200/jaeger-*/_search?pretty"

``` 