# Jaeger+部署指南：使用 Elasticsearch 作为存储后端

### K8S 部署（kubectl）

1.  创建命名空间
    

```yaml
kubectl create namespace jaeger-system
```

2.  查看可用的存储类，用于设置 PersistentVolumeClaim 的 storageClassName，给 ElasticSearch 用。
    

```yaml
kubectl get sc
```

3.  创建一个 jaeger-es-config.yaml 文件，添加以下内容
    

> 注意：将 `${StorageClassName}` 替换为第 2 步获取的存储类

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
  storageClassName:  ${StorageClassName}  # 使用集群中可用的存储类
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

4.  部署
    

```yaml
kubectl apply -f jaeger-es-config.yaml
```

5.  检查部署状态
    

*   基础检查
    

```plaintext
# 检查 Pod 状态
kubectl get pods -n jaeger-system

# 检查 Service 状态
kubectl get svc -n jaeger-system

# 检查 ConfigMap
kubectl get configmap -n jaeger-system

# 检查 PVC 状态
kubectl get pvc -n jaeger-system
```

*   上报数据后，检查 es 索引
    

```yaml
# kubectl exec -it $(kubectl get pod -l app=elasticsearch -n jaeger-system -o jsonpath='{.items[0].metadata.name}') -n jaeger-system -- curl localhost:9200/_cat/indices
Defaulted container "elasticsearch" out of: elasticsearch, fix-permissions (init)
yellow open jaeger-main-jaeger-service-2025-05-08 cWBu77OHQ4eljaOSAtf7wg 5 1   25 0 48.1kb 48.1kb
```

### K8S 部署（helm）

[https://github.com/jaegertracing/helm-charts/tree/v2](https://github.com/jaegertracing/helm-charts/tree/v2)

1.   创建命名空间
    

```yaml
kubectl create namespace jaeger-system
```

2.   添加 helm 仓库
    

```yaml
# 添加 Elastic 仓库
helm repo add elastic https://helm.elastic.co

# 添加 Jaeger 仓库
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts

# 更新仓库
helm repo update
```

3.   为 Elasticsearch 创建 values.yaml (elasticsearch-values.yaml):
    

*   请先确认集群中可用的存储类，然后在 volumeClaimTemplate 配置中将占位符 "${StorageClassName}" 替换为实际的存储类名称。
    

```yaml
clusterName: "elasticsearch"
nodeGroup: "master"
replicas: 1

# 资源配置
resources:
  requests:
    cpu: "100m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "512Mi"

# 存储配置，需要替换 ${StorageClassName}
volumeClaimTemplate:
  accessModes: [ "ReadWriteOnce" ]
  storageClassName: "${StorageClassName}"
  resources:
    requests:
      storage: 20Gi

# ES配置
esConfig:
  elasticsearch.yml: |
    xpack.security.enabled: false
    discovery.type: single-node

# JVM配置
esJavaOpts: "-Xmx512m -Xms512m"
```

4.   为 Jaeger 创建 values.yaml (jaeger-values.yaml):
    

```yaml
provisionDataStore:
  cassandra: false
  elasticsearch: false  # 单独安装ES

storage:
  type: elasticsearch
  options:
    es:
      server-urls: http://elasticsearch-master:9200
      index-prefix: jaeger-main

# OTLP配置
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

5.   安装 es
    

```yaml
helm install elasticsearch elastic/elasticsearch \
  --namespace jaeger-system \
  -f elasticsearch-values.yaml
```

6.   安装 jaeger
    

```yaml
helm install jaeger jaegertracing/jaeger \
  --namespace jaeger-system \
  -f jaeger-values.yaml
```

7.   验证安装
    

```yaml
# 检查所有 pods 是否正常运行
kubectl get pods -n jaeger-system

# 检查服务
kubectl get svc -n jaeger-system

# 检查 ingress
kubectl get ingress -n jaeger-system
```

### Docker Compose 部署

1.  创建 jaeger collector 配置
    

基于 [Jaeger 官方 Elasticsearch 配置模板](https://github.com/jaegertracing/jaeger/blob/v2.5.0/cmd/jaeger/config-elasticsearch.yaml) 调整，主要修改了 Elasticsearch 服务地址配置。
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

2.  创建 docker-compose.yaml
    

```yaml
# To run a specific version of Jaeger, use environment variable, e.g.:
#     JAEGER_VERSION=2.0.0 HOTROD_VERSION=1.63.0 docker compose up

services:

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.5.2
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m  # 控制内存使用
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    networks:
      - jaeger-example
    volumes:
      - esdata:/usr/share/elasticsearch/data  # 数据持久化
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
        condition: service_healthy  # 等待 ES 健康检查通过
    

networks:
  jaeger-example:

volumes:
  esdata:
    driver: local
```

3.  启动所有服务
    

```plaintext
# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 销毁
docker compose down
```

4.  生成测试数据后，查看 es 索引
    

```yaml
# 查看所有索引
curl -X GET "localhost:9200/_cat/indices?v"
health status index                                 uuid                   pri rep docs.count docs.deleted store.size pri.store.size
yellow open   jaeger-main-jaeger-service-xxxx-xx-xx  xxxxxxxxxxxxxxxxxxxx   5   1         11            0     20.6kb         20.6kb
yellow open   jaeger-main-jaeger-span-xxxx-xx-xx     xxxxxxxxxxxxxxxxxxxx   5   1       2648            0    464.8kb        464.8kb



# 查看 jaeger 相关索引
curl -X GET "localhost:9200/jaeger-*/_search?pretty"

```