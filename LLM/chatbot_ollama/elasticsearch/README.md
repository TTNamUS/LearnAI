

### Integrating with Elasticsearch

To integrate Ollama with Elasticsearch using Docker-Compose, follow these steps:

1. **Create a Docker-Compose File**: Create a file named `docker-compose.yaml` with the following content:

```yaml
version: '3.8'

services:
    elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.4.3
    container_name: elasticsearch
    environment:
        - discovery.type=single-node
        - xpack.security.enabled=false
    ports:
        - "9200:9200"
        - "9300:9300"

    ollama:
    image: ollama/ollama
    container_name: ollama
    volumes:
        - ollama:/root/.ollama
    ports:
        - "11434:11434"

volumes:
    ollama:
```

2. **Run Docker-Compose**: Execute the following command to start the services:

```bash
docker-compose up
```

3. **Re-run the Module 1 Notebook**: Ensure everything is set up correctly by re-running the module 1 notebook.

## Conclusion

By following these steps, you can set up and use Ollama to run LLMs on a CPU, download models, and integrate with Elasticsearch.
