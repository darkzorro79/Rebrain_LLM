python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key "my-secret-token" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
