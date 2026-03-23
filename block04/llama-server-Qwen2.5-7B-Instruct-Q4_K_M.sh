sudo systemctl stop ollama
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M -ub 2048 -b 2048 -c 15000 --jinja --host 0.0.0.0 --port 8080
