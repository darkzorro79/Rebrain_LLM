from vllm import LLM, SamplingParams

# 1. Подготовка данных
# Мы подаем сразу список строк. vLLM сам соберет их в батч.
prompts = [
    "Привет, как твои дела?",
    "Объясни в одном предложении, что такое квантовая запутанность.",
    "Напиши короткое стихотворение про программиста и кофе."
]

# 2. Настройка параметров генерации
# temperature=0.8 — добавляем немного креативности
# top_p=0.95 — отсекаем совсем маловероятные токены
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000)

# 3. Инициализация движка (Engine)
# Здесь происходит магия: загрузка весов и выделение памяти под PagedAttention.
# Если у вас мало видеопамяти, vLLM может занять её всю.
# Можно добавить параметр gpu_memory_utilization=0.9
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")

# 4. Запуск генерации
# Это блокирующая операция. Движок обработает все промпты, используя Continuous Batching.
outputs = llm.generate(prompts, sampling_params)

# 5. Вывод результатов
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)
