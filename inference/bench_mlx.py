#!/usr/bin/env python3
"""Benchmark raw MLX inference for Qwen2.5-0.5B-Instruct Q4."""
import time
import json
import mlx_lm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPT_SHORT = "Explain quantum computing in simple terms."
PROMPT_LONG = "Write a detailed essay about the history of artificial intelligence, covering its origins in the 1950s, the development of expert systems in the 1980s, the AI winter, the resurgence with deep learning in the 2010s, and the current state of large language models. Include key figures like Alan Turing, John McCarthy, Geoffrey Hinton, and Yann LeCun. Discuss the ethical implications of AI development and its potential impact on society. " * 3
MAX_TOKENS = 200
N_RUNS = 5

print(f"Loading model: {MODEL}")
t0 = time.perf_counter()
model, tokenizer = mlx_lm.load(MODEL)
load_time = time.perf_counter() - t0
print(f"Model loaded in {load_time:.2f}s")

results = []
for prompt_name, prompt in [("short", PROMPT_SHORT), ("long", PROMPT_LONG)]:
    prompt_tokens = tokenizer.encode(prompt)
    n_prompt = len(prompt_tokens)
    print(f"\n=== Prompt: {prompt_name} ({n_prompt} tokens) ===")

    for run in range(N_RUNS):
        t_start = time.perf_counter()
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=True
        )
        t_end = time.perf_counter()
        wall_ms = (t_end - t_start) * 1000
        n_gen = len(tokenizer.encode(response)) if response else 0
        print(f"  Run {run+1}: {wall_ms:.1f}ms wall, ~{n_gen} tokens generated")
        results.append({
            "engine": "mlx-raw",
            "prompt": prompt_name,
            "run": run + 1,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "wall_ms": round(wall_ms, 1),
            "load_s": round(load_time, 2),
        })

print("\n=== RAW RESULTS ===")
print(json.dumps(results, indent=2))
