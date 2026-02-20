"""
vLLM Scheduler Demo — 2-request batch on macOS (CPU).

Run this from PyCharm with the debugger attached.
Breakpoints are inserted directly into the scheduler code.

Interesting things to inspect at each breakpoint:
  - self.waiting / self.running    → queue states
  - num_scheduled_tokens           → token budget allocation
  - scheduler_output               → the final scheduling decision
  - request.num_computed_tokens    → progress per request
  - request.status                 → WAITING / RUNNING / FINISHED
"""

from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",     # ~250MB, runs fine on CPU
    max_model_len=256,             # keep short for demo
    max_num_seqs=4,                # small batch cap
    enforce_eager=True,            # skip torch.compile for faster startup
    dtype="float32",               # safest for CPU on macOS
)

# Two requests with different prompt lengths to see interesting scheduling
prompts = [
    "The capital of France is",                          # short prompt
    "In a galaxy far far away, there once lived a very", # longer prompt
]
params = SamplingParams(max_tokens=16, temperature=0.0)

print(">>> Generating — breakpoints will fire in the scheduler")
outputs = llm.generate(prompts, params)

for out in outputs:
    print(f"\n[{out.request_id}]")
    print(f"  prompt:  {out.prompt!r}")
    print(f"  output:  {out.outputs[0].text!r}")
