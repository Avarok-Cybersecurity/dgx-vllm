#!/usr/bin/env python3
"""Fix doubled prefix bug in Qwen3Next create_qkvz_proj method.

Bug: Both the caller and create_qkvz_proj append '.in_proj_qkvz' to prefix,
resulting in 'model.layers.X.linear_attn.in_proj_qkvz.in_proj_qkvz' which
doesn't match the quantization ignore list pattern.

Fix: Only change the prefix inside create_qkvz_proj method body, keeping
the caller's prefix=f"{prefix}.in_proj_qkvz" intact.
"""

path = "/app/vllm/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    content = f.read()

# Find create_qkvz_proj method and only fix the prefix inside it
method_start = content.find("def create_qkvz_proj(")
if method_start < 0:
    print("SKIP: create_qkvz_proj not found (model may not be in this vLLM version)")
    exit(0)

# Find the next method/class definition to bound our search
next_method = content.find("\ndef ", method_start + 1)
if next_method < 0:
    next_method = len(content)

method_body = content[method_start:next_method]

# Check if the doubled prefix exists in the method body
old_pattern = 'prefix=f"{prefix}.in_proj_qkvz"'
if old_pattern in method_body:
    fixed_body = method_body.replace(old_pattern, "prefix=prefix", 1)
    content = content[:method_start] + fixed_body + content[next_method:]
    with open(path, "w") as f:
        f.write(content)
    print("Fix applied: create_qkvz_proj doubled prefix removed")
else:
    print("SKIP: pattern not found in create_qkvz_proj (may already be fixed)")
