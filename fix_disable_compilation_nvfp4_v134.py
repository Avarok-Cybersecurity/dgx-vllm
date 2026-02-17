#!/usr/bin/env python3
"""
v134: Disable torch.compile for NVFP4 to bypass AutogradCUDA dispatch issue

The operator is fundamentally registered for AutogradCUDA, not plain CUDA.
Since we can't fix the registration (v128-v133 all failed), disable compilation
for NVFP4 quantization to allow eager execution to work.
"""
import os
import re

vllm_dir = "/app/vllm"

print("=" * 70)
print("v134: Disable torch.compile for NVFP4")
print("=" * 70)
print()
print("Problem: cutlass_fp4_group_mm registered for AutogradCUDA, not CUDA")
print("Solution: Disable torch.compile when quantization=modelopt_fp4")
print()

# Modify CompilationConfig to disable for NVFP4
config_file = f"{vllm_dir}/vllm/config/compilation.py"

with open(config_file, 'r') as f:
    content = f.read()

# Find __post_init__ and add NVFP4 check
pattern = r'(def __post_init__\(self\) -> None:.*?\n)(        # Resolve)'

replacement = r'''\1        # v134: Disable compilation for NVFP4
        if hasattr(self, '_model_config') and self._model_config:
            if getattr(self._model_config, 'quantization', None) == 'modelopt_fp4':
                import logging
                logger = logging.getLogger(__name__)
                logger.info("v134: Disabling torch.compile for NVFP4 (AutogradCUDA dispatch workaround)")
                self.level = None
                return
        
\2'''

if "v134: Disable compilation for NVFP4" not in content:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Added NVFP4 compilation disable to CompilationConfig.__post_init__")
else:
    print("ℹ️  Already added")

# Also add to from_model_config to be safe
pattern2 = r'(def from_model_config\(cls, model_config.*?\n.*?\n)(        return cls\()'

replacement2 = r'''\1        # v134: Disable for NVFP4
        if model_config.quantization == 'modelopt_fp4':
            import logging
            logger = logging.getLogger(__name__)
            logger.info("v134: NVFP4 detected, forcing compilation level=None")
            level = None
        
\2'''

if "v134: NVFP4 detected" not in content:
    with open(config_file, 'r') as f:
        content = f.read()
    
    content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Added NVFP4 check to from_model_config")
else:
    print("ℹ️  Already added to from_model_config")

print()
print("=" * 70)
print("✅ v134 fix applied!")
print("=" * 70)
print()
print("Effect: torch.compile will be completely disabled for NVFP4 models")
print("Models will run in eager mode, avoiding the AutogradCUDA dispatch issue.")
