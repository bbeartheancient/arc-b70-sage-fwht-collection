# SPDX-License-Identifier: Apache-2.0
"""Setup script for xpu_flash_attn.

Install with:
    pip install --no-build-isolation -e .
"""
from setuptools import setup, find_packages

setup(
    name="xpu-flash-attn",
    version="0.1.0",
    description="FlashAttention-compatible interface for Intel XPU via SDPA",
    author="Local build",
    packages=find_packages(include=["xpu_flash_attn", "xpu_flash_attn.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.2"],
)
