#!/usr/bin/env python3
"""Run the frozen zero-shot POS evaluator with Llama 3.1 Instruct."""

import qwen_pos_eval


qwen_pos_eval.EVALUATOR_VERSION = "llama-3.1-8b-pos-zero-shot-v1"


if __name__ == "__main__":
    qwen_pos_eval.main()
