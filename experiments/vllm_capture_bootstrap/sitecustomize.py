"""Early worker bootstrap used only by DTGB final-hidden-state capture."""

import os


if os.environ.get("DTGB_VLLM_CAPTURE_FINAL_HIDDEN") == "1":
    try:
        from experiments.modules.llm_lp.vllm_final_hidden_patch import (
            install_llama_final_hidden_state_capture,
        )

        install_llama_final_hidden_state_capture()
    except ModuleNotFoundError as exc:
        # ``conda run`` may invoke helper interpreters outside the target env
        # before starting its vLLM Python. Those helpers do not have vLLM and
        # do not execute model code, so they should ignore the capture hook.
        if exc.name != "vllm":
            raise
