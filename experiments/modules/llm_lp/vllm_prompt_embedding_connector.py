"""Low-I/O vLLM connector for one last-prompt-token hidden state per request.

vLLM's public ``extract_hidden_states`` path exposes Llama auxiliary states at
decoder-block *inputs*.  The representation used by the language-model head is
instead the state after the last decoder block and final RMSNorm.  This module
installs a narrow runtime shim that lets the existing extraction path request
that final state with the sentinel layer id ``num_hidden_layers``.
"""

import os
from typing import Any

import safetensors.torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from experiments.modules.llm_lp.vllm_final_hidden_patch import (
    install_llama_final_hidden_state_capture,
)


# This is a fallback for in-process engines. Spawned workers receive the same
# patch earlier through experiments/vllm_capture_bootstrap/sitecustomize.py.
install_llama_final_hidden_state_capture()


class LastTokenHiddenStatesConnector(ExampleHiddenStatesConnector):
    """Store only the requested layers for the final prompt token, once."""

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """Build metadata only for new/full-prefill requests.

        The stock example connector also re-adds every cached decode request
        and assumes it allocated a new block. Normal decode steps often do not,
        and the last prompt-token state is already persisted by that point.
        """
        metadata = ExampleHiddenStatesConnectorMetadata()
        for new_request in scheduler_output.scheduled_new_reqs:
            token_ids = new_request.prompt_token_ids or []
            filename = os.path.join(
                self._storage_path, f"{new_request.req_id}.safetensors"
            )
            metadata.add_request(
                new_request.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_request.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_request.req_id] = filename
            self._active_requests[new_request.req_id] = new_request
            self._req_blocks[new_request.req_id] = list(new_request.block_ids[0])
        return metadata

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        if not isinstance(attn_metadata, CacheOnlyAttentionMetadata):
            raise TypeError(
                "LastTokenHiddenStatesConnector requires CacheOnlyAttentionMetadata."
            )
        connector_metadata = self._get_connector_metadata()
        if not isinstance(connector_metadata, ExampleHiddenStatesConnectorMetadata):
            raise TypeError("Unexpected connector metadata type.")

        os.makedirs(self._storage_path, exist_ok=True)
        flat_cache = kv_layer.flatten(0, 1)
        for request in connector_metadata.requests:
            # Decode steps revisit active requests as cached requests. Chunked
            # prefill is disabled for capture mode, so the new-request pass has
            # the complete prompt and is the only pass that needs to save it.
            if not request.new_req:
                continue
            num_prompt_tokens = int(request.token_ids.shape[0])
            if num_prompt_tokens < 1 or request.slot_mapping.numel() < num_prompt_tokens:
                raise RuntimeError(
                    f"Invalid prompt slot mapping for request {request.req_id}."
                )
            last_slot = int(request.slot_mapping[num_prompt_tokens - 1].item())
            hidden_states = flat_cache[last_slot].detach().cpu()
            token_ids = request.token_ids[-1:].detach().cpu()
            safetensors.torch.save_file(
                {"hidden_states": hidden_states, "token_ids": token_ids},
                request.filename,
            )
