"""Runtime compatibility patch for final-state extraction in vLLM 0.17."""


def install_llama_final_hidden_state_capture() -> None:
    """Use a compile-compatible Llama subclass that exposes final RMSNorm.

    vLLM auxiliary layer ids normally identify residual streams immediately
    before decoder blocks.  The DTGB capture path uses ``num_hidden_layers`` as
    a sentinel for the final normalized state consumed by the LM head.
    """

    from itertools import islice

    import vllm

    if not str(vllm.__version__).startswith("0.17."):
        raise RuntimeError(
            "DTGB final-RMSNorm capture is validated against vLLM 0.17.x; "
            f"detected {vllm.__version__}. Refusing to risk exporting a mislabeled state."
        )

    from vllm.compilation.decorators import support_torch_compile
    from vllm.distributed import get_pp_group
    from vllm.model_executor.models import llama as llama_module
    from vllm.sequence import IntermediateTensors

    causal_lm_class = llama_module.LlamaForCausalLM
    current_init_model = causal_lm_class._init_model
    if getattr(current_init_model, "_dtgb_final_hidden_state_capture", False):
        return

    base_model_class = llama_module.LlamaModel

    @support_torch_compile(
        dynamic_arg_dims={
            "input_ids": 0,
            "positions": 0,
            "intermediate_tensors": 0,
            "inputs_embeds": 0,
        },
        shape_invariants=llama_module.llama_model_invariants,
    )
    class DTGBFinalHiddenLlamaModel(base_model_class):
        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=None,
            **extra_layer_kwargs,
        ):
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.embed_input_ids(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            auxiliary_states = []
            for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)
            ):
                if idx in self.aux_hidden_state_layers:
                    auxiliary_states.append(hidden_states + residual)
                hidden_states, residual = layer(
                    positions, hidden_states, residual, **extra_layer_kwargs
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "residual": residual}
                )

            hidden_states, _ = self.norm(hidden_states, residual)
            final_state_id = int(self.config.num_hidden_layers)
            if final_state_id in self.aux_hidden_state_layers:
                auxiliary_states.append(hidden_states)

            if auxiliary_states:
                return hidden_states, auxiliary_states
            return hidden_states

    def init_model_with_final_hidden_capture(
        self,
        vllm_config,
        prefix="",
        layer_type=llama_module.LlamaDecoderLayer,
    ):
        return DTGBFinalHiddenLlamaModel(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    init_model_with_final_hidden_capture._dtgb_final_hidden_state_capture = True
    init_model_with_final_hidden_capture._dtgb_original_init_model = current_init_model
    causal_lm_class._init_model = init_model_with_final_hidden_capture
