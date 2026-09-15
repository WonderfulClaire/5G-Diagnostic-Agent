"""Local Transformers backend for real tool-use rollouts on a single GPU."""


class LocalModelBackend:
    def __init__(self, model_path, device="cuda:0", max_new_tokens=512, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages, schemas):
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tools=schemas, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] + self.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Trajectory exceeds the model context limit")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
