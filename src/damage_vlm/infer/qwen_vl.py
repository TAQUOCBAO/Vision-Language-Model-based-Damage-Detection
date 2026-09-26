"""Qwen3-VL LoRA / merged-model inference helpers for holdout / submission."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Literal

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from damage_vlm.config import load_prompts_config
from damage_vlm.data.build_sft import build_official_user_prompt, build_user_prompt


DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QuantMode = Literal["none", "4bit", "8bit"]


def _user_text_for_chat(image_id: str, *, official: bool = False) -> str:
    """Build user text without ``<image>`` (image is a separate content part)."""
    text = build_official_user_prompt(image_id) if official else build_user_prompt(image_id)
    return text.replace("<image>", "").lstrip()


def build_messages(
    image_path: Path | str,
    image_id: str,
    *,
    official: bool = False,
) -> list[dict[str, Any]]:
    system = str(load_prompts_config()["system"]).strip()
    path = Path(image_path)
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(path.resolve())},
                {"type": "text", "text": _user_text_for_chat(image_id, official=official)},
            ],
        },
    ]


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bitsandbytes_config(quant: QuantMode):
    if quant == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required for BitsAndBytesConfig") from exc
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True)


class Qwen3VLLoRAGenerator:
    """Load Qwen3-VL (optional LoRA / merged) and generate JSON strings per image."""

    def __init__(
        self,
        adapter_dir: Path | str | None = None,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        model_dir: Path | str | None = None,
        device: str | None = None,
        max_new_tokens: int = 384,
        image_max_pixels: int = 262144,
        quant: QuantMode = "none",
        gpu_hold: list | None = None,
    ) -> None:
        """Load weights for inference.

        Args:
            adapter_dir: LoRA adapter directory (``adapter_config.json`` present).
            model_dir: Merged full model directory (no LoRA). Takes precedence over
                ``base_model`` when set.
            quant: ``none`` (bf16), ``4bit``, or ``8bit`` (bitsandbytes). 4-bit cuts
                VRAM roughly 3–4× vs bf16 for the 8B backbone.
        """
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.model_dir = Path(model_dir) if model_dir else None
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.image_max_pixels = image_max_pixels
        self.quant = quant
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available to PyTorch. Official / full-precision "
                    "inference requires a GPU. Check nvidia-smi, then either reboot "
                    "or reload nvidia_uvm (sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm). "
                    "Pass device='cpu' only for an explicit CPU fallback."
                )
            device = "cuda:0"
        self.device = device
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {self.device} but torch.cuda.is_available() is False. "
                "The NVIDIA driver may be stuck (cuInit 999). Reload nvidia_uvm or reboot."
            )

        pretrained = str(self.model_dir) if self.model_dir is not None else base_model
        processor_src = pretrained if self.model_dir is not None else base_model

        self.processor = AutoProcessor.from_pretrained(
            processor_src,
            trust_remote_code=True,
        )
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = image_max_pixels

        quant_config = _bitsandbytes_config(quant)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if quant_config is not None:
            if not str(device).startswith("cuda"):
                raise ValueError("bitsandbytes 4/8-bit load requires CUDA")
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = {"": 0}
        else:
            # No device_map: transformers' caching_allocator_warmup would
            # pre-allocate ~16GB empty tensors and OOM if another job is on GPU.
            load_kwargs["dtype"] = (
                torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            )

        model = AutoModelForImageTextToText.from_pretrained(pretrained, **load_kwargs)

        if self.adapter_dir is not None and (self.adapter_dir / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(self.adapter_dir))
        elif self.model_dir is None and self.adapter_dir is not None:
            raise FileNotFoundError(
                f"No adapter_config.json under {self.adapter_dir}"
            )

        if str(self.device).startswith("cuda") and quant == "none":
            # Drop the VRAM pin without empty_cache() so this process keeps the
            # reserved pool; TGI cannot steal it while we place the model.
            if gpu_hold:
                gpu_hold.clear()
                gc.collect()
            model = model.to(device=self.device, dtype=torch.bfloat16)

        model.eval()
        self.model = model
        param_dev = next(self.model.parameters()).device
        if str(self.device).startswith("cuda") and param_dev.type != "cuda":
            raise RuntimeError(
                f"Model loaded on {param_dev}, expected {self.device}. "
                "Refuse to run official / full inference on CPU."
            )

    @torch.inference_mode()
    def generate(
        self,
        image_path: Path | str,
        image_id: str | None = None,
        *,
        official: bool = False,
    ) -> str:
        path = Path(image_path)
        image_id = image_id or path.stem
        messages = build_messages(path, image_id, official=official)

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Install vision helpers: pip install "qwen-vl-utils"'
            ) from exc

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        device = _model_device(self.model)
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, prompt_len:]
        out = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, generated, new_tokens, image_inputs, video_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out.strip()

    def __call__(self, image_path: Path) -> str:
        return self.generate(image_path, image_path.stem)
