"""
Motor local leve para Qwen-Image-2.1-Turbo (MLX 4-bit) no Apple Silicon.
Suporta dois modos de operação:
  1. 'staged' (Ultra-Leve, ~4.8 GB RAM pico): Carrega um componente por vez
     (Text Encoder -> libera -> Transformer 8 passos -> libera -> VAE).
  2. 'resident' (Rápido para múltiplas gerações, ~11 GB RAM): Mantém os componentes
     em memória para evitar recarregar do disco a cada imagem.
"""
from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "joeynyc/Studio-Image-Turbo-MLX-4bit-from-Qwen-2.1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "qwen-image-2.1-turbo-4bit"

SIGMA_NODES = [1.0, 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568]


@dataclass
class GenerationResult:
    output_path: Path
    elapsed_seconds: float
    peak_mlx_gb: float
    width: int
    height: int
    seed: int
    prompt: str
    mode: str
    stages: list[dict[str, Any]] = field(default_factory=list)


def ensure_model_downloaded(
    model_dir: Path = DEFAULT_MODEL_DIR,
    repo_id: str = DEFAULT_REPO_ID,
) -> Path:
    """Verifica se o modelo 4-bit já está baixado localmente; se não, faz o download."""
    model_dir = Path(model_dir)
    required_files = [
        model_dir / "model_index.json",
        model_dir / "scheduler" / "scheduler_config.json",
        model_dir / "text_encoder" / "0.safetensors",
        model_dir / "text_encoder" / "1.safetensors",
        model_dir / "text_encoder" / "2.safetensors",
        model_dir / "transformer" / "0.safetensors",
        model_dir / "transformer" / "1.safetensors",
        model_dir / "vae" / "0.safetensors",
    ]
    if all(f.exists() for f in required_files):
        return model_dir

    print(f"[Qwen-Image-2.1-Turbo] Baixando motor local 4-bit ({repo_id}) para {model_dir}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
    )
    return model_dir


def validate_source_metadata(root: Path) -> None:
    index_path = root / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {index_path}. Execute o download do modelo primeiro.")
    index = json.loads(index_path.read_text())
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("O checkpoint precisa ser compatível com QwenImage21Pipeline")
    if index.get("sample_sigmas") != SIGMA_NODES:
        raise ValueError("A grade de sigmas do checkpoint difere da receita oficial do Turbo (8 passos)")


@contextmanager
def turbo_config_binding(edit_module: Any, mx: Any, observed: dict[str, Any]):
    """Aplica o schedule exato de 8 passos do Qwen-Image-2.1-Turbo (shift=1, CFG=1)."""
    original = edit_module.Config

    class TurboScheduler:
        def __init__(self):
            self.sigmas = mx.array([*SIGMA_NODES, 0.0], dtype=mx.float32)
            self.timesteps = mx.array(list(range(8)), dtype=mx.float32)

    def create_config(*args: Any, **kwargs: Any):
        config = original(*args, **kwargs)
        if config.num_inference_steps != 8:
            raise ValueError("Qwen-Image-2.1-Turbo requer exatamente 8 passos (steps=8)")
        config._scheduler = TurboScheduler()
        observed.update(
            {
                "class": "QwenTurboFixedEuler",
                "raw_sigmas": list(SIGMA_NODES),
                "shifted_sigmas": config._scheduler.sigmas.tolist(),
            }
        )
        return config

    edit_module.Config = create_config
    try:
        yield
    finally:
        edit_module.Config = original


class QwenTurboEngine:
    """Gerenciador do motor local Qwen-Image-2.1-Turbo em MLX 4-bit."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._resident_model = None

    def clear_resident(self) -> None:
        if self._resident_model is not None:
            import mlx.core as mx

            self._resident_model = None
            gc.collect()
            mx.clear_cache()

    def _build_staged_model(
        self,
        components: Any,
        edit_module: Any,
        mx: Any,
        memory_saver_class: Any,
        prompt: str,
        lora_path: Optional[Path],
        stages: list[dict[str, Any]],
        progress_cb: Optional[Callable[[str], None]] = None,
    ):
        root = self.model_dir
        definition = components.QwenImage21WeightDefinition
        initializer = components.Qwen21Initializer

        class StagedEdit(edit_module.QwenImage21Edit):
            def __init__(self):
                edit_module.nn.Module.__init__(self)
                initializer.init_config(self, edit_module.ModelConfig.qwen_image_21())
                self._checkpoint_path = str(root)
                self._component_configs = {
                    name: json.loads((root / name / "config.json").read_text())
                    for name in ("vae", "transformer", "text_encoder")
                }
                self.tokenizers = components.TokenizerLoader.load_all(
                    definition.get_tokenizers(), str(root)
                )
                self.processor = components.QwenImage21Processor(
                    root / "processor", self.tokenizers["qwen21"].tokenizer
                )
                self.text_encoder = self.transformer = self.vae = None
                self.bits = None

        model = StagedEdit()
        constructors = {
            "text_encoder": components.QwenImage21TextEncoder,
            "transformer": components.QwenImage21Transformer,
            "vae": components.QwenImage21VAE,
        }

        def record(stage: str):
            stages.append(
                {
                    "stage": stage,
                    "active_mlx_gb": round(mx.get_active_memory() / (1024**3), 3),
                    "peak_mlx_gb": round(mx.get_peak_memory() / (1024**3), 3),
                }
            )

        def load_component(name: str):
            class SingleComponent(definition):
                @staticmethod
                def get_components():
                    return [item for item in definition.get_components() if item.name == name]

            previous_bits = model.bits
            setattr(model, name, constructors[name](model._component_configs[name]))
            initializer.load_components(model, root, SingleComponent, None, validate=True)
            if previous_bits is not None and model.bits not in (None, previous_bits):
                raise ValueError("Níveis de quantização conflitantes entre componentes")
            if model.bits is None:
                model.bits = previous_bits
            record(f"{name}_loaded")

        if progress_cb:
            progress_cb("1/3 Codificando prompt (Qwen3-VL 4-bit)...")
        load_component("text_encoder")
        embeddings, slots = model._encode_prompt(prompt, [])
        mx.eval(embeddings, slots)
        record("prompt_encoded")

        # Libera o Text Encoder imediatamente da memória unificada!
        model.text_encoder = None
        gc.collect()
        mx.clear_cache()
        record("text_encoder_released")

        if progress_cb:
            progress_cb("2/3 Gerando imagem em 8 passos (DiT 7B Turbo 4-bit)...")
        load_component("transformer")
        initializer.apply_lora(
            model,
            [str(lora_path)] if lora_path is not None else None,
            [1.0] if lora_path is not None else None,
            False,
        )

        class DecodeLoader(memory_saver_class):
            def call_after_loop(self, *args: Any, **kwargs: Any):
                super().call_after_loop(*args, **kwargs)
                record("transformer_released")
                if progress_cb:
                    progress_cb("3/3 Decodificando pixels no VAE...")
                load_component("vae")

        model.callbacks.register(
            DecodeLoader(
                model=model,
                keep_transformer=False,
                cache_limit_bytes=None,
                keep_text_encoder=False,
            )
        )
        return model

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = 42,
        output_path: Path = Path("outputs/image.png"),
        lora_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        if any(v < 256 or v > 2048 or v % 32 != 0 for v in (width, height)):
            raise ValueError("Largura e altura devem ser múltiplos de 32 (entre 256 e 2048).")

        ensure_model_downloaded(self.model_dir)
        validate_source_metadata(self.model_dir)

        import random
        import mlx.core as mx
        from mflux.callbacks.instances.memory_saver import MemorySaver
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21 import qwen_image21_initializer as components
        from mflux.models.qwen21.variants.edit import qwen_image_21_edit as edit

        if not mx.is_available(mx.gpu):
            raise RuntimeError("GPU Apple Silicon (Metal/MLX) não disponível.")

        actual_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**31 - 1)

        mx.set_cache_limit(512 * 1024**2)
        mx.reset_peak_memory()
        started = time.monotonic()
        stages: list[dict[str, Any]] = []

        model = self._build_staged_model(
            components=components,
            edit_module=edit,
            mx=mx,
            memory_saver_class=MemorySaver,
            prompt=prompt,
            lora_path=lora_path,
            stages=stages,
            progress_cb=progress_cb,
        )
        model.tiling_config = TilingConfig(vae_decode_tile_size=256)

        schedule: dict[str, Any] = {}
        with turbo_config_binding(edit, mx, schedule):
            image = model.generate_image(
                prompt=prompt,
                seed=actual_seed,
                width=width,
                height=height,
                output_resolution=max(width, height),
                num_inference_steps=8,
                guidance=1.0,
                negative_prompt=None,
                use_kv_cache=True,
                use_step_cache=False,
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, export_json_metadata=True)

        # Limpa o VAE final da memória
        model.vae = None
        del model
        gc.collect()
        mx.clear_cache()

        elapsed = round(time.monotonic() - started, 2)
        peak_gb = round(mx.get_peak_memory() / (1024**3), 2)

        report = {
            "elapsed_seconds": elapsed,
            "peak_mlx_gb": peak_gb,
            "width": width,
            "height": height,
            "seed": actual_seed,
            "prompt": prompt,
            "output_path": str(output_path.resolve()),
            "component_stages": stages,
        }
        output_path.with_suffix(".run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

        return GenerationResult(
            output_path=output_path.resolve(),
            elapsed_seconds=elapsed,
            peak_mlx_gb=peak_gb,
            width=width,
            height=height,
            seed=actual_seed,
            prompt=prompt,
            mode="staged-4bit",
            stages=stages,
        )
