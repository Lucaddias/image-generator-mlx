"""
Motor local de vídeo com LTX-2.3 (MLX 4-bit) no Apple Silicon.
Usa o pipeline destilado do ltx-2-mlx em estágios:
  1. Gemma 3 12B (4-bit) codifica o prompt -> libera memória
  2. DiT 22B destilado (4-bit) gera em meia resolução (8 passos),
     sobe 2x no espaço latente e refina em resolução cheia (3 passos)
  3. VAE de vídeo + vocoder decodificam imagem e áudio sincronizado
Aceita só texto (text-to-video) ou texto + imagem inicial (image-to-video).
"""
from __future__ import annotations

import gc
import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent / "models"

LTX_REPO_ID = "dgrauet/ltx-2.3-mlx-q4"
LTX_REVISION = "f306259444116d71927188af2fbbf7fdc1a8089a"
DEFAULT_LTX_DIR = MODELS_DIR / "ltx-2.3-mlx-q4"

GEMMA_REPO_ID = "mlx-community/gemma-3-12b-it-4bit"
GEMMA_REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
DEFAULT_GEMMA_DIR = MODELS_DIR / "gemma-3-12b-it-4bit"

# Só os arquivos que o pipeline destilado usa (~20.5 GB em vez dos ~60 GB do pacote completo).
LTX_DISTILLED_FILES = [
    "*.json",
    "LICENSE",
    "README.md",
    "connector.safetensors",
    "transformer-distilled-1.1.safetensors",
    "vae_encoder.safetensors",
    "vae_decoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
]

# O LTX-2.3 foi treinado a 24 fps; a contagem de quadros segue a grade 8k+1.
FRAME_RATE = 24


@dataclass
class VideoGenerationResult:
    output_path: Path
    elapsed_seconds: float
    peak_mlx_gb: float
    width: int
    height: int
    num_frames: int
    frame_rate: int
    seed: int
    prompt: str
    image: Optional[str]


def seconds_to_frames(seconds: float, frame_rate: int = FRAME_RATE) -> int:
    """Converte uma duração em segundos para a contagem de quadros válida mais próxima (8k+1)."""
    return max(1, round(seconds * frame_rate / 8)) * 8 + 1


def check_ffmpeg() -> None:
    """Confere se o ffmpeg existe e abre. Sem ele o vídeo é gerado, mas falha só na hora de salvar o MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg não encontrado. Instale com: brew install ffmpeg")
    proc = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "O ffmpeg instalado não abre (tente: brew upgrade ffmpeg):\n" + proc.stderr.strip()[-500:]
        )


def ensure_video_models_downloaded(
    ltx_dir: Path = DEFAULT_LTX_DIR,
    gemma_dir: Path = DEFAULT_GEMMA_DIR,
) -> tuple[Path, Path]:
    """Verifica se o LTX-2.3 q4 e o Gemma 3 12B já estão baixados; se não, faz o download (~28.6 GB)."""
    ltx_dir, gemma_dir = Path(ltx_dir), Path(gemma_dir)

    ltx_required = [ltx_dir / name for name in LTX_DISTILLED_FILES if "*" not in name]
    if not all(f.exists() for f in ltx_required):
        print(f"[LTX-2.3] Baixando pipeline destilado 4-bit ({LTX_REPO_ID}, ~20.5 GB) para {ltx_dir}...")
        ltx_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=LTX_REPO_ID,
            revision=LTX_REVISION,
            local_dir=str(ltx_dir),
            allow_patterns=LTX_DISTILLED_FILES,
        )

    gemma_required = [gemma_dir / "config.json", gemma_dir / "model.safetensors.index.json"]
    if not all(f.exists() for f in gemma_required):
        print(f"[LTX-2.3] Baixando codificador de texto ({GEMMA_REPO_ID}, ~8.1 GB) para {gemma_dir}...")
        gemma_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=GEMMA_REPO_ID,
            revision=GEMMA_REVISION,
            local_dir=str(gemma_dir),
        )

    return ltx_dir, gemma_dir


class LTXVideoEngine:
    """Gerenciador do motor local de vídeo LTX-2.3 (pipeline destilado, MLX 4-bit)."""

    def __init__(
        self,
        ltx_dir: Path = DEFAULT_LTX_DIR,
        gemma_dir: Path = DEFAULT_GEMMA_DIR,
        low_ram: bool = False,
    ):
        self.ltx_dir = Path(ltx_dir)
        self.gemma_dir = Path(gemma_dir)
        # Lê os blocos do transformer direto do disco: menos memória, um pouco mais lento.
        self.low_ram = low_ram

    def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 512,
        num_frames: int = 49,
        seed: Optional[int] = 42,
        image: Optional[Path] = None,
        output_path: Path = Path("outputs/video.mp4"),
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> VideoGenerationResult:
        if any(v < 256 or v > 1280 or v % 64 != 0 for v in (width, height)):
            raise ValueError("Largura e altura devem ser múltiplos de 64 (entre 256 e 1280).")
        if num_frames < 9 or (num_frames - 1) % 8 != 0:
            raise ValueError("O número de quadros deve seguir a grade 8k+1 (9, 17, 25, ..., 49, 97, 121).")
        if image is not None and not Path(image).exists():
            raise FileNotFoundError(f"Imagem inicial não encontrada: {image}")
        check_ffmpeg()

        if progress_cb:
            progress_cb("Verificando arquivos do modelo...")
        ensure_video_models_downloaded(self.ltx_dir, self.gemma_dir)

        import mlx.core as mx
        from ltx_pipelines_mlx import DistilledPipeline

        if not mx.is_available(mx.gpu):
            raise RuntimeError("GPU Apple Silicon (Metal/MLX) não disponível.")

        actual_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**31 - 1)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mx.reset_peak_memory()
        started = time.monotonic()

        if progress_cb:
            progress_cb("Gerando vídeo (Gemma -> DiT 8+3 passos -> VAE + áudio)...")
        pipe = DistilledPipeline(
            model_dir=str(self.ltx_dir),
            gemma_model_id=str(self.gemma_dir),
            low_memory=True,
            low_ram_streaming=self.low_ram,
        )
        pipe.generate_and_save(
            prompt=prompt,
            output_path=str(output_path),
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(FRAME_RATE),
            seed=actual_seed,
            image=str(image) if image is not None else None,
        )

        # Libera o pipeline inteiro da memória unificada
        del pipe
        gc.collect()
        mx.clear_cache()

        elapsed = round(time.monotonic() - started, 2)
        peak_gb = round(mx.get_peak_memory() / (1024**3), 2)

        report = {
            "model": LTX_REPO_ID,
            "pipeline": "distilled",
            "elapsed_seconds": elapsed,
            "peak_mlx_gb": peak_gb,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": FRAME_RATE,
            "duration_seconds": round(num_frames / FRAME_RATE, 2),
            "seed": actual_seed,
            "prompt": prompt,
            "image": str(Path(image).resolve()) if image is not None else None,
            "low_ram": self.low_ram,
            "output_path": str(output_path.resolve()),
        }
        output_path.with_suffix(".run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

        return VideoGenerationResult(
            output_path=output_path.resolve(),
            elapsed_seconds=elapsed,
            peak_mlx_gb=peak_gb,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=FRAME_RATE,
            seed=actual_seed,
            prompt=prompt,
            image=str(image) if image is not None else None,
        )
