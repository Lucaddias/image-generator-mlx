#!/usr/bin/env python3
"""
CLI local para geração de vídeo com LTX-2.3 (MLX 4-bit), com áudio sincronizado.
Ideal para uso direto no terminal ou integração com agentes CLI (Claude Code, OpenCode, Codex).

Exemplos:
  # Baixar apenas os pesos do modelo (~28.6 GB):
  .venv/bin/python generate_video.py --download-only

  # Texto para vídeo (5 segundos, ~3 min num M5 24 GB):
  .venv/bin/python generate_video.py --prompt "Ondas quebrando numa praia ao pôr do sol, câmera lenta" --seconds 5

  # Animar uma imagem gerada pelo Qwen:
  .venv/bin/python generate_video.py --prompt "A câmera se aproxima devagar" --image outputs/qwen_20261027_160536.png
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from video_engine import (
    DEFAULT_GEMMA_DIR,
    DEFAULT_LTX_DIR,
    FRAME_RATE,
    LTXVideoEngine,
    ensure_video_models_downloaded,
    seconds_to_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gerador local de vídeo LTX-2.3 (22B destilado, MLX 4-bit para Apple Silicon)"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Descrição do vídeo (cena, movimento de câmera, sons).",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        default=None,
        help="Imagem inicial opcional (image-to-video). Ex: um PNG da pasta outputs/.",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=768,
        help="Largura do vídeo (múltiplo de 64, padrão: 768).",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=512,
        help="Altura do vídeo (múltiplo de 64, padrão: 512).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help=f"Duração aproximada em segundos a {FRAME_RATE} fps (padrão: 5).",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=None,
        help="Número exato de quadros (8k+1: 49, 97, 121...). Tem prioridade sobre --seconds.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=-1,
        help="Seed para reprodutibilidade (-1 para aleatório, padrão: -1).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Caminho do arquivo MP4 de saída (padrão: outputs/ltx_<timestamp>.mp4).",
    )
    parser.add_argument(
        "--low-ram",
        action="store_true",
        help="Lê o transformer do disco aos poucos: usa menos memória, fica um pouco mais lento.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_LTX_DIR,
        help="Pasta onde o LTX-2.3 q4 está/será armazenado.",
    )
    parser.add_argument(
        "--gemma-path",
        type=Path,
        default=DEFAULT_GEMMA_DIR,
        help="Pasta onde o Gemma 3 12B (codificador de texto) está/será armazenado.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Apenas baixa/verifica os arquivos do modelo (~28.6 GB) e encerra.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime apenas o JSON final no stdout (ótimo para integração com Claude Code / OpenCode).",
    )

    args = parser.parse_args()

    if args.download_only:
        ltx_dir, gemma_dir = ensure_video_models_downloaded(args.model_path, args.gemma_path)
        print(f"Modelo pronto em: {ltx_dir}\nCodificador de texto pronto em: {gemma_dir}")
        return 0

    if not args.prompt:
        parser.error("O argumento --prompt (-p) é obrigatório (a menos que use --download-only).")

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path("outputs") / f"ltx_{ts}.mp4"

    num_frames = args.frames if args.frames is not None else seconds_to_frames(args.seconds)

    def log_progress(msg: str):
        if not args.json:
            print(f"[LTX-2.3] {msg}", file=sys.stderr)

    engine = LTXVideoEngine(ltx_dir=args.model_path, gemma_dir=args.gemma_path, low_ram=args.low_ram)
    result = engine.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=num_frames,
        seed=args.seed,
        image=args.image,
        output_path=args.output,
        progress_cb=log_progress,
    )

    summary = {
        "status": "success",
        "output_path": str(result.output_path),
        "elapsed_seconds": result.elapsed_seconds,
        "peak_mlx_gb": result.peak_mlx_gb,
        "width": result.width,
        "height": result.height,
        "num_frames": result.num_frames,
        "duration_seconds": round(result.num_frames / result.frame_rate, 2),
        "seed": result.seed,
        "prompt": result.prompt,
        "image": result.image,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print("\n✅ Vídeo gerado com sucesso!")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
