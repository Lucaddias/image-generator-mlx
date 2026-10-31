#!/usr/bin/env python3
"""
CLI local para geração de imagens com Qwen-Image-2.1-Turbo (MLX 4-bit).
Ideal para uso direto no terminal ou integração com agentes CLI (Claude Code, OpenCode, Codex).

Exemplos:
  # Baixar apenas os pesos do modelo (~9.9 GB):
  .venv/bin/python generate.py --download-only

  # Gerar uma imagem:
  .venv/bin/python generate.py --prompt "Um gato astronauta em Tóquio cyberpunk, placa neon escrita 'OLÁ MUNDO'" --width 512 --height 512
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from engine import DEFAULT_MODEL_DIR, QwenTurboEngine, ensure_model_downloaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gerador local Qwen-Image-2.1-Turbo (7B DiT, MLX 4-bit para Apple Silicon)"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Descrição textual da imagem a ser gerada.",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=512,
        help="Largura da imagem (múltiplo de 32, padrão: 512).",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=512,
        help="Altura da imagem (múltiplo de 32, padrão: 512).",
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
        help="Caminho do arquivo PNG de saída (padrão: outputs/qwen_<timestamp>.png).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Pasta onde o modelo 4-bit está/será armazenado.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Apenas baixa/verifica os arquivos do modelo (~9.9 GB) e encerra.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime apenas o JSON final no stdout (ótimo para integração com Claude Code / OpenCode).",
    )

    args = parser.parse_args()

    if args.download_only:
        path = ensure_model_downloaded(args.model_path)
        print(f"Modelo pronto em: {path}")
        return 0

    if not args.prompt:
        parser.error("O argumento --prompt (-p) é obrigatório (a menos que use --download-only).")

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path("outputs") / f"qwen_{ts}.png"

    def log_progress(msg: str):
        if not args.json:
            print(f"[Qwen-Turbo] {msg}", file=sys.stderr)

    engine = QwenTurboEngine(model_dir=args.model_path)
    result = engine.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
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
        "seed": result.seed,
        "prompt": result.prompt,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print("\n✅ Imagem gerada com sucesso!")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
