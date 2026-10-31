#!/usr/bin/env python3
"""
Interface Web Local (Gradio) para o Qwen-Image-2.1-Turbo (MLX 4-bit).
"""
from datetime import datetime
from pathlib import Path

import gradio as gr

from engine import DEFAULT_MODEL_DIR, QwenTurboEngine

engine = QwenTurboEngine(model_dir=DEFAULT_MODEL_DIR)

ASPECT_PRESETS = {
    "Quadrado Rápido (512×512)": (512, 512),
    "Quadrado HD (768×768)": (768, 768),
    "Quadrado Full (1024×1024)": (1024, 1024),
    "Paisagem 16:9 (896×512)": (896, 512),
    "Paisagem HD (1024×576)": (1024, 576),
    "Retrato 9:16 (512×896)": (512, 896),
    "Retrato HD (576×1024)": (576, 1024),
}


def apply_preset(preset_name: str):
    w, h = ASPECT_PRESETS.get(preset_name, (512, 512))
    return w, h


def list_gallery_images() -> list[str]:
    out_dir = Path("outputs")
    if not out_dir.exists():
        return []
    pngs = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in pngs[:24]]


def generate_ui(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    progress=gr.Progress(track_tqdm=False),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Digite um prompt para gerar a imagem.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / f"qwen_{ts}.png"

    def on_step(msg: str):
        progress(None, desc=msg)

    result = engine.generate(
        prompt=prompt.strip(),
        width=int(width),
        height=int(height),
        seed=int(seed),
        output_path=output_path,
        progress_cb=on_step,
    )

    stats_md = (
        f"### ⚡ Métricas da Execução (Apple Silicon M5 — MLX 4-bit)\n"
        f"- **Tempo total:** `{result.elapsed_seconds}s` (8 passos Turbo)\n"
        f"- **Pico de Memória GPU (MLX):** `{result.peak_mlx_gb} GB` *(execução em estágios)*\n"
        f"- **Resolução:** `{result.width}×{result.height}` | **Seed:** `{result.seed}`\n"
        f"- **Arquivo salvo:** `{result.output_path}`"
    )
    return str(result.output_path), stats_md, list_gallery_images()


with gr.Blocks(title="Qwen-Image-2.1-Turbo Local Studio (MLX 4-bit)") as demo:
    gr.Markdown(
        """
        # 🎨 Qwen-Image-2.1-Turbo (7B) — Motor Local Leve (MLX 4-bit)
        Rodando 100% local no seu **Mac Apple Silicon** com carregamento em estágios (~4.7 GB de pico de VRAM).
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Prompt (Inglês ou Português — suporte nativo a textos na imagem entre aspas)",
                placeholder='Ex: A cinematic neon coffee shop sign that reads "CAFÉ DO LUCA", rainy night, reflections on wet cobblestones, 8k',
                lines=4,
            )

            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    choices=list(ASPECT_PRESETS.keys()),
                    value="Quadrado Rápido (512×512)",
                    label="Proporção / Resolução",
                )
                seed_input = gr.Number(
                    label="Seed (-1 = Aleatório)",
                    value=-1,
                    precision=0,
                )

            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    step=32,
                    value=512,
                    label="Largura (px)",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    step=32,
                    value=512,
                    label="Altura (px)",
                )

            generate_btn = gr.Button("🚀 Gerar Imagem (8 passos Turbo)", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    [
                        'A small red robot holding a white cardboard sign that says "HELLO MAC M5", clean studio photograph, soft rim light',
                        "Quadrado Rápido (512×512)",
                        42,
                    ],
                    [
                        'A cyberpunk ramen stall in Tokyo at night with a glowing neon sign reading "QWEN 2.1 TURBO", steam rising, wet street reflections',
                        "Paisagem 16:9 (896×512)",
                        100,
                    ],
                    [
                        'Minimalist vintage travel poster of Rio de Janeiro at sunset, bold typography at the top reading "BRASIL", warm retro colors',
                        "Retrato 9:16 (512×896)",
                        7,
                    ],
                ],
                inputs=[prompt_input, preset_dropdown, seed_input],
                label="Exemplos rápidos",
            )

        with gr.Column(scale=5):
            output_image = gr.Image(label="Imagem Gerada", type="filepath")
            stats_output = gr.Markdown("Aguardando geração...")

    with gr.Accordion("🖼️ Galeria de Imagens Geradas Localmente (pasta outputs/)", open=False):
        gallery = gr.Gallery(value=list_gallery_images, columns=4, height="auto")

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[width_slider, height_slider],
    )

    generate_btn.click(
        fn=generate_ui,
        inputs=[prompt_input, width_slider, height_slider, seed_input],
        outputs=[output_image, stats_output, gallery],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
