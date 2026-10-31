#!/usr/bin/env python3
"""
Interface Web Local (Gradio) para o Qwen-Image-2.1-Turbo (imagem) e o LTX-2.3 (vídeo), ambos em MLX 4-bit.
"""
from datetime import datetime
from pathlib import Path

import gradio as gr

from engine import DEFAULT_MODEL_DIR, QwenTurboEngine
from video_engine import FRAME_RATE, LTXVideoEngine, seconds_to_frames

engine = QwenTurboEngine(model_dir=DEFAULT_MODEL_DIR)

# Imagem e vídeo dividem a mesma GPU/memória unificada: uma geração por vez.
GPU_QUEUE = "mlx_gpu"

ASPECT_PRESETS = {
    "Quadrado Rápido (512×512)": (512, 512),
    "Quadrado HD (768×768)": (768, 768),
    "Quadrado Full (1024×1024)": (1024, 1024),
    "Paisagem 16:9 (896×512)": (896, 512),
    "Paisagem HD (1024×576)": (1024, 576),
    "Retrato 9:16 (512×896)": (512, 896),
    "Retrato HD (576×1024)": (576, 1024),
}

VIDEO_PRESETS = {
    "Rascunho Rápido (512×320)": (512, 320),
    "Paisagem 3:2 (768×512)": (768, 512),
    "Paisagem HD 16:9 (1024×576)": (1024, 576),
    "Quadrado (512×512)": (512, 512),
    "Retrato 2:3 (512×768)": (512, 768),
    "Retrato HD 9:16 (576×1024)": (576, 1024),
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


def generate_video_ui(
    prompt: str,
    image_path: str | None,
    preset_name: str,
    seconds: float,
    seed: int,
    low_ram: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Digite um prompt para gerar o vídeo.")

    width, height = VIDEO_PRESETS.get(preset_name, (768, 512))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / f"ltx_{ts}.mp4"

    def on_step(msg: str):
        progress(None, desc=msg)

    video_engine = LTXVideoEngine(low_ram=bool(low_ram))
    result = video_engine.generate(
        prompt=prompt.strip(),
        width=width,
        height=height,
        num_frames=seconds_to_frames(float(seconds)),
        seed=int(seed),
        image=Path(image_path) if image_path else None,
        output_path=output_path,
        progress_cb=on_step,
    )

    minutes, secs = divmod(int(result.elapsed_seconds), 60)
    stats_md = (
        f"### 🎬 Métricas da Execução (LTX-2.3 destilado — MLX 4-bit)\n"
        f"- **Tempo total:** `{minutes}min {secs}s`\n"
        f"- **Pico de Memória GPU (MLX):** `{result.peak_mlx_gb} GB`\n"
        f"- **Resolução:** `{result.width}×{result.height}` | "
        f"**Duração:** `{result.num_frames / result.frame_rate:.1f}s` ({result.num_frames} quadros a {result.frame_rate} fps) | "
        f"**Seed:** `{result.seed}`\n"
        f"- **Arquivo salvo:** `{result.output_path}`"
    )
    return str(result.output_path), stats_md


def send_to_video(image_path: str | None):
    if not image_path:
        raise gr.Error("Gere uma imagem primeiro.")
    return image_path, gr.Tabs(selected="video")


with gr.Blocks(title="Image Generator MLX — Imagem e Vídeo Local") as demo:
    gr.Markdown(
        """
        # 🎨 Estúdio Local MLX — Imagem (Qwen-Image-2.1-Turbo) e Vídeo (LTX-2.3)
        Rodando 100% local no seu **Mac Apple Silicon** com carregamento em estágios.
        """
    )

    with gr.Tabs() as tabs:
        with gr.Tab("🖼️ Imagem", id="image"):
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
                    animate_btn = gr.Button("🎬 Animar esta imagem (vai para a aba Vídeo)")
                    stats_output = gr.Markdown("Aguardando geração...")

            with gr.Accordion("🖼️ Galeria de Imagens Geradas Localmente (pasta outputs/)", open=False):
                gallery = gr.Gallery(value=list_gallery_images, columns=4, height="auto")

        with gr.Tab("🎬 Vídeo", id="video"):
            with gr.Row():
                with gr.Column(scale=5):
                    video_prompt_input = gr.Textbox(
                        label="Prompt (descreva a cena, o movimento da câmera e os sons — o vídeo sai com áudio)",
                        placeholder="Ex: Waves crashing on a rocky beach at sunset, slow dolly forward, seagulls calling, warm golden light",
                        lines=4,
                    )
                    video_image_input = gr.Image(
                        label="Imagem inicial (opcional — deixe vazio para gerar só a partir do texto)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )

                    with gr.Row():
                        video_preset_dropdown = gr.Dropdown(
                            choices=list(VIDEO_PRESETS.keys()),
                            value="Paisagem 3:2 (768×512)",
                            label="Proporção / Resolução",
                        )
                        video_seed_input = gr.Number(
                            label="Seed (-1 = Aleatório)",
                            value=-1,
                            precision=0,
                        )

                    with gr.Row():
                        video_seconds_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=5,
                            label=f"Duração (segundos, {FRAME_RATE} fps)",
                        )
                        video_low_ram = gr.Checkbox(
                            label="Modo pouca memória (--low-ram)",
                            info="Lê o modelo do disco aos poucos. Use se o Mac travar ou usar muito swap.",
                            value=False,
                        )

                    video_generate_btn = gr.Button("🎬 Gerar Vídeo (LTX-2.3 destilado)", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            [
                                "Waves crashing on a rocky beach at sunset, slow dolly forward, seagulls calling, warm golden light, cinematic",
                                "Paisagem 3:2 (768×512)",
                                42,
                            ],
                            [
                                "A cozy coffee shop on a rainy night, steam rising from a cup on the counter, rain tapping on the window, soft jazz playing",
                                "Paisagem 3:2 (768×512)",
                                7,
                            ],
                        ],
                        inputs=[video_prompt_input, video_preset_dropdown, video_seed_input],
                        label="Exemplos rápidos",
                    )

                with gr.Column(scale=5):
                    output_video = gr.Video(label="Vídeo Gerado")
                    video_stats_output = gr.Markdown(
                        "Aguardando geração... A primeira vez baixa ~28.6 GB de modelo. "
                        "Num M5 24 GB, 5 s em 768×512 levam ~3 min e em 1024×576 ~5 min."
                    )

    preset_dropdown.change(
        fn=apply_preset,
        inputs=[preset_dropdown],
        outputs=[width_slider, height_slider],
    )

    generate_btn.click(
        fn=generate_ui,
        inputs=[prompt_input, width_slider, height_slider, seed_input],
        outputs=[output_image, stats_output, gallery],
        concurrency_id=GPU_QUEUE,
        concurrency_limit=1,
    )

    animate_btn.click(
        fn=send_to_video,
        inputs=[output_image],
        outputs=[video_image_input, tabs],
    )

    video_generate_btn.click(
        fn=generate_video_ui,
        inputs=[
            video_prompt_input,
            video_image_input,
            video_preset_dropdown,
            video_seconds_slider,
            video_seed_input,
            video_low_ram,
        ],
        outputs=[output_video, video_stats_output],
        concurrency_id=GPU_QUEUE,
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
