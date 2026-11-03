"""
Backends de geração do pipeline automático.

  - LocalBackend: os motores deste repositório (Qwen-Image-2.1-Turbo + LTX-2.3, MLX 4-bit no Mac).
  - ComfyUIBackend: qualquer workflow do ComfyUI (local ou numa GPU alugada), via API HTTP.
    Você exporta o workflow em "API format" e diz em quais nós entram prompt, seed, imagem etc.

Os dois só recebem texto e, no vídeo, a imagem que o próprio pipeline gerou.
"""
from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

Progress = Optional[Callable[[str], None]]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}


class BackendError(RuntimeError):
    pass


def frames_for(seconds: float, fps: int, multiple: int) -> int:
    """Quadros na grade multiple*k+1 que o modelo exige (LTX: 8k+1, Wan: 4k+1)."""
    return max(1, round(float(seconds) * fps / multiple)) * multiple + 1


class LocalBackend:
    """Qwen-Image-2.1-Turbo (imagem) + LTX-2.3 (vídeo) rodando no próprio Mac."""

    name = "local"

    def __init__(self, config: dict[str, Any], image_cfg: dict[str, Any]):
        self.low_ram = bool(config.get("low_ram", False))
        lora = image_cfg.get("lora")
        self.lora_path = Path(lora) if lora else None

    def generate_image(self, prompt: str, width: int, height: int, seed: int, output_path: Path,
                       progress_cb: Progress = None) -> Path:
        from engine import QwenTurboEngine

        result = QwenTurboEngine().generate(
            prompt=prompt,
            width=int(width),
            height=int(height),
            seed=int(seed),
            output_path=output_path.with_suffix(".png"),
            lora_path=self.lora_path,
            progress_cb=progress_cb,
        )
        return Path(result.output_path)

    def generate_video(self, prompt: str, image_path: Path, width: int, height: int, seconds: float,
                       seed: int, output_path: Path, progress_cb: Progress = None) -> Path:
        from video_engine import FRAME_RATE, LTXVideoEngine

        result = LTXVideoEngine(low_ram=self.low_ram).generate(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_frames=frames_for(seconds, FRAME_RATE, 8),
            seed=int(seed),
            image=image_path,
            output_path=output_path.with_suffix(".mp4"),
            progress_cb=progress_cb,
        )
        return Path(result.output_path)


# --------------------------------------------------------------------------- ComfyUI


def inject_inputs(workflow: dict[str, Any], mapping: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Copia o workflow (API format) e grava cada valor no nó/entrada indicado em `mapping`.

    mapping: {"prompt": ["6", "text"], "seed": ["3", "seed"], ...}. Chaves sem mapeamento são ignoradas,
    exceto "prompt", que é obrigatória.
    """
    if "prompt" not in mapping:
        raise BackendError("O mapeamento do workflow precisa de 'prompt': [id_do_nó, nome_da_entrada].")
    wf = copy.deepcopy(workflow)
    for key, value in values.items():
        target = mapping.get(key)
        if target is None:
            continue
        if not (isinstance(target, (list, tuple)) and len(target) == 2):
            raise BackendError(f"Mapeamento '{key}' precisa ser [id_do_nó, nome_da_entrada].")
        node_id, input_name = str(target[0]), str(target[1])
        node = wf.get(node_id)
        if not isinstance(node, dict) or "inputs" not in node:
            raise BackendError(
                f"Nó '{node_id}' (para '{key}') não existe no workflow. Exporte com 'Save (API Format)'."
            )
        node["inputs"][input_name] = value
    return wf


def pick_output_file(history_entry: dict[str, Any], exts: set[str]) -> dict[str, Any]:
    """Escolhe o arquivo final na resposta de /history (prefere 'output' a 'temp' das prévias)."""
    candidates = []
    for node_output in (history_entry.get("outputs") or {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and Path(str(item.get("filename", ""))).suffix.lower() in exts:
                    candidates.append(item)
    if not candidates:
        raise BackendError(f"O workflow terminou sem gerar arquivo {sorted(exts)}. Ele tem um nó de salvar?")
    saved = [c for c in candidates if c.get("type") == "output"]
    return (saved or candidates)[-1]


class ComfyUIBackend:
    """Executa workflows exportados do ComfyUI (ex.: Wan 2.2 texto->imagem e imagem->vídeo)."""

    name = "comfyui"

    def __init__(self, config: dict[str, Any], base_dir: Path):
        import requests

        self._requests = requests
        self.url = str(config.get("url") or "http://127.0.0.1:8188").rstrip("/")
        self.timeout = float(config.get("timeout_seconds", 1800))
        self.poll = float(config.get("poll_seconds", 2))
        self.client_id = uuid.uuid4().hex
        self.image_cfg = self._load_stage(config, "image", base_dir)
        self.video_cfg = self._load_stage(config, "video", base_dir)
        self.fps = int(self.video_cfg.get("fps", 16))
        self.frame_multiple = int(self.video_cfg.get("frame_multiple", 4))

    @staticmethod
    def _load_stage(config: dict[str, Any], stage: str, base_dir: Path) -> dict[str, Any]:
        cfg = config.get(stage) or {}
        workflow_path = cfg.get("workflow")
        if not workflow_path:
            raise BackendError(f"backend.{stage}.workflow não configurado (arquivo .json em API format).")
        path = Path(workflow_path)
        if not path.is_absolute():
            path = base_dir / path
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if "nodes" in workflow and "links" in workflow:
            raise BackendError(f"{path} está no formato da interface. No ComfyUI use 'Export (API)'.")
        return {**cfg, "workflow_data": workflow, "inputs": cfg.get("inputs") or {}}

    # -- HTTP

    def _check(self, response, action: str):
        if response.status_code >= 400:
            raise BackendError(f"ComfyUI recusou {action} ({response.status_code}): {response.text[:800]}")
        return response

    def _upload_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as fh:
            response = self._requests.post(
                f"{self.url}/upload/image",
                files={"image": (image_path.name, fh, "image/png")},
                data={"overwrite": "true", "type": "input"},
                timeout=120,
            )
        info = self._check(response, "o upload da imagem").json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    def _run(self, workflow: dict[str, Any], exts: set[str], output_path: Path, progress_cb: Progress) -> Path:
        response = self._requests.post(
            f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=60
        )
        prompt_id = self._check(response, "o workflow").json()["prompt_id"]
        if progress_cb:
            progress_cb(f"ComfyUI: workflow na fila ({prompt_id[:8]})...")

        deadline = time.monotonic() + self.timeout
        while True:
            history = self._check(
                self._requests.get(f"{self.url}/history/{prompt_id}", timeout=30), "a consulta"
            ).json()
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise BackendError(f"ComfyUI falhou: {json.dumps(status.get('messages'))[:800]}")
                if status.get("completed", True):
                    break
            if time.monotonic() > deadline:
                raise BackendError(f"ComfyUI não terminou em {self.timeout:.0f}s (prompt {prompt_id}).")
            time.sleep(self.poll)

        item = pick_output_file(entry, exts)
        file_response = self._check(
            self._requests.get(
                f"{self.url}/view",
                params={"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output")},
                timeout=300,
            ),
            "o download do resultado",
        )
        final_path = output_path.with_suffix(Path(item["filename"]).suffix.lower())
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(file_response.content)
        return final_path

    # -- interface do pipeline

    def generate_image(self, prompt: str, width: int, height: int, seed: int, output_path: Path,
                       progress_cb: Progress = None) -> Path:
        workflow = inject_inputs(
            self.image_cfg["workflow_data"],
            self.image_cfg["inputs"],
            {"prompt": prompt, "seed": int(seed), "width": int(width), "height": int(height)},
        )
        return self._run(workflow, IMAGE_EXTS, output_path, progress_cb)

    def generate_video(self, prompt: str, image_path: Path, width: int, height: int, seconds: float,
                       seed: int, output_path: Path, progress_cb: Progress = None) -> Path:
        if "image" not in self.video_cfg["inputs"]:
            raise BackendError("backend.video.inputs precisa de 'image': [id_do_nó_LoadImage, 'image'].")
        uploaded = self._upload_image(image_path)
        workflow = inject_inputs(
            self.video_cfg["workflow_data"],
            self.video_cfg["inputs"],
            {
                "prompt": prompt,
                "image": uploaded,
                "seed": int(seed),
                "width": int(width),
                "height": int(height),
                "frames": frames_for(seconds, self.fps, self.frame_multiple),
            },
        )
        return self._run(workflow, VIDEO_EXTS, output_path, progress_cb)


def make_backend(config: dict[str, Any], image_cfg: dict[str, Any], base_dir: Path):
    kind = str(config.get("type") or "local")
    if kind == "local":
        return LocalBackend(config, image_cfg)
    if kind == "comfyui":
        return ComfyUIBackend(config, base_dir)
    raise BackendError(f"backend.type '{kind}' não existe (use 'local' ou 'comfyui').")
