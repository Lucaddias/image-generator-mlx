#!/usr/bin/env python3
"""
Pipeline automático: lote YAML -> prompts -> imagens -> vídeos, sem intervenção.

Só personagens fictícios criados pela IA. O fluxo começa sempre em texto, não aceita foto de
referência, e o vídeo só anima imagens geradas pelo próprio pipeline (conferidas por SHA-256).
Todo prompt passa pela validação de content_safety.py e todo arquivo sai marcado como gerado por IA.

Etapas (cada uma roda para o lote inteiro antes da próxima, para carregar cada modelo uma vez):
  1. prompts  — template + LLM local opcional, validados
  2. imagens  — backend local (Qwen-Image) ou ComfyUI
  3. vídeos   — imagem -> vídeo no mesmo backend

O estado fica em outputs/pipeline/<lote>/manifest.json. Rodar de novo retoma de onde parou,
tenta de novo os jobs que deram erro e refaz só o que mudou no YAML.
Apagou uma imagem de que não gostou? O job vira "removido" e não ganha vídeo.

Exemplos:
  .venv/bin/python pipeline.py preview jobs/exemplo.yaml
  .venv/bin/python pipeline.py run jobs/exemplo.yaml
  .venv/bin/python pipeline.py run jobs/exemplo.yaml --stop-after images   # curadoria antes dos vídeos
  .venv/bin/python pipeline.py watch queue/                                # processa cada YAML que cair na pasta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

import content_safety as cs
from prompt_builder import (
    Batch,
    BatchError,
    LLMRewriter,
    draft_image_prompt,
    draft_video_prompt,
    finalize_prompt,
    load_batch,
)

DEFAULT_OUT = Path("outputs") / "pipeline"
NOTICE = (
    "AI-generated content. Fictional characters created entirely by AI."
)
GENERATOR = "image-generator-mlx/pipeline"

PENDING, PROMPTED, IMAGED, DONE, REMOVED, FAILED = (
    "pending", "prompted", "imaged", "done", "removed", "failed",
)


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- trava de GPU


class GpuLock:
    """Impede dois pipelines ao mesmo tempo: dois vídeos juntos passam dos 24 GB e o Mac entra em swap."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pid = self._owner()
                if pid is not None and self._alive(pid):
                    raise RuntimeError(f"Outro pipeline já está usando a GPU (pid {pid}, trava {self.path}).")
                self.path.unlink(missing_ok=True)  # trava de um processo que morreu
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
            return self
        raise RuntimeError(f"Não consegui criar a trava {self.path}.")

    def __exit__(self, *exc: Any) -> None:
        if self._owner() == os.getpid():
            self.path.unlink(missing_ok=True)

    def _owner(self) -> Optional[int]:
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


# --------------------------------------------------------------------------- procedência


def provenance(batch: Batch, job_id: str, entry: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "ai_generated": True,
        "notice": NOTICE,
        "generator": GENERATOR,
        "backend": batch.backend.get("type"),
        "batch": batch.name,
        "job_id": job_id,
        "kind": kind,
        "seed": entry["seed"],
        "prompt": entry["image_prompt"] if kind == "image" else entry["video_prompt"],
        "source_image_sha256": entry.get("image_sha256") if kind == "video" else None,
        "created_at": now_iso(),
    }


def stamp_png(path: Path, info: dict[str, Any]) -> None:
    """Grava a marca de IA nos metadados do PNG, preservando os que já existiam (ex.: workflow do ComfyUI)."""
    from PIL import Image, PngImagePlugin

    tmp = path.with_name(path.stem + ".stamping.png")
    with Image.open(path) as im:
        im.load()
        meta = PngImagePlugin.PngInfo()
        for key, value in (getattr(im, "text", None) or {}).items():
            if not key.startswith("AI-") and isinstance(value, str):
                meta.add_itxt(key, value)
        meta.add_itxt("AI-Generated", "true")
        meta.add_itxt("AI-Notice", NOTICE)
        meta.add_itxt("AI-Provenance", json.dumps(info, ensure_ascii=False))
        save_kwargs: dict[str, Any] = {"format": "PNG", "pnginfo": meta}
        for key in ("icc_profile", "exif", "dpi"):
            if im.info.get(key):
                save_kwargs[key] = im.info[key]
        im.save(tmp, **save_kwargs)
    os.replace(tmp, path)


def stamp_video(path: Path, info: dict[str, Any]) -> None:
    """Remux sem recodificar, só adicionando os metadados de procedência."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log(f"ffmpeg não encontrado: {path.name} fica só com o .provenance.json.")
        return
    tmp = path.with_name(path.stem + ".stamping" + path.suffix)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-map", "0", "-c", "copy",
        "-metadata", f"comment={NOTICE}",
        "-metadata", f"description=AI-generated by {GENERATOR}, job {info['job_id']}, seed {info['seed']}",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        log(f"Não consegui gravar metadados em {path.name}: {proc.stderr.strip()[-300:]}")
        return
    os.replace(tmp, path)


def write_sidecar(path: Path, info: dict[str, Any]) -> None:
    path.with_name(path.name + ".provenance.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- manifesto


class Manifest:
    def __init__(self, path: Path, batch: Batch):
        self.path = Path(path)
        data: dict[str, Any] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
        self.jobs: dict[str, dict[str, Any]] = data.get("jobs", {})
        self.data = {"batch": batch.name, "source": str(batch.source), "jobs": self.jobs}

    def save(self, rejected: list[tuple[str, str]]) -> None:
        self.data["updated_at"] = now_iso()
        self.data["rejected"] = dict(rejected)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)


def _fresh_entry(seed: int, image_hash: str, video_hash: str) -> dict[str, Any]:
    return {
        "status": PENDING,
        "seed": seed,
        "image_hash": image_hash,
        "video_hash": video_hash,
        "image_prompt": None,
        "image_prompt_source": None,
        "video_prompt": None,
        "video_prompt_source": None,
        "image_path": None,
        "image_sha256": None,
        "video_path": None,
        "removed": False,
        "error": None,
        "timings": {},
    }


def _status(entry: dict[str, Any]) -> str:
    if entry.get("removed"):
        return REMOVED
    if entry.get("error"):
        return FAILED
    if entry.get("video_path"):
        return DONE
    if entry.get("image_path"):
        return IMAGED
    if entry.get("image_prompt") and entry.get("video_prompt"):
        return PROMPTED
    return PENDING


def sync_entries(batch: Batch, manifest: Manifest, redo: set[str]) -> None:
    """Alinha o manifesto com o YAML: cria jobs novos, refaz os que mudaram, marca arquivos apagados."""
    for job in batch.jobs:
        image_hash, video_hash = batch.image_hash(job), batch.video_hash(job)
        entry = manifest.jobs.get(job.id)
        if entry is None or entry.get("image_hash") != image_hash or job.id in redo:
            if entry is not None:
                log(f"{job.id}: {'refazendo (--redo)' if job.id in redo else 'mudou no YAML, refazendo'}.")
            manifest.jobs[job.id] = entry = _fresh_entry(job.seed, image_hash, video_hash)
        elif entry.get("video_hash") != video_hash:
            log(f"{job.id}: parte do vídeo mudou no YAML, a imagem é mantida.")
            entry.update(video_hash=video_hash, video_prompt=None, video_prompt_source=None, video_path=None)

        entry["error"] = None  # erros anteriores ganham nova tentativa
        for key in ("image_path", "video_path"):
            if entry.get(key) and not Path(entry[key]).exists() and not entry.get("removed"):
                log(f"{job.id}: {Path(entry[key]).name} foi apagado, o job fica como removido.")
                entry["removed"] = True
        entry["status"] = _status(entry)


# --------------------------------------------------------------------------- etapas


def validate_sizes(batch: Batch) -> None:
    iw, ih = int(batch.image["width"]), int(batch.image["height"])
    vw, vh = int(batch.video["width"]), int(batch.video["height"])
    if batch.backend.get("type", "local") == "local":
        if any(v < 256 or v > 2048 or v % 32 for v in (iw, ih)):
            raise BatchError("image.width/height: múltiplos de 32 entre 256 e 2048 (Qwen-Image).")
        if any(v < 256 or v > 1280 or v % 64 for v in (vw, vh)):
            raise BatchError("video.width/height: múltiplos de 64 entre 256 e 1280 (LTX-2.3).")
    if abs(iw / ih - vw / vh) > 0.02:
        log(f"Aviso: a imagem ({iw}×{ih}) e o vídeo ({vw}×{vh}) têm proporções diferentes; o vídeo vai cortar ou distorcer.")


def stage_prompts(batch: Batch, manifest: Manifest) -> None:
    pending = [j for j in batch.jobs if _needs_prompts(manifest.jobs[j.id])]
    if not pending:
        return
    log(f"Etapa 1/3 — prompts de {len(pending)} job(s){' com LLM local' if batch.llm else ''}.")
    rewriter = LLMRewriter(**batch.llm) if batch.llm else None
    try:
        for job in pending:
            entry = manifest.jobs[job.id]
            try:
                if not entry["image_prompt"]:
                    entry["image_prompt"], entry["image_prompt_source"] = finalize_prompt(
                        draft_image_prompt(job), "image", job, rewriter, batch.blocked_names
                    )
                if not entry["video_prompt"]:
                    entry["video_prompt"], entry["video_prompt_source"] = finalize_prompt(
                        draft_video_prompt(job), "video", job, rewriter, batch.blocked_names
                    )
            except cs.SafetyError as exc:
                entry["error"] = str(exc)
                log(f"{job.id}: {exc}")
            entry["status"] = _status(entry)
            manifest.save(batch.rejected)
    finally:
        if rewriter is not None:
            rewriter.close()


def _needs_prompts(entry: dict[str, Any]) -> bool:
    return not entry.get("removed") and not (entry.get("image_prompt") and entry.get("video_prompt"))


def _run_job_stage(batch, manifest, job, label, fn) -> None:
    entry = manifest.jobs[job.id]
    started = time.monotonic()
    try:
        fn(entry)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # um job com problema não derruba o lote
        entry["error"] = f"{label}: {exc}"
        log(f"{job.id}: falhou em {label}: {exc}")
    else:
        entry["timings"][label] = round(time.monotonic() - started, 1)
    entry["status"] = _status(entry)
    manifest.save(batch.rejected)


def stage_images(batch: Batch, manifest: Manifest, backend, out_dir: Path) -> None:
    todo = [j for j in batch.jobs if _status(manifest.jobs[j.id]) == PROMPTED]
    if not todo:
        return
    log(f"Etapa 2/3 — {len(todo)} imagem(ns) no backend '{backend.name}'.")
    for n, job in enumerate(todo, start=1):
        def generate(entry: dict[str, Any], job=job, n=n) -> None:
            log(f"[{n}/{len(todo)}] imagem {job.id} (seed {entry['seed']})")
            path = backend.generate_image(
                prompt=entry["image_prompt"],
                width=batch.image["width"],
                height=batch.image["height"],
                seed=entry["seed"],
                output_path=out_dir / f"{job.id}.image.png",
                progress_cb=lambda msg: log(f"  {msg}"),
            )
            info = provenance(batch, job.id, entry, "image")
            if path.suffix.lower() == ".png":
                stamp_png(path, info)
            write_sidecar(path, info)
            entry["image_sha256"] = sha256_file(path)
            entry["image_path"] = str(path.resolve())

        _run_job_stage(batch, manifest, job, "imagem", generate)


def stage_videos(batch: Batch, manifest: Manifest, backend, out_dir: Path) -> None:
    todo = [j for j in batch.jobs if _status(manifest.jobs[j.id]) == IMAGED]
    if not todo:
        return
    log(f"Etapa 3/3 — {len(todo)} vídeo(s) no backend '{backend.name}'.")
    for n, job in enumerate(todo, start=1):
        def generate(entry: dict[str, Any], job=job, n=n) -> None:
            image_path = Path(entry["image_path"])
            # Só anima o que o pipeline gerou: trocar o arquivo por outra foto quebra o hash.
            if sha256_file(image_path) != entry["image_sha256"]:
                raise cs.SafetyError(
                    f"{image_path.name} foi alterada fora do pipeline. O vídeo só anima imagens geradas "
                    f"pelo próprio pipeline; use --redo {job.id} para gerar outra."
                )
            log(f"[{n}/{len(todo)}] vídeo {job.id} ({batch.video['seconds']}s)")
            path = backend.generate_video(
                prompt=entry["video_prompt"],
                image_path=image_path,
                width=batch.video["width"],
                height=batch.video["height"],
                seconds=batch.video["seconds"],
                seed=entry["seed"],
                output_path=out_dir / f"{job.id}.video.mp4",
                progress_cb=lambda msg: log(f"  {msg}"),
            )
            info = provenance(batch, job.id, entry, "video")
            stamp_video(path, info)
            write_sidecar(path, info)
            entry["video_path"] = str(path.resolve())

        _run_job_stage(batch, manifest, job, "vídeo", generate)


def summarize(batch: Batch, manifest: Manifest, out_dir: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in batch.jobs:
        status = manifest.jobs[job.id]["status"]
        counts[status] = counts.get(status, 0) + 1
    if batch.rejected:
        counts["rejected"] = len(batch.rejected)
    return {
        "batch": batch.name,
        "output_dir": str(out_dir.resolve()),
        "manifest": str(manifest.path.resolve()),
        "counts": counts,
        "failed": {j.id: manifest.jobs[j.id]["error"] for j in batch.jobs if manifest.jobs[j.id]["error"]},
        "rejected": dict(batch.rejected),
    }


def run_batch(batch: Batch, out_root: Path, stop_after: Optional[str] = None,
              redo: Optional[set[str]] = None, force: bool = False) -> dict[str, Any]:
    validate_sizes(batch)
    out_dir = Path(out_root) / batch.name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_dir / "manifest.json", batch)
    redo = set(redo or ())
    if force:
        redo |= {j.id for j in batch.jobs}
    unknown = redo - {j.id for j in batch.jobs}
    if unknown:
        raise BatchError(f"--redo com ids que não existem no lote: {sorted(unknown)}")

    for job_id, reason in batch.rejected:
        log(f"{job_id}: RECUSADO — {reason}")

    sync_entries(batch, manifest, redo)
    manifest.save(batch.rejected)

    stage_prompts(batch, manifest)
    if stop_after != "prompts":
        from backends import make_backend

        backend = make_backend(batch.backend, batch.image, batch.source.parent)
        stage_images(batch, manifest, backend, out_dir)
        if stop_after != "images":
            stage_videos(batch, manifest, backend, out_dir)

    return summarize(batch, manifest, out_dir)


# --------------------------------------------------------------------------- comandos


def cmd_preview(args: argparse.Namespace) -> int:
    batch = load_batch(args.batch)
    validate_sizes(batch)
    print(f"Lote '{batch.name}': {len(batch.jobs)} job(s) válidos, {len(batch.rejected)} recusado(s). "
          f"Backend: {batch.backend.get('type')}. LLM: {(batch.llm or {}).get('model') or 'não (só template)'}.\n")
    for job_id, reason in batch.rejected:
        print(f"✗ {job_id}: {reason}\n")
    for job in batch.jobs:
        print(f"● {job.id}  (seed {job.seed}, estilo {job.style})")
        for kind, draft in (("image", draft_image_prompt(job)), ("video", draft_video_prompt(job))):
            prompt, _ = finalize_prompt(draft, kind, job, None, batch.blocked_names)
            print(f"  {'IMAGEM' if kind == 'image' else 'VÍDEO '}: {prompt}")
        print()
    return 1 if batch.rejected else 0


def cmd_run(args: argparse.Namespace) -> int:
    batch = load_batch(args.batch)
    with GpuLock(args.out / ".gpu.lock"):
        summary = run_batch(batch, args.out, args.stop_after, set(args.redo or ()), args.force)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["failed"] else 2


def _move(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = dest_dir / f"{stamp}_{path.name}"
    shutil.move(str(path), target)
    return target


def cmd_watch(args: argparse.Namespace) -> int:
    queue = Path(args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    log(f"Observando {queue.resolve()} a cada {args.interval}s. Solte arquivos .yaml ali. Ctrl+C para sair.")
    with GpuLock(args.out / ".gpu.lock"):
        while True:
            ready = sorted(
                (p for p in queue.iterdir()
                 if p.is_file() and p.suffix in (".yaml", ".yml")
                 and time.time() - p.stat().st_mtime > 5),  # espera o arquivo terminar de ser escrito
                key=lambda p: p.stat().st_mtime,
            )
            for path in ready:
                log(f"Novo lote: {path.name}")
                try:
                    summary = run_batch(load_batch(path), args.out)
                    dest = queue / ("failed" if summary["failed"] else "done")
                    report = summary
                except (BatchError, cs.SafetyError, yaml.YAMLError, RuntimeError, OSError, ValueError) as exc:
                    log(f"{path.name}: {exc}")
                    dest, report = queue / "failed", {"error": str(exc)}
                moved = _move(path, dest)
                moved.with_suffix(".result.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                log(f"{path.name} -> {dest.name}/")
            time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline automático prompt -> imagem -> vídeo (personagens fictícios criados por IA)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_preview = sub.add_parser("preview", help="Valida o lote e mostra os prompts, sem usar a GPU.")
    p_preview.add_argument("batch", type=Path)
    p_preview.set_defaults(func=cmd_preview)

    p_run = sub.add_parser("run", help="Roda (ou retoma) um lote inteiro.")
    p_run.add_argument("batch", type=Path)
    p_run.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Pasta de saída (padrão: {DEFAULT_OUT}).")
    p_run.add_argument("--stop-after", choices=["prompts", "images"], default=None,
                       help="Para depois dessa etapa (ex.: revisar imagens antes dos vídeos).")
    p_run.add_argument("--redo", nargs="+", metavar="ID", help="Refaz esses jobs do zero.")
    p_run.add_argument("--force", action="store_true", help="Refaz todos os jobs do zero.")
    p_run.add_argument("--json", action="store_true", help="Resumo final em uma linha JSON.")
    p_run.set_defaults(func=cmd_run)

    p_watch = sub.add_parser("watch", help="Processa automaticamente cada YAML que aparecer numa pasta.")
    p_watch.add_argument("queue", type=Path, nargs="?", default=Path("queue"))
    p_watch.add_argument("--interval", type=float, default=30.0, help="Segundos entre verificações (padrão: 30).")
    p_watch.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Pasta de saída (padrão: {DEFAULT_OUT}).")
    p_watch.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (BatchError, cs.SafetyError, yaml.YAMLError, RuntimeError, FileNotFoundError) as exc:
        log(f"Erro: {exc}")
        return 1
    except KeyboardInterrupt:
        log("Interrompido. Rode o mesmo comando para retomar.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
