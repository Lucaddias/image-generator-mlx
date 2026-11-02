"""
Montador de prompts do pipeline automático.

Arquivo de lote (YAML) -> lista de JobSpec -> prompt de imagem + prompt de vídeo.

  - Personagens fictícios ficam num cadastro (`characters:`) e são reutilizados por nome.
    O nome nunca vai para o prompt, só a descrição e a idade.
  - `generate:` cria jobs automaticamente sorteando combinações das listas em `pick:`.
    O sorteio é determinístico por índice: aumentar `count` mantém os jobs já feitos.
  - Um LLM local (mlx-lm) pode reescrever os rascunhos. A saída dele passa pelas mesmas
    travas; se falhar, o pipeline volta para o rascunho do template.

Nada aqui importa MLX no topo: dá para validar e pré-visualizar lotes em qualquer máquina.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

import content_safety as cs

REPO_DIR = Path(__file__).resolve().parent

# Vocabulário de realismo fotográfico. O conteúdo da cena vem do lote; aqui é só "câmera".
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "photo": {
        "image_prefix": "RAW photo, hyper-realistic photograph",
        "image_suffix": (
            "shot on a full-frame camera, 85mm lens, f/1.8, natural skin texture with visible pores, "
            "subsurface scattering, sharp focus on the eyes, subtle film grain, Kodak Portra 400 colors"
        ),
        "video_style": (
            "Photorealistic live-action footage with natural skin texture, realistic motion blur "
            "and true-to-life proportions, shot on a cinema camera"
        ),
    },
    "cinematic": {
        "image_prefix": "cinematic film still, hyper-realistic",
        "image_suffix": (
            "anamorphic lens, shallow depth of field, volumetric light, professional color grading, "
            "35mm film grain, natural skin texture"
        ),
        "video_style": "Cinematic live-action footage, anamorphic lens, shallow depth of field, 24 fps film look",
    },
    "studio": {
        "image_prefix": "professional studio photograph, hyper-realistic",
        "image_suffix": (
            "softbox key light with rim light, seamless backdrop, 100mm lens, f/4, "
            "high detail, natural skin texture"
        ),
        "video_style": "Studio footage with controlled soft lighting and a steady camera, photorealistic",
    },
    "amateur": {
        "image_prefix": "candid smartphone photo, hyper-realistic",
        "image_suffix": "natural ambient light, slight sensor noise, unedited, realistic proportions",
        "video_style": "Handheld smartphone footage, natural light, slight camera shake, photorealistic",
    },
}

JOB_FIELDS = {
    "id", "subjects", "style", "scene", "action", "wardrobe", "lighting",
    "extra", "motion", "camera", "sound", "seed",
}
TEXT_FIELDS = ("scene", "action", "wardrobe", "lighting", "extra", "motion", "camera", "sound")
# O que muda a imagem. O resto (motion, camera, sound) só muda o vídeo.
IMAGE_FIELDS = ("subjects", "style", "scene", "action", "wardrobe", "lighting", "extra", "seed")

TOP_LEVEL_KEYS = {
    "backend", "image", "video", "llm", "blocked_names_file",
    "characters", "defaults", "jobs", "generate",
}
DEFAULT_IMAGE = {"width": 768, "height": 1024, "lora": None}
DEFAULT_VIDEO = {"width": 576, "height": 768, "seconds": 4}
DEFAULT_BACKEND = {"type": "local", "low_ram": False}
DEFAULT_LLM = {"model": None, "max_tokens": 300, "temperature": 0.7}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}


class BatchError(ValueError):
    """Erro de estrutura no arquivo de lote (campo desconhecido, tipo errado...)."""


@dataclass(frozen=True)
class Character:
    description: str
    age: int


@dataclass
class JobSpec:
    id: str
    subjects: list[Character]
    style: str = "photo"
    scene: str = ""
    action: str = ""
    wardrobe: str = ""
    lighting: str = ""
    extra: str = ""
    motion: str = ""
    camera: str = ""
    sound: str = ""
    seed: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Batch:
    name: str
    source: Path
    jobs: list[JobSpec]
    rejected: list[tuple[str, str]]
    image: dict[str, Any]
    video: dict[str, Any]
    backend: dict[str, Any]
    llm: Optional[dict[str, Any]]
    blocked_names: list[str] = field(default_factory=list)

    def image_hash(self, job: JobSpec) -> str:
        spec = job.to_dict()
        payload = {
            "job": {k: spec[k] for k in IMAGE_FIELDS},
            "image": self.image,
            "backend": self.backend.get("type"),
            "llm": (self.llm or {}).get("model"),
        }
        return _hash(payload)

    def video_hash(self, job: JobSpec) -> str:
        payload = {"job": job.to_dict(), "video": self.video, "image_hash": self.image_hash(job)}
        return _hash(payload)


def _hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stable_seed(job_id: str) -> int:
    """Seed fixa derivada do id: o mesmo job gera a mesma imagem ao retomar o lote."""
    return int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


# --------------------------------------------------------------------------- leitura do lote


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BatchError(f"'{where}' precisa ser um mapa (chave: valor).")
    return value


def _parse_character(raw: Any, where: str, blocked_names: Iterable[str]) -> Character:
    raw = _require_mapping(raw, where)
    unknown = set(raw) - {"description", "age"}
    if unknown:
        raise BatchError(f"{where}: campos desconhecidos {sorted(unknown)} (use description e age).")
    description = str(raw.get("description") or "").strip()
    if not description:
        raise BatchError(f"{where}: 'description' é obrigatório.")
    age = cs.check_subject_age(raw.get("age"), where=where)
    cs.check_text(description, where=f"{where} (description)", blocked_names=blocked_names)
    return Character(description=description, age=age)


def _parse_characters(raw: Any, blocked_names: Iterable[str]) -> dict[str, Character]:
    characters = {}
    for name, spec in _require_mapping(raw, "characters").items():
        where = f"personagem '{name}'"
        # O nome não entra no prompt, mas nome de pessoa real no cadastro já é sinal de deepfake.
        cs.check_text(str(name), where=f"{where} (nome)", blocked_names=blocked_names)
        characters[str(name)] = _parse_character(spec, where, blocked_names)
    return characters


def _parse_subjects(raw: Any, job_id: str, characters: dict[str, Character], blocked_names) -> list[Character]:
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise BatchError(f"job '{job_id}': 'subjects' precisa listar ao menos um personagem.")
    subjects = []
    for i, item in enumerate(raw, start=1):
        where = f"job '{job_id}', personagem {i}"
        if isinstance(item, str):
            if item not in characters:
                raise BatchError(f"{where}: '{item}' não está no cadastro 'characters'.")
            subjects.append(characters[item])
        else:
            subjects.append(_parse_character(item, where, blocked_names))
    return subjects


def parse_job(raw: dict[str, Any], characters: dict[str, Character], blocked_names: Iterable[str] = ()) -> JobSpec:
    """Valida um job já mesclado com os defaults. Erros de estrutura: BatchError; de conteúdo: SafetyError."""
    unknown = set(raw) - JOB_FIELDS
    if unknown:
        raise BatchError(f"job '{raw.get('id', '?')}': campos desconhecidos {sorted(unknown)}.")
    job_id = str(raw.get("id") or "")
    if not _ID_RE.match(job_id):
        raise BatchError(f"id de job inválido: '{job_id}' (use letras, números, '-' e '_', até 64).")

    style = str(raw.get("style") or "photo")
    if style not in STYLE_PRESETS:
        raise BatchError(f"job '{job_id}': style '{style}' não existe ({', '.join(STYLE_PRESETS)}).")

    texts = {}
    for name in TEXT_FIELDS:
        value = raw.get(name)
        if value is not None and not isinstance(value, (str, int, float)):
            raise BatchError(f"job '{job_id}': '{name}' precisa ser texto.")
        texts[name] = "" if value is None else str(value).strip()

    seed = raw.get("seed", -1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BatchError(f"job '{job_id}': 'seed' precisa ser inteiro.")

    subjects = _parse_subjects(raw.get("subjects"), job_id, characters, blocked_names)
    for name, value in texts.items():
        cs.check_text(value, where=f"job '{job_id}', campo '{name}'", blocked_names=blocked_names)

    return JobSpec(
        id=job_id,
        subjects=subjects,
        style=style,
        seed=seed if seed >= 0 else stable_seed(job_id),
        **texts,
    )


def _generated_jobs(raw: Any) -> list[dict[str, Any]]:
    """Expande o bloco `generate:` em jobs crus, sorteando uma opção de cada lista em `pick:`."""
    gen = _require_mapping(raw, "generate")
    if not gen:
        return []
    unknown = set(gen) - {"count", "seed", "id_prefix", "base", "pick"}
    if unknown:
        raise BatchError(f"generate: campos desconhecidos {sorted(unknown)}.")
    count = gen.get("count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1000:
        raise BatchError("generate.count precisa ser um inteiro entre 1 e 1000.")
    base = _require_mapping(gen.get("base"), "generate.base")
    pick = _require_mapping(gen.get("pick"), "generate.pick")
    for key, options in pick.items():
        if key not in JOB_FIELDS - {"id", "seed"}:
            raise BatchError(f"generate.pick: '{key}' não é um campo sorteável.")
        if not isinstance(options, list) or not options:
            raise BatchError(f"generate.pick.{key} precisa ser uma lista com opções.")
    prefix = str(gen.get("id_prefix") or "auto")
    batch_seed = gen.get("seed", 0)

    jobs = []
    for i in range(count):
        # Um gerador por índice: o job N é sempre o mesmo, não importa o count.
        rng = random.Random(f"{batch_seed}:{i}")
        picked = {key: rng.choice(pick[key]) for key in sorted(pick)}
        jobs.append(_merge(base, picked) | {"id": f"{prefix}-{i + 1:03d}"})
    return jobs


def _resolve_blocked_names(data: dict[str, Any], source: Path) -> list[str]:
    configured = data.get("blocked_names_file")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = source.parent / path
        return cs.load_blocked_names(path)
    default = REPO_DIR / "blocked_names.txt"
    return cs.load_blocked_names(default) if default.exists() else []


def load_batch(source: Path) -> Batch:
    source = Path(source)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BatchError(f"{source}: o arquivo precisa ser um mapa YAML.")
    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise BatchError(f"{source}: chaves desconhecidas {sorted(unknown)}.")

    image = _merge(DEFAULT_IMAGE, _require_mapping(data.get("image"), "image"))
    video = _merge(DEFAULT_VIDEO, _require_mapping(data.get("video"), "video"))
    backend = _merge(DEFAULT_BACKEND, _require_mapping(data.get("backend"), "backend"))
    llm_raw = _require_mapping(data.get("llm"), "llm")
    llm = _merge(DEFAULT_LLM, llm_raw) if llm_raw.get("model") else None

    blocked_names = _resolve_blocked_names(data, source)
    characters = _parse_characters(data.get("characters"), blocked_names)
    defaults = _require_mapping(data.get("defaults"), "defaults")

    raw_jobs = data.get("jobs") or []
    if not isinstance(raw_jobs, list):
        raise BatchError("'jobs' precisa ser uma lista.")
    raw_jobs = [_merge(defaults, _require_mapping(j, "jobs[]")) for j in raw_jobs]
    raw_jobs += [_merge(defaults, j) for j in _generated_jobs(data.get("generate"))]
    if not raw_jobs:
        raise BatchError(f"{source}: nenhum job em 'jobs' nem em 'generate'.")

    jobs, rejected, seen = [], [], set()
    for raw in raw_jobs:
        job_id = str(raw.get("id") or "")
        if job_id in seen:
            raise BatchError(f"id de job repetido: '{job_id}'.")
        seen.add(job_id)
        try:
            jobs.append(parse_job(raw, characters, blocked_names))
        except cs.SafetyError as exc:
            rejected.append((job_id, str(exc)))

    return Batch(
        name=source.stem,
        source=source.resolve(),
        jobs=jobs,
        rejected=rejected,
        image=image,
        video=video,
        backend=backend,
        llm=llm,
        blocked_names=blocked_names,
    )


# --------------------------------------------------------------------------- prompts


def subjects_phrase(subjects: list[Character]) -> str:
    """'an adult woman with red hair, 30 years old' — a idade adulta sempre explícita no prompt."""
    parts = []
    for c in subjects:
        desc = c.description.strip().rstrip(".,;")
        desc = re.sub(r"(?i)^(?:an?|the)\s+", "", desc)
        if not re.match(r"(?i)^adult\b", desc):
            desc = f"adult {desc}"
        parts.append(f"an {desc}, {c.age} years old")
    if len(parts) == 1:
        return parts[0]
    return f"{_NUMBER_WORDS.get(len(parts), len(parts))} adults: " + "; ".join(parts)


def _clean(text: str) -> str:
    return text.strip().rstrip(",.; ")


def draft_image_prompt(job: JobSpec) -> str:
    preset = STYLE_PRESETS[job.style]
    parts = [
        preset["image_prefix"],
        subjects_phrase(job.subjects),
        job.wardrobe,
        job.action,
        job.scene,
        job.lighting,
        job.extra,
        preset["image_suffix"],
    ]
    return ", ".join(_clean(p) for p in parts if p and _clean(p))


def draft_video_prompt(job: JobSpec) -> str:
    """Parágrafo corrido: ação -> pessoas -> cenário -> câmera -> luz -> áudio (formato que LTX/Wan entendem melhor)."""
    preset = STYLE_PRESETS[job.style]
    sentences = []
    if job.motion:
        sentences.append(job.motion[0].upper() + job.motion[1:])
    who = subjects_phrase(job.subjects)
    if job.wardrobe:
        wardrobe = job.wardrobe if job.wardrobe.lower().startswith("wearing") else f"wearing {job.wardrobe}"
        who = f"{who}, {wardrobe}"
    sentences.append(f"The scene shows {who}")
    if job.scene:
        sentences.append(f"Setting: {job.scene}")
    if job.camera:
        sentences.append(f"Camera: {job.camera}")
    if job.lighting:
        sentences.append(f"Lighting: {job.lighting}")
    sentences.append(preset["video_style"])
    if job.sound:
        sentences.append(f"Audio: {job.sound}")
    return " ".join(_clean(s) + "." for s in sentences if _clean(s))


_LLM_RULES = (
    "Every person is a fictional adult created entirely by AI: keep the word 'adult' and every age "
    "exactly as written. Do not add names, real people, celebrities, brands or anyone underage. "
    "Keep every fact from the draft. Output only the final prompt, in English, as one paragraph."
)
_LLM_INSTRUCTIONS = {
    "image": (
        "Rewrite the draft below into a richer, hyper-realistic photographic prompt for a text-to-image "
        "model (under 120 words): describe the people, clothing, pose, setting, lighting, lens and skin "
        "detail. " + _LLM_RULES
    ),
    "video": (
        "Rewrite the draft below into a prompt for an image-to-video model (under 150 words), as one "
        "flowing paragraph in present tense: start with the main action, then body movement and "
        "gestures, the people, the setting, camera movement, lighting and finally the sounds. " + _LLM_RULES
    ),
}


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*(?:here(?:'s| is)[^:\n]*:|(?:final )?prompt:)\s*", "", text.strip(), flags=re.I)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs[0].strip().strip('"').strip() if paragraphs else ""


def rewrite_is_faithful(rewritten: str, job: JobSpec) -> bool:
    """O LLM não pode tirar a âncora de idade adulta do prompt."""
    lowered = rewritten.lower()
    return "adult" in lowered and all(str(c.age) in rewritten for c in job.subjects)


class LLMRewriter:
    """Reescreve rascunhos com um LLM local via mlx-lm. Carrega uma vez por lote e libera no final."""

    def __init__(self, model: str, max_tokens: int = 300, temperature: float = 0.7):
        self.model_path = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._model = None
        self._tokenizer = None
        if "gemma" in model.lower():
            print(
                "[pipeline] Aviso: a Gemma Prohibited Use Policy proíbe gerar conteúdo sexualmente "
                "explícito. Escolha um LLM cuja licença permita o seu uso.",
                file=sys.stderr,
            )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_path)

    def generate(self, draft: str, kind: str, seed: int) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._ensure_loaded()
        # Instruções na mensagem do usuário: funciona até em templates sem papel "system".
        messages = [{"role": "user", "content": f"{_LLM_INSTRUCTIONS[kind]}\n\nDraft:\n{draft}"}]
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        mx.random.seed(seed)
        output = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )
        return clean_llm_output(output)

    def close(self) -> None:
        if self._model is None:
            return
        import mlx.core as mx

        self._model = self._tokenizer = None
        gc.collect()
        mx.clear_cache()


def finalize_prompt(
    draft: str,
    kind: str,
    job: JobSpec,
    rewriter: Optional[LLMRewriter] = None,
    blocked_names: Iterable[str] = (),
) -> tuple[str, str]:
    """Devolve (prompt, origem). Origem: 'template', 'llm' ou 'template (llm recusado: motivo)'."""
    where = f"prompt de {'imagem' if kind == 'image' else 'vídeo'} do job '{job.id}'"
    cs.check_text(draft, where=where, blocked_names=blocked_names)
    if rewriter is None:
        return draft, "template"

    try:
        rewritten = rewriter.generate(draft, kind, job.seed)
    except Exception as exc:  # o LLM é opcional: falha nele não derruba o lote
        return draft, f"template (llm falhou: {exc})"
    if not rewritten:
        return draft, "template (llm devolveu vazio)"
    if not rewrite_is_faithful(rewritten, job):
        return draft, "template (llm removeu a idade adulta)"
    violations = cs.find_violations(rewritten, blocked_names)
    if violations:
        return draft, f"template (llm recusado: {violations[0]})"
    return rewritten, "llm"
