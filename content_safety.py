"""
Travas de segurança do pipeline automático (texto -> imagem -> vídeo).

O pipeline gera pessoas hiper-realistas em lote, sem revisão humana entre as etapas.
Por isso todo texto que chega à GPU passa por aqui: o que o usuário escreveu no YAML,
o que saiu das variações automáticas e o que um LLM local reescreveu.

Regras fixas (não são configuráveis de propósito):
  1. Só personagens fictícios criados pela IA: nada de semelhança com pessoas reais
     (celebridades, "sósia", "deepfake", troca de rosto, nomes da lista local).
  2. Nenhum indício de menor de idade (termos em PT/EN e idades abaixo de MIN_AGE).
  3. Toda pessoa da cena declara uma idade adulta explícita (>= MIN_AGE).

As listas pegam os casos óbvios e os erros de digitação comuns. Não substituem bom senso:
falsos positivos são aceitos (a mensagem diz qual termo bloqueou) para não deixar passar
nada ambíguo.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Margem acima dos 18: a idade aparente de uma pessoa gerada por IA é imprecisa.
MIN_AGE = 21


class SafetyError(ValueError):
    """Conteúdo recusado pelas travas do pipeline."""


@dataclass(frozen=True)
class Violation:
    rule: str
    match: str

    def __str__(self) -> str:
        return f"{self.rule}: '{self.match}'"


# Termos comparados depois de normalizar (minúsculas, sem acento, leetspeak desfeito).
_MINOR_TERMS = [
    # Inglês
    r"child(?:ren)?", r"childlike", r"child like", r"kids?", r"kiddie", r"minors", r"a minor",
    r"under ?age", r"teens?", r"teenage", r"teenagers?", r"pre ?teens?", r"tweens?",
    r"loli", r"lolita", r"lolicon", r"shota", r"shotacon", r"jailbait", r"barely legal",
    r"school ?girls?", r"school ?boys?", r"school uniform", r"high ?school",
    r"middle school", r"elementary school", r"junior high",
    r"young (?:girl|boy)s?", r"little (?:girl|boy)s?", r"baby face",
    r"bab(?:y|ies)(?! (?:oil|blue|pink))", r"toddlers?", r"infants?", r"juvenile", r"adolescents?",
    r"thirteen", r"fourteen", r"fifteen", r"sixteen", r"seventeen",
    # Português
    r"criancas?", r"infantil", r"menor(?:es)? de idade", r"adolescentes?",
    r"pre ?adolescentes?", r"novinh[ao]s?", r"ninfet(?:a|inha)s?", r"garotinh[ao]s?",
    r"menin[ao]s?", r"mocinhas?", r"colegia(?:l|is)", r"uniforme escolar", r"ensino medio",
    r"ensino fundamental", r"puberdade", r"treze", r"[cq]uatorze", r"catorze", r"quinze",
    r"dezesseis", r"dezessete",
]

# Termos que só fazem sentido com acento (sem acento viram outras palavras, ex.: "bebe" = beber).
_MINOR_TERMS_ACCENTED = [r"bebês?"]

_REAL_PERSON_TERMS = [
    # Inglês
    r"celebrit(?:y|ies)", r"famous", r"look ?alikes?", r"likeness", r"deep ?fakes?",
    r"face ?swap(?:s|ped|ping)?",
    r"real person", r"real people", r"based on a real",
    # Português
    r"celebridades?", r"famos[ao]s?", r"sosias?", r"parecid[ao]s? com", r"troca de rosto",
    r"pessoas? rea(?:l|is)",
]

# Idade escrita com número: "19 years old", "19yo", "19 y/o", "19 anos", "aged 19", "idade 19".
# Rodam sobre o texto já sem pontuação (ver _words).
_AGE_PATTERNS = [
    re.compile(r"\b(\d{1,3})\s*-?\s*(?:years?|yrs?)(?:\s*-?\s*old)?\b"),
    re.compile(r"\b(\d{1,3})\s*(?:yo|y o)\b"),  # "19yo" e "19 y/o" (a barra vira espaço)
    re.compile(r"\b(\d{1,3})\s*anos?\b"),
    re.compile(r"\b(?:aged?|idade(?:\s+de)?)\s*(\d{1,3})\b"),
]

# Idades por extenso que já são maiores de idade, mas ficam abaixo da margem de MIN_AGE.
_AGE_WORDS = {"eighteen": 18, "nineteen": 19, "dezoito": 18, "dezenove": 19}
_AGE_WORDS_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(_AGE_WORDS) + r")(?![a-z0-9])")

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _words(text: str) -> str:
    """Texto só com letras/números separados por um espaço (pra casar com \\b de forma estável)."""
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


def _compile(terms: list[str]) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(terms) + r")(?![a-z0-9])")


_MINOR_RE = _compile(_MINOR_TERMS)
_MINOR_ACCENTED_RE = re.compile(r"(?<!\w)(?:" + "|".join(_MINOR_TERMS_ACCENTED) + r")(?!\w)")
_REAL_PERSON_RE = _compile(_REAL_PERSON_TERMS)


def normalize(text: str) -> str:
    """Minúsculas, sem acento, sem pontuação; leetspeak desfeito (t33n -> teen)."""
    return _words(_strip_accents(text.lower()).translate(_LEET))


def load_blocked_names(path: Optional[Path]) -> list[str]:
    """Lê uma lista local de nomes de pessoas reais a bloquear (um por linha, '#' comenta)."""
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lista de nomes bloqueados não encontrada: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def find_violations(text: str, blocked_names: Iterable[str] = ()) -> list[Violation]:
    """Lista tudo que bloqueia o texto. Lista vazia = liberado."""
    if not text:
        return []
    violations: list[Violation] = []
    lowered = text.lower()
    plain = _words(_strip_accents(lowered))  # números intactos, para as idades
    leet = normalize(text)

    for pattern in _AGE_PATTERNS:
        for m in pattern.finditer(plain):
            if int(m.group(1)) < MIN_AGE:
                violations.append(Violation(f"idade abaixo de {MIN_AGE}", m.group(0).strip()))

    for m in _AGE_WORDS_RE.finditer(plain):
        if _AGE_WORDS[m.group(1)] < MIN_AGE:
            violations.append(Violation(f"idade abaixo de {MIN_AGE}", m.group(1)))

    for regex, rule in (
        (_MINOR_RE, "termo associado a menor de idade"),
        (_REAL_PERSON_RE, "semelhança com pessoa real / deepfake"),
    ):
        seen = set()
        for source in (plain, leet):
            for m in regex.finditer(source):
                if m.group(0) not in seen:
                    seen.add(m.group(0))
                    violations.append(Violation(rule, m.group(0)))

    for m in _MINOR_ACCENTED_RE.finditer(lowered):
        violations.append(Violation("termo associado a menor de idade", m.group(0)))

    for name in blocked_names:
        needle = normalize(name)
        if needle.strip() and needle in leet:
            violations.append(Violation("nome da lista de pessoas reais bloqueadas", name))

    return violations


def check_text(text: str, *, where: str, blocked_names: Iterable[str] = ()) -> None:
    """Levanta SafetyError se o texto violar alguma regra."""
    violations = find_violations(text, blocked_names)
    if violations:
        details = "; ".join(str(v) for v in violations)
        raise SafetyError(f"{where} recusado ({details}).")


def check_subject_age(age: object, *, where: str) -> int:
    """Confere que a idade declarada é um inteiro adulto (>= MIN_AGE) e devolve o valor."""
    if isinstance(age, bool) or not isinstance(age, int):
        raise SafetyError(
            f"{where}: declare a idade como número inteiro (ex.: age: 30). "
            f"Todo personagem precisa de idade adulta explícita, mínimo {MIN_AGE}."
        )
    if age < MIN_AGE:
        raise SafetyError(f"{where}: idade {age} abaixo do mínimo de {MIN_AGE} anos.")
    if age > 120:
        raise SafetyError(f"{where}: idade {age} inválida.")
    return age
