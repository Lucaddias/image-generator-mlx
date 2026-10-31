# 🎨 Image Generator MLX — Qwen-Image-2.1-Turbo local no Apple Silicon

Fluxo local completo para geração de imagens com **Qwen-Image-2.1-Turbo (7B DiT)**, otimizado para Macs Apple Silicon via **MLX 4-bit** com carregamento em estágios (*staged loading*). Tudo roda 100% offline depois do download do modelo.

**Built with Qwen.**

## ⚡ Destaques

- **Tamanho em disco:** `~9.9 GB` (em vez dos 33 GB em BF16)
- **Pico de memória GPU (MLX):** `~4.7 GB` a 512×512, graças ao carregamento em 3 estágios
- **Passos de inferência:** `8 passos` (schedule nativo Turbo)
- **Texto na imagem:** escreve palavras dentro da imagem, basta colocá-las entre `"aspas"` no prompt
- **Duas formas de uso:** interface web (Gradio) e CLI (ótima para agentes como Claude Code, OpenCode e Codex)

---

## 📋 Requisitos

| Item | Mínimo |
| --- | --- |
| Computador | Mac com Apple Silicon (M1, M2, M3, M4, M5…). **Macs Intel não são suportados.** |
| Sistema | macOS 14 (Sonoma) ou mais recente |
| Memória | 16 GB de RAM unificada (24 GB+ recomendado para resoluções acima de 768 px) |
| Disco | ~12 GB livres (9.9 GB do modelo + ~1.3 GB do ambiente Python) |
| Software | Git e Python 3.12 (o `uv` instala o Python para você) |

---

## 🛠️ Instalação

### 1. Instale as ferramentas básicas

Abra o **Terminal** e instale as ferramentas de linha de comando da Apple (traz o Git):

```bash
xcode-select --install
```

Instale o [uv](https://docs.astral.sh/uv/), gerenciador de Python que cuida da versão certa automaticamente:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Feche e abra o Terminal de novo para o comando `uv` ficar disponível.

### 2. Clone o repositório

```bash
git clone https://github.com/Lucaddias/image-generator-mlx.git
```

```bash
cd image-generator-mlx
```

### 3. Crie o ambiente virtual com Python 3.12

```bash
uv venv --python 3.12
```

### 4. Instale as dependências

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

> A instalação baixa ~1.3 GB (PyTorch, MLX, Gradio e o [MFLUX](https://github.com/mflux-community/mflux), que vem direto do GitHub numa versão fixada).

### 5. Baixe o modelo (~9.9 GB)

```bash
.venv/bin/python generate.py --download-only
```

Os pesos são baixados do Hugging Face ([joeynyc/Studio-Image-Turbo-MLX-4bit-from-Qwen-2.1](https://huggingface.co/joeynyc/Studio-Image-Turbo-MLX-4bit-from-Qwen-2.1)) para `models/qwen-image-2.1-turbo-4bit/`. Esse passo é opcional: se você pular, o download acontece automaticamente na primeira geração.

### 6. Teste a instalação

```bash
.venv/bin/python generate.py --prompt 'A small red robot holding a white sign that says "HELLO", clean studio photograph' --seed 42
```

Se tudo estiver certo, a imagem aparece em `outputs/` junto com um relatório `.run.json` de tempo e memória.

<details>
<summary><b>Alternativa: instalar sem o uv (pip puro)</b></summary>

Você precisa ter o Python 3.12 instalado (por exemplo, `brew install python@3.12`).

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install --upgrade pip
```

```bash
.venv/bin/pip install -r requirements.txt
```

Depois siga a partir do passo 5.
</details>

---

## 🚀 Uso 1: Interface web local (Gradio)

```bash
.venv/bin/python app.py
```

Depois acesse **http://127.0.0.1:7860** no navegador. A interface tem presets de proporção (quadrado, paisagem, retrato), seed, exemplos prontos e uma galeria com as imagens já geradas.

> Rode os comandos sempre de dentro da pasta do projeto: as imagens são salvas em `outputs/` relativo à pasta atual.

---

## 💻 Uso 2: Terminal / CLI

```bash
.venv/bin/python generate.py \
  --prompt 'A small red robot holding a white sign that says "HELLO M5", clean studio photograph, soft light' \
  --width 512 \
  --height 512 \
  --seed 42
```

### Opções do CLI

| Opção | Descrição |
| --- | --- |
| `--prompt` / `-p` | Descrição da imagem (textos entre `"aspas"` são desenhados na imagem) |
| `--width` / `-W` | Largura em pixels, múltiplo de 32 entre 256 e 2048 (padrão `512`) |
| `--height` / `-H` | Altura em pixels, múltiplo de 32 entre 256 e 2048 (padrão `512`) |
| `--seed` / `-s` | Seed numérica (`-1` para aleatória, padrão) |
| `--output` / `-o` | Caminho do `.png` de saída (padrão `outputs/qwen_<timestamp>.png`) |
| `--model-path` | Pasta onde o modelo está ou será baixado |
| `--download-only` | Só baixa/verifica o modelo e encerra |
| `--json` | Imprime apenas o JSON de resultado no stdout (ideal para agentes CLI) |

---

## 🤖 Uso 3: Claude Code, OpenCode ou Codex como "prompt enhancer"

Qualquer agente CLI pode transformar uma ideia curta em um prompt rico e chamar o motor local:

```bash
claude "Crie um prompt fotográfico detalhado em inglês para uma cafeteria futurista com placa neon escrita 'LUCA COFFEE' e execute .venv/bin/python generate.py --json -p '<SEU_PROMPT>'"
```

---

## 🧯 Solução de problemas

| Problema | Solução |
| --- | --- |
| `GPU Apple Silicon (Metal/MLX) não disponível` | O MLX só funciona em Macs Apple Silicon com macOS 14+. Confira em menu Apple > Sobre Este Mac. |
| `No solution found` / `mlx has no wheels` ao instalar | Seu macOS é anterior ao 14 ou o Python não é 3.12. Atualize o macOS ou recrie o ambiente com `uv venv --python 3.12`. |
| `command not found: uv` | Feche e abra o Terminal depois de instalar o uv, ou rode `source $HOME/.local/bin/env`. |
| Download do modelo interrompido | Rode `.venv/bin/python generate.py --download-only` de novo: ele continua de onde parou. |
| Mac lento ou usando swap em resoluções altas | Use 512×512 ou 768×768 e feche apps pesados. A memória sobe com a resolução. |
| `Largura e altura devem ser múltiplos de 32` | Use valores como 512, 544, 576, 768, 896, 1024. |

---

## 📂 Estrutura do projeto

- [`engine.py`](engine.py): motor MLX 4-bit com carregamento em 3 estágios (`Text Encoder` → libera memória → `DiT 7B Turbo, 8 passos` → libera memória → `VAE Decode`).
- [`generate.py`](generate.py): script de linha de comando (CLI).
- [`app.py`](app.py): interface web Gradio com presets de proporção e galeria local.
- [`requirements.txt`](requirements.txt): dependências com versões fixadas.
- `models/qwen-image-2.1-turbo-4bit/`: pesos quantizados em 4-bit (baixados no passo 5, fora do Git).
- `outputs/`: imagens `.png` geradas e relatórios `.run.json` (fora do Git).

---

## ⚖️ Licença do modelo

Os pesos do modelo **não** fazem parte deste repositório. Eles são uma conversão não oficial para MLX do [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) e seguem a **Qwen Research License**: uso para pesquisa/avaliação, **não comercial**, a menos que você obtenha uma licença comercial separada com o detentor dos direitos. O arquivo `LICENSE` e o `NOTICE` completos vêm junto com o modelo baixado em `models/qwen-image-2.1-turbo-4bit/`.

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
