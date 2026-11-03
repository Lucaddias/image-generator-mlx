# 🎨 Image Generator MLX — imagem e vídeo locais no Apple Silicon

[![testes](https://github.com/Lucaddias/image-generator-mlx/actions/workflows/testes.yml/badge.svg)](https://github.com/Lucaddias/image-generator-mlx/actions/workflows/testes.yml)

Fluxo local completo para geração de **imagens** com **Qwen-Image-2.1-Turbo (7B DiT)** e de **vídeos com áudio** com **LTX-2.3 (22B, destilado)**, otimizados para Macs Apple Silicon via **MLX 4-bit** com carregamento em estágios (*staged loading*). Tudo roda 100% offline depois do download dos modelos.

**Built with Qwen.**

## ⚡ Destaques

### 🖼️ Imagem (Qwen-Image-2.1-Turbo)
- **Tamanho em disco:** `~9.9 GB` (em vez dos 33 GB em BF16)
- **Pico de memória GPU (MLX):** `~4.7 GB` a 512×512, graças ao carregamento em 3 estágios
- **Passos de inferência:** `8 passos` (schedule nativo Turbo)
- **Texto na imagem:** escreve palavras dentro da imagem, basta colocá-las entre `"aspas"` no prompt

### 🎬 Vídeo (LTX-2.3)
- **Vídeo com áudio sincronizado**, a partir de texto ou animando uma imagem (inclusive as geradas pelo Qwen)
- **Tamanho em disco:** `~28.6 GB` (pipeline destilado 4-bit + codificador de texto Gemma 3 12B 4-bit)
- **Passos de inferência:** `8 + 3 passos` (meia resolução → upscale 2× → refino)
- **24 fps**, de 1 a 5 segundos por clipe

Tempos medidos num MacBook Pro **M5 (GPU de 10 núcleos), 24 GB**, uma geração por vez:

| Vídeo | Tempo total | Pico de memória GPU (MLX) | Pico do processo |
| --- | --- | --- | --- |
| 768×512, 5 s (121 quadros), texto → vídeo | 3 min 13 s | 12.6 GB | 17.6 GB |
| 1024×576, 5 s (121 quadros), texto → vídeo | 4 min 47 s | 11.7 GB | 17.7 GB |
| 768×512, 3 s (73 quadros), imagem → vídeo | 2 min 12 s | 11.8 GB | 18.4 GB |

### Para os dois
- **Duas formas de uso:** interface web (Gradio, com abas Imagem e Vídeo) e CLI (ótima para agentes como Claude Code, OpenCode e Codex)

---

## 📋 Requisitos

| Item | Mínimo |
| --- | --- |
| Computador | Mac com Apple Silicon (M1, M2, M3, M4, M5…). **Macs Intel não são suportados.** |
| Sistema | macOS 14 (Sonoma) ou mais recente |
| Memória | 16 GB de RAM unificada para imagem; **24 GB+ para vídeo** |
| Disco | ~12 GB livres só para imagem; **~41 GB** com vídeo (9.9 GB + 28.6 GB dos modelos + ~1.4 GB do ambiente Python) |
| Extra para vídeo | [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`) |
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

> A instalação baixa ~1.4 GB (PyTorch, MLX, Gradio, o [MFLUX](https://github.com/mflux-community/mflux) e o [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), os dois últimos direto do GitHub em versões fixadas).

### 5. Baixe o modelo de imagem (~9.9 GB)

```bash
.venv/bin/python generate.py --download-only
```

Os pesos são baixados do Hugging Face ([joeynyc/Studio-Image-Turbo-MLX-4bit-from-Qwen-2.1](https://huggingface.co/joeynyc/Studio-Image-Turbo-MLX-4bit-from-Qwen-2.1)) para `models/qwen-image-2.1-turbo-4bit/`. Esse passo é opcional: se você pular, o download acontece automaticamente na primeira geração.

### 6. Teste a instalação

```bash
.venv/bin/python generate.py --prompt 'A small red robot holding a white sign that says "HELLO", clean studio photograph' --seed 42
```

Se tudo estiver certo, a imagem aparece em `outputs/` junto com um relatório `.run.json` de tempo e memória.

### 7. (Opcional) Baixe o modelo de vídeo (~28.6 GB)

Só precisa se for gerar vídeos. Instale o ffmpeg e baixe os pesos:

```bash
brew install ffmpeg
```

```bash
.venv/bin/python generate_video.py --download-only
```

São baixados o LTX-2.3 4-bit ([dgrauet/ltx-2.3-mlx-q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4), só os ~20.5 GB que o pipeline destilado usa) para `models/ltx-2.3-mlx-q4/` e o codificador de texto Gemma 3 12B 4-bit ([mlx-community/gemma-3-12b-it-4bit](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit), ~8.1 GB) para `models/gemma-3-12b-it-4bit/`. Assim como na imagem, se pular este passo o download acontece no primeiro vídeo.

Teste com um clipe curto:

```bash
.venv/bin/python generate_video.py --prompt "Waves crashing on a rocky beach at sunset, seagulls calling" --seconds 2 --seed 42
```

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

Depois acesse **http://127.0.0.1:7860** no navegador. A interface tem duas abas:

- **🖼️ Imagem:** presets de proporção (quadrado, paisagem, retrato), seed, exemplos prontos e uma galeria com as imagens já geradas. O botão **"Animar esta imagem"** leva a imagem gerada para a aba Vídeo.
- **🎬 Vídeo:** prompt, imagem inicial opcional, resolução, duração (1 a 5 s), seed e o modo pouca memória. O vídeo sai em `.mp4` com áudio.

Imagem e vídeo usam a mesma GPU, então as gerações entram numa fila e rodam uma de cada vez.

> ⚠️ A fila só vale dentro da interface. Não gere vídeo pelo terminal enquanto a interface estiver gerando (nem o contrário): dois vídeos ao mesmo tempo passam dos 24 GB e o Mac começa a usar swap, deixando os dois muito mais lentos.

> Rode os comandos sempre de dentro da pasta do projeto: as imagens são salvas em `outputs/` relativo à pasta atual.

---

## 💻 Uso 2: Imagem pelo terminal

```bash
.venv/bin/python generate.py \
  --prompt 'A small red robot holding a white sign that says "HELLO M5", clean studio photograph, soft light' \
  --width 512 \
  --height 512 \
  --seed 42
```

### Opções do CLI de imagem

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

## 🎬 Uso 3: Vídeo pelo terminal

Texto para vídeo:

```bash
.venv/bin/python generate_video.py \
  --prompt "A cozy coffee shop on a rainy night, steam rising from a cup, rain tapping on the window, soft jazz" \
  --seconds 2 \
  --seed 7
```

Animar uma imagem (por exemplo, uma gerada pelo Qwen):

```bash
.venv/bin/python generate_video.py \
  --prompt "The camera slowly pushes in as neon lights flicker and rain falls" \
  --image outputs/qwen_20261027_160536.png \
  --seconds 2
```

### Opções do CLI de vídeo

| Opção | Descrição |
| --- | --- |
| `--prompt` / `-p` | Descrição da cena, do movimento de câmera e dos sons |
| `--image` / `-i` | Imagem inicial opcional (image-to-video) |
| `--width` / `-W` | Largura em pixels, múltiplo de 64 entre 256 e 1280 (padrão `768`) |
| `--height` / `-H` | Altura em pixels, múltiplo de 64 entre 256 e 1280 (padrão `512`) |
| `--seconds` | Duração aproximada em segundos a 24 fps (padrão `5`) |
| `--frames` / `-f` | Número exato de quadros na grade 8k+1 (`49`, `97`, `121`…); tem prioridade sobre `--seconds` |
| `--seed` / `-s` | Seed numérica (`-1` para aleatória, padrão) |
| `--output` / `-o` | Caminho do `.mp4` de saída (padrão `outputs/ltx_<timestamp>.mp4`) |
| `--low-ram` | Lê o transformer do disco aos poucos: menos memória, um pouco mais lento |
| `--model-path` / `--gemma-path` | Pastas onde os modelos de vídeo estão ou serão baixados |
| `--download-only` | Só baixa/verifica os modelos de vídeo e encerra |
| `--json` | Imprime apenas o JSON de resultado no stdout |

---

## 🤖 Uso 4: Claude Code, OpenCode ou Codex como "prompt enhancer"

Qualquer agente CLI pode transformar uma ideia curta em um prompt rico e chamar o motor local:

```bash
claude "Crie um prompt fotográfico detalhado em inglês para uma cafeteria futurista com placa neon escrita 'LUCA COFFEE' e execute .venv/bin/python generate.py --json -p '<SEU_PROMPT>'"
```

---

## 🔁 Uso 5: Pipeline automático (prompt → imagem → vídeo)

O [`pipeline.py`](pipeline.py) lê um lote em YAML e faz tudo sozinho: monta os prompts de imagem e de vídeo, gera as imagens, anima cada uma e marca os arquivos como gerados por IA. Ele retoma de onde parou e também pode observar uma pasta de fila para rodar sem intervenção.

```bash
.venv/bin/python pipeline.py preview jobs/exemplo.yaml
```

```bash
.venv/bin/python pipeline.py run jobs/exemplo.yaml
```

```bash
.venv/bin/python pipeline.py watch queue/
```

Trabalha só com personagens fictícios descritos em texto: não aceita foto de referência, exige idade declarada para cada personagem e só anima imagens que o próprio pipeline gerou. Além dos modelos locais, roda qualquer workflow do ComfyUI (por exemplo, Wan 2.2).

Guia completo: [`docs/pipeline-automatico.md`](docs/pipeline-automatico.md).

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
| Vídeo: `múltiplos de 64` | No vídeo a regra é múltiplo de 64: 320, 512, 576, 768, 1024. |
| Vídeo: Mac travando ou usando muito swap | Marque "Modo pouca memória" na interface (ou use `--low-ram`), reduza a resolução ou a duração e feche apps pesados. |
| Vídeo: `ffmpeg não encontrado` | Instale com `brew install ffmpeg`. |
| Vídeo: `O ffmpeg instalado não abre` / `Library not loaded: ...libx265...` | O Homebrew atualizou uma biblioteca e o ffmpeg ficou para trás. Rode `brew upgrade ffmpeg`. O gerador confere o ffmpeg antes de começar, para não perder a geração na hora de salvar. |

---

## 📂 Estrutura do projeto

- [`engine.py`](engine.py): motor de imagem MLX 4-bit com carregamento em 3 estágios (`Text Encoder` → libera memória → `DiT 7B Turbo, 8 passos` → libera memória → `VAE Decode`).
- [`video_engine.py`](video_engine.py): motor de vídeo LTX-2.3 com o pipeline destilado do ltx-2-mlx (`Gemma 3 12B` → libera memória → `DiT 22B, 8 + 3 passos` → libera memória → `VAE de vídeo + áudio`).
- [`generate.py`](generate.py): CLI de imagem.
- [`generate_video.py`](generate_video.py): CLI de vídeo.
- [`app.py`](app.py): interface web Gradio com abas Imagem e Vídeo, presets e galeria local.
- [`pipeline.py`](pipeline.py): pipeline automático em lote (prompts → imagens → vídeos), com retomada, fila e procedência.
- [`prompt_builder.py`](prompt_builder.py): leitura do lote YAML, personagens fictícios, variações automáticas e LLM local opcional.
- [`content_safety.py`](content_safety.py): validação de conteúdo dos prompts.
- [`backends.py`](backends.py): backends do pipeline: local (Qwen + LTX) e ComfyUI.
- [`jobs/exemplo.yaml`](jobs/exemplo.yaml): lote de exemplo comentado.
- [`docs/pipeline-automatico.md`](docs/pipeline-automatico.md): guia do pipeline.
- [`tests/`](tests/): testes das partes que não dependem de GPU (`python -m unittest discover -s tests -t .`).
- [`requirements.txt`](requirements.txt): dependências com versões fixadas.
- `models/qwen-image-2.1-turbo-4bit/`: pesos de imagem 4-bit (baixados no passo 5, fora do Git).
- `models/ltx-2.3-mlx-q4/` e `models/gemma-3-12b-it-4bit/`: pesos de vídeo 4-bit (baixados no passo 7, fora do Git).
- `outputs/`: imagens `.png`, vídeos `.mp4` e relatórios `.run.json` (fora do Git).

---

## ⚖️ Licenças

O código deste repositório está sob a licença [MIT](LICENSE). Os pesos dos modelos **não** fazem parte dele. Cada um tem a sua licença, que vem junto com os arquivos baixados em `models/`:

- **Imagem — Qwen-Image-2.1-Turbo:** conversão não oficial para MLX do [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), sob a **Qwen Research License**: uso para pesquisa/avaliação, **não comercial**, a menos que você obtenha uma licença comercial separada com o detentor dos direitos.
- **Vídeo — LTX-2.3:** conversão para MLX do LTX-2.3 da Lightricks, sob a **LTX-2 Community License Agreement**. O uso é livre para qualquer finalidade, mas entidades com receita anual de US$ 10 milhões ou mais precisam de uma licença comercial paga. Há também uma lista de usos proibidos (Anexo A), que inclui usar o modelo num produto ou serviço que concorra diretamente com os produtos comerciais da Lightricks sem uma licença separada. Leia o texto completo em `models/ltx-2.3-mlx-q4/LICENSE` antes de usar comercialmente.
- **Codificador de texto do vídeo — Gemma 3 12B:** sob os [Gemma Terms of Use](https://ai.google.dev/gemma/terms) do Google.

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
