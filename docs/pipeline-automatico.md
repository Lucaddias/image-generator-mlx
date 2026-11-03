# 🔁 Pipeline automático: prompt → imagem → vídeo

O `pipeline.py` lê um lote em YAML e faz tudo sozinho: monta os prompts de imagem e de vídeo, gera as imagens, anima cada uma e marca os arquivos como gerados por IA. Se for interrompido, retoma de onde parou.

```
lote.yaml ──► [1] prompts ──► [2] imagens ──► [3] vídeos ──► outputs/pipeline/<lote>/
              template +      Qwen-Image      LTX-2.3          manifest.json
              LLM opcional    ou ComfyUI      ou ComfyUI       *.image.png / *.video.mp4
              └───────────────── validação em todas as etapas ─────────────────┘
```

Cada etapa roda para o lote inteiro antes da próxima. Assim cada modelo carrega uma vez e a memória do Mac fica livre entre um estágio e outro.

---

## ✅ Regras de conteúdo

| Regra | Como é aplicada |
| --- | --- |
| Personagens fictícios, descritos em texto | O fluxo começa sempre em texto, sem entrada de foto de referência. Termos de semelhança com pessoas específicas e os nomes da sua lista `blocked_names.txt` bloqueiam o job. |
| O vídeo só anima imagens do próprio pipeline | O SHA-256 de cada imagem fica no manifesto. Se o arquivo for trocado, o vídeo é recusado. |
| Idade declarada | Todo personagem declara `age` (mínimo **21**), e a idade vai escrita no prompt (`an adult woman ..., 29 years old`). |
| O LLM também é validado | Se o LLM local tirar a idade do prompt ou incluir algum termo bloqueado, o pipeline descarta a reescrita e volta para o template. |
| Procedência | O PNG ganha os metadados `AI-Generated`, `AI-Notice` e `AI-Provenance`, e o MP4 ganha os campos `comment`/`description`. Cada arquivo também tem um `.provenance.json` ao lado. |

A validação é heurística e prefere falso positivo a deixar passar algo ambíguo. A mensagem de erro sempre diz qual termo bloqueou o job.

---

## 🚀 Como usar

### 1. Escreva o lote

Copie `jobs/exemplo.yaml` e edite. Os arquivos em `jobs/` (fora o exemplo) e a pasta `queue/` ficam fora do Git.

| Campo do job | Vai para | Exemplo |
| --- | --- | --- |
| `subjects` | imagem e vídeo | `[lara]` (do cadastro `characters`) ou `[{description: ..., age: 30}]` |
| `style` | imagem e vídeo | `photo`, `cinematic`, `studio`, `amateur` |
| `wardrobe`, `action`, `scene`, `lighting`, `extra` | imagem (e o vídeo reaproveita cena e luz) | texto em inglês |
| `motion`, `camera`, `sound` | vídeo | `she turns toward the camera`, `slow dolly in`, `rain on the window` |
| `seed` | os dois | omitido = seed fixa derivada do `id` |

- `characters:` é o cadastro de personagens fictícios. O nome é só uma etiqueta interna e nunca entra no prompt.
- `defaults:` vale para todos os jobs.
- `generate:` cria jobs automaticamente, sorteando uma opção de cada lista em `pick:`. O sorteio é fixo por índice, então aumentar o `count` mantém os jobs já gerados.

### 2. Confira sem gastar GPU

```bash
.venv/bin/python pipeline.py preview jobs/meu_lote.yaml
```

Mostra os prompts finais e os jobs recusados, com o motivo. Sai com código 1 se algum job foi recusado.

### 3. Rode

```bash
.venv/bin/python pipeline.py run jobs/meu_lote.yaml
```

Para fazer curadoria antes dos vídeos, que são a parte cara (cerca de 3 min por clipe de 5 s num M5):

```bash
.venv/bin/python pipeline.py run jobs/meu_lote.yaml --stop-after images
```

Apague as imagens de que não gostou em `outputs/pipeline/meu_lote/` e rode o `run` de novo: os jobs apagados viram `removed` e não ganham vídeo.

| Opção | Efeito |
| --- | --- |
| `--stop-after prompts\|images` | Para depois da etapa indicada |
| `--redo ID [ID...]` | Refaz esses jobs do zero |
| `--force` | Refaz o lote inteiro |
| `--json` | Resumo numa linha JSON (para agentes CLI) |

Rodar o mesmo lote de novo **retoma o trabalho**: os jobs com erro ganham nova tentativa, e só é refeito o que mudou no YAML. Se você mudar só `motion`, `camera`, `sound` ou as opções de vídeo, a imagem é mantida e só o vídeo é refeito.

### 4. Deixe automático (pasta de fila)

```bash
.venv/bin/python pipeline.py watch queue/
```

Cada `.yaml` que cair em `queue/` é processado e movido para `queue/done/` ou `queue/failed/`, com um `.result.json` ao lado. Um lote por vez: uma trava em `outputs/pipeline/.gpu.lock` impede dois pipelines ao mesmo tempo. Não rode o `app.py` gerando junto, porque dois vídeos simultâneos passam dos 24 GB.

Para iniciar sozinho quando você entra no Mac, crie `~/Library/LaunchAgents/com.local.pipeline-watch.plist` (troque o caminho pelo da sua pasta):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.local.pipeline-watch</string>
  <key>WorkingDirectory</key><string>/Users/SEU_USUARIO/image-generator-mlx</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/SEU_USUARIO/image-generator-mlx/.venv/bin/python</string>
    <string>pipeline.py</string>
    <string>watch</string>
    <string>queue</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/Users/SEU_USUARIO/image-generator-mlx/outputs/pipeline/watch.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.local.pipeline-watch.plist
```

---

## 🧠 LLM local opcional (prompts mais ricos)

Sem LLM, os prompts saem do template: previsíveis, rápidos e fáceis de auditar. Com um bloco `llm:` no lote, um modelo local rodando via `mlx-lm` (que já vem no `requirements.txt`) reescreve cada rascunho:

```yaml
llm:
  model: models/seu-llm-mlx   # pasta local ou repo do Hugging Face em formato MLX
  max_tokens: 300
  temperature: 0.7
```

- Use um modelo *instruct* **sem modo de raciocínio**. Se ele "pensar" demais, os tokens acabam antes do prompt sair e o pipeline volta para o template.
- Confira se a licença do LLM permite o seu uso.
- O LLM carrega uma vez por lote e é liberado antes das imagens. Todas as reescritas passam pela validação.

---

## 🖥️ Backend ComfyUI (Wan 2.2 e outros modelos)

O backend `comfyui` roda qualquer workflow do ComfyUI, local ou numa GPU alugada. O **Wan 2.2** (licença Apache 2.0) é uma boa escolha: faz imagem→vídeo e também texto→imagem (gerando 1 quadro).

1. Monte e teste no ComfyUI um workflow de texto→imagem e outro de imagem→vídeo (por exemplo, Wan 2.2 I2V). O de vídeo precisa de um nó `LoadImage`.
2. Exporte cada um em **API format**: no menu *Workflow → Export (API)*. Nas versões antigas, ative o *Dev mode* nas configurações e use *Save (API Format)*. O arquivo da interface comum, que tem `nodes`/`links`, é recusado.
3. Salve os arquivos em `workflows/` e abra o JSON para ver os ids dos nós: cada chave do topo (`"3"`, `"6"`...) é um nó, e `inputs` traz as entradas dele.
4. No lote, aponte cada valor para `[id do nó, nome da entrada]`:

```yaml
backend:
  type: comfyui
  url: http://127.0.0.1:8188
  image:
    workflow: ../workflows/texto_para_imagem_api.json
    inputs: {prompt: ["6", "text"], seed: ["3", "seed"], width: ["5", "width"], height: ["5", "height"]}
  video:
    workflow: ../workflows/imagem_para_video_api.json
    fps: 16             # Wan 2.2 14B; o 5B é 24
    frame_multiple: 4   # Wan: quadros 4k+1
    inputs: {prompt: ["6", "text"], image: ["52", "image"], seed: ["3", "seed"], frames: ["50", "length"]}
```

Só `prompt` é obrigatório (e `image` no vídeo). Valores sem mapeamento ficam como estão no workflow.

**GPU alugada:** o ComfyUI não tem autenticação. Não exponha a porta 8188 na internet: use um túnel SSH e mantenha `url: http://127.0.0.1:8188`.

```bash
ssh -N -L 8188:127.0.0.1:8188 usuario@endereco-do-pod
```

---

## 📂 Saídas

```
outputs/pipeline/<lote>/
├── manifest.json                    # estado de cada job (prompts, seeds, hashes, tempos, erros)
├── <id>.image.png                   # imagem com metadados de IA
├── <id>.image.png.provenance.json
├── <id>.video.mp4                   # vídeo com metadados de IA
└── <id>.video.mp4.provenance.json
```

Estados no manifesto: `pending`, `prompted`, `imaged`, `done`, `removed` (você apagou o arquivo) e `failed` (o erro fica registrado e o job ganha nova tentativa na próxima execução). Jobs recusados pela validação aparecem em `rejected`.

---

## 🧪 Testes

As partes que não dependem de GPU (validação, montagem de prompts, manifesto, retomada, trava de GPU e helpers do ComfyUI) têm testes que rodam em qualquer máquina:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```
