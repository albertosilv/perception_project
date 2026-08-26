# Perception Project — OpenCV + dlib

Esqueleto base de um pipeline de percepção computacional (detecção facial +
landmarks), estruturado para você adicionar novas features (emoção, gaze,
reconhecimento, pose, etc) sem precisar reorganizar o projeto do zero.

## Estrutura

```
perception_project/
├── main.py                        # ponto de entrada (loop de captura + display)
├── requirements.txt
├── config/
│   └── config.yaml                # TODA configuração fica aqui, fora do código
├── models/                        # modelos do dlib (.dat) ficam aqui
├── scripts/
│   └── download_models.py         # baixa o shape_predictor de 68 pontos
├── src/
│   ├── core/
│   │   ├── camera.py              # abstração da fonte de vídeo
│   │   ├── config_loader.py       # carrega o YAML como objeto navegável
│   │   └── logger.py              # logger padronizado
│   ├── perception/
│   │   ├── base_detector.py       # interface que todo detector deve seguir
│   │   ├── types.py               # Face, BoundingBox, FrameResult
│   │   ├── face_detector.py       # backend Haar ou dlib HOG
│   │   ├── landmark_detector.py   # 68 pontos faciais (dlib)
│   │   └── pipeline.py            # orquestra os detectores em sequência
│   └── utils/
│       └── visualization.py       # desenha bbox/landmarks/FPS no frame
├── tests/
│   └── test_pipeline.py
└── data/output/                   # logs, capturas, resultados
```

## Instalação

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **dlib pode dar trabalho para compilar no Windows.** Se `pip install dlib`
> falhar, instale o CMake e um compilador C++ (Visual Studio Build Tools),
> ou use `conda install -c conda-forge dlib`.

Baixe o modelo de landmarks (68 pontos):

```bash
python scripts/download_models.py
```

Se for usar o backend `haar` (mais leve, não requer dlib para detecção de
rosto — só para landmarks), copie o arquivo `haarcascade_frontalface_default.xml`
que já vem instalado junto com o OpenCV para `models/`, ou aponte
`face_detection.haar_cascade_path` no config para o caminho instalado:

```python
import cv2
print(cv2.data.haarcascades)
```

## Uso

```bash
python main.py
```

Pressione **`q`** para encerrar.

Para usar outro arquivo de configuração:

```bash
python main.py --config config/outro_config.yaml
```

## Configuração (`config/config.yaml`)

Tudo que é "parâmetro" fica no YAML, não no código:

- `camera`: fonte de vídeo, resolução, espelhamento
- `face_detection.backend`: `"haar"` ou `"dlib_hog"`
- `landmarks.enabled`: liga/desliga a extração dos 68 pontos
- `display`: o que aparece na tela (bbox, FPS, etc)
- `logging`: nível de log e se salva em arquivo

## Como adicionar uma nova feature

O projeto foi pensado para isso ser o caminho mais curto possível.
Exemplo: adicionar um detector de emoção.

**1. Crie o detector** em `src/perception/emotion_detector.py`, herdando de
`BaseDetector`:

```python
from src.perception.base_detector import BaseDetector

class EmotionDetector(BaseDetector):
    def __init__(self, cfg):
        # carregue seu modelo aqui
        ...

    def detect(self, frame, faces):
        for face in faces:
            # recorte o rosto usando face.bbox, rode seu modelo,
            # e salve o resultado em face.extra
            face.extra["emotion"] = "feliz"
        return faces
```

**2. Registre no pipeline** (`src/perception/pipeline.py`):

```python
# no __init__:
self.emotion_detector = EmotionDetector(cfg.emotion)

# no process():
faces = self.emotion_detector.detect(frame, faces)
```

**3. Adicione a config** correspondente em `config/config.yaml`:

```yaml
emotion:
  model_path: "models/emotion_model.h5"
```

**4. (Opcional) Desenhe o resultado** em `src/utils/visualization.py`,
lendo `face.extra["emotion"]`.

Pronto — sem tocar em `main.py`, `camera.py` ou `face_detector.py`.

## Ideias de próximas features

- Reconhecimento facial (`face_recognition` / embeddings com dlib)
- Rastreamento entre frames (ex: `dlib.correlation_tracker` ou SORT/DeepSORT)
- Estimativa de pose da cabeça (solvePnP com os landmarks)
- Detecção de piscar de olhos (Eye Aspect Ratio a partir dos landmarks)
- Detecção de emoção (modelo CNN treinado em FER2013)
- Contagem/heatmap de pessoas na cena

## Testes

```bash
pytest tests/
```
