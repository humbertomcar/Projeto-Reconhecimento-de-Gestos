# Como Rodar

## Docker
Primeiramente, é construir o container do Docker

`build docker -t dockerfile .`

Em seguida, vamos abrir o programa no container criado:


## **⚠️ Importante**

1. **Webcam**

- Para a execução adequada do programa, é necessário uma Webcam conectada ao dispositivo.
- Quando não há Webcams disponíveis, o Docker vai retornar um erro
- Caso a Webcam esteja conectada, mas não está sendo reconhecida, verificar o caminho da Webcam na linha `--device=/dev/video0` do arquivo devcontainer.json

2. **Python e Dependências**

- As bibliotecas necessárias estão disponíveis no requirements.txt
- É necessário que a versão do python seja entre 3.07 até 3.10 para utilizar a biblioteca mediapipe