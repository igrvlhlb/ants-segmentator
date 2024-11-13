# Ants Segmentator

Como rodar:

```sh
./generate-inputs.sh path/to/img/image.jpg
```

cria em `path/to/img/` os arquivos `image-grayscale.bmp` e `image-sobel-thin.bmp`. Esses arquivos são a imagem em escala de cinza e a imagem delineada, respectivamente.

Para rodar o algoritmo, basta rodar

```sh
uv run ants.py path/to/img/image-grayscale.bmp path/to/img/image-sobel-thin.bmp [-o output/dir/] [algorithm args]
```

ou

```sh
poetry run python3 ants.py path/to/img/image-grayscale.bmp path/to/img/image-sobel-thin.bmp [-o output/dir/] [algorithm args]
```

Onde `output-dir` é o diretório onde os arquivos de saída serão salvos e `algorithm args` são os argumentos específicos do algoritmo escolhido.

Para gerar uma imagem comparativa entre a imagem original (com endpoints destacados) e a imagem delineada, basta rodar

```sh
sh merge.sh output/dir/
```
