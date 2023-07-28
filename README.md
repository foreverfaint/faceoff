# faceoff

Build a face swap demo based on [insightface](https://github.com/deepinsight/insightface), [gfpgan](https://github.com/TencentARC/GFPGAN) and [streamlit](https://github.com/streamlit/streamlit).

This demo is tested on Ubuntu 22.04.2 with python 3.10.12

## 

## Model Preparation

This demo depends on multiple models

|Model Name|Description|Location|
|:-----|:----|:----|
|[inswapper_128.onnx](https://huggingface.co/deepinsight/inswapper/tree/main)|insightface face swapper model|https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx|
|[GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4)|GFPGAN face restore models|https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth|
|detection_Resnet50_Final.pth|GFPGAN uses this model for face detection internally|Downloaded by GFPGAN at runtime|
|parsing_parsenet.pth|GFPGAN uses this model internally|Downloaded by GFPGAN at runtime|
|[buffalo_l.zip](https://github.com/deepinsight/insightface/releases)|Face analysis|https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip|

Please:

- Download and save `inswapper_128.onnx` and `GFPGANv1.4.pth` to `./models` folder which is referred in the demo.
- `detection_Resnet50_Final.pth` and `parsing_parsenet.pth` will be automatically downloaded to `./gfpgan/weights` folder when you run the demo.
- `buffalo_l.zip` will be automatically downloaded to `~/.insightface/models` folder when you run the demo.
 
```bash
$ mkdir ./models

$ cd ./models

# download inswapper_128.onnx and GFPGANv1.4.pth
$ wget https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
$ wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

## Run it locally

We use [pyenv](https://github.com/pyenv/pyenv) + [poetry](https://python-poetry.org/) to create an virtual environment to run the demo. 

> Assume pyenv and poetry have been installed on your machine.

```bash
# Before you use pyenv to install python 3.10, you must ensure some dependences installed on your ubuntu
$ sudo apt update && sudo apt install lzma liblzma-dev libbz2-dev

# If you see any warning like 'WARNING: The Python bz2 extension was not compiled. Missing the bzip2 lib?', please apt install the related dependences and try again.
$ pyenv install 3.10

# set your faceoff folder with python 3.10
$ cd /path/to/faceoff_folder
$ pyenv local 3.10

# poetry will use python 3.10 to setup your virtual environment
$ poetry install 

# Start it
$ poetry run streamlit run app.py
```

If you want remove this environment, you should do:

```bash
# Find your env name
$ poetry env list
faceoff-LR4bX88f-py3.10

# Delete it
$ poetry env remove faceoff-LR4bX88f-py3.10
```

## Run it in Docker

You can also build everything into a docker image and run it everywhere. Before you build the image, you need ensure all the models ready locally because we will copy the models into the image for app launch acceleration. You can see `copy to docker` instruction in `dev.dockerfile` like

```dockerfile
COPY ./models/buffalo_l /root/.insightface/models/buffalo_l
COPY ./models/*.pth ./models
COPY ./models/*.onnx ./models
COPY ./gfpgan /workdir/gfpgan
```

So please ensure all the models at the positions. Especially:

- Run the demo locally once to trigger downloading GFPGAN and insightface face analyzer models to your local folder.
- Copy `~/.insightface/models` (insightface downloads its models here originally) to `./models`.

```bash
# Build it!
$ chmod +x ./scripts/build/dev.sh
$ ./scripts/build/dev.sh

# Run it! 
$ ./scripts/run/dev.sh
```
