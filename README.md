cudatoolkit 9.0
cudnn 7.6

` TensorFlow 1.12.0 + Keras 2.2.4 on Python 3.6.	floydhub/tensorflow `
https://docs.floydhub.com/guides/environments/

` tensorflow-1.12.0	2.7, 3.3-3.6	GCC 4.8	Bazel 0.15.0 `
https://www.tensorflow.org/install/source#tested_build_configurations

#### Required
https://github.com/NVIDIA/nvidia-docker
`docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi`

#### Build
`docker build . -t keremgocen/automl:latest --build-arg USE_PYTHON_3_NOT_2=True`

#### Run
`docker run -v $(pwd):/tf -it --rm -p 8888:8888 keremgocen/automl`

Use the token in terminal to login jupyter
```
$The Jupyter Notebook is running at:
$http://(f741b4528f3e or 127.0.0.1):8888/?token=188590c09964723af497d3900d3f4814195cc18e40f42243
```
