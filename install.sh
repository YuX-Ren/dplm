env_name=dplm

conda create -n ${env_name} python=3.10 pip
conda activate ${env_name}

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install esm

pip install -e .

pip install -e vendor/openfold

