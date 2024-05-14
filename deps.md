install conda pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tqdm seaborn pandas matplotlib numpy tomli msgpack msgpack-numpy portalocker h5py zstandard ipykernel ipympl torchCompactRadius scipy scikit-image scikit-learn numba
pip install scipy scikit-image scikit-learn
pip install numba



conda create -n "torch_22" python==3.11
conda activate torch_22

conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda install nvidia/label/cuda-12.1.0::cuda-cudart -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install tqdm seaborn pandas matplotlib numpy tomli msgpack msgpack-numpy portalocker h5py zstandard ipykernel ipympl torchCompactRadius scipy scikit-image scikit-learn numba
