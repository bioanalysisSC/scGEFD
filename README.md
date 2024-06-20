#### Install requirements


In order to perform clustering calculations more quickly, we employ clustering using pure PyTorch, cited by "Learning Representation for Clustering via Prototype Scattering and Positive Sampling", and the code can be found at https://github.com/Hzzone/torch_clustering.

We need first clone it:

```shell
git clone --depth 1 https://github.com/Hzzone/torch_clustering tmp && mv tmp/torch_clustering . && rm -rf tmp
```

and then install other requirements:

```shell
pip install -r requirements.txt
```

#### Training Commands
The config files are in `config/`, just run the following command:
```shell
python main.py --dataname Quake_Smart-seq2_Diaphragm
```