
<h1 align="center"> <p>DyG-Mamba</p></h1>
<h3 align="center">
    <p>"Continuous State Space Modeling on Dynamic Graphs" (DyG-Mamba)</p>
</h3>

![Overall Framework](./DyG-Mamba.pdf)


This is the code for "Continuous State Space Modeling on Dynamic Graphs" (`DyG-Mamba`).
DyG-Mamba is a new continuous state space model (SSM) for dynamic graph learning. Specifically, we first found that using inputs as control signals for SSM is not suitable for continuous-time dynamic network data with irregular sampling intervals, resulting in models being insensitive to time information and lacking generalization properties. We directly utilize irregular time spans as control signals for SSM to achieve significant robustness and generalization.

## Requirement

This code is modified from [DyGLib](https://github.com/yule-BUAA/DyGLib), and is tested in **Python 3.10.13**. The requirements are listed below:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm tabulate
pip install scipy
pip install scikit-learn
pip install mamba-ssm
```


## Benchmark and Preprocessing
Twelve datasets are used in DyG-Mamba, including Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Can. Parl., 
US Legis., UN Trade, UN Vote, and Contact. The first five datasets are bipartite, and the others only contain nodes with a single type.

Most of the used original dynamic graph datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg), 
which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o). 
Please download them and put them in ```DG_data``` folder. 

We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```


## Run the code

DyG-Mamba supports dynamic link prediction under both transductive and inductive settings with three (i.e., random, historical, and inductive) negative sampling strategies, as well as dynamic node classification.

### Launch dynamic link prediction training
```shell
python train_link_prediction.py --dataset_name ${dataset_name} --model_name DyG-Mamba --gpu ${gpu}
```

`${dataset_name}` is the dataset name, `${gpu}` is the lanched GPU ID

### Launch dynamic link prediction evaluation
```shell
python evaluate_link_prediction.py --dataset_name ${dataset_name} --model_name DyG-Mamba --negative_sample_strategy ${negative_sample_strategy} --gpu ${gpu}
```
`${negative_sample_strategy}` is selecttived from three negative sampling strategy (*random*, *historical*, *inductive*).


### Lauch dynamic node classification training
Dynamic node classification could be performed on Wikipedia and Reddit (the only two datasets with dynamic labels).
```shell
python train_node_classification.py --dataset_name ${dataset_name} --model_name DyG-Mamba --gpu ${gpu}
```

### Lauch dynamic node classification evaluation
```shell
python evaluate_node_classification.py --dataset_name ${dataset_name} --model_name DyG-Mamba --gpu ${gpu}
```


### Launch the robustness of the pretrained encoder
We randomly add `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]` noises to sampled neighbors in evaluation to test the robustness.

```shell
python evaluate_link_prediction_robustness_test --dataset_name ${dataset_name} --model_name DyG-Mamba --gpu ${gpu}
```

## Acknowledgments
We are grateful to the authors of [DyGLib](https://github.com/yule-BUAA/DyGLib) for making a fintastic archtecture codes publicly available.
