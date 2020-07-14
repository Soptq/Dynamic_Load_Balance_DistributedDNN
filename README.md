# DBS

Official Pytorch implementation of "DBS: Dynamic Batch Size for Distributed Deep Neural Network Training".

## Quickstart
### Cloning
```
git clone https://github.com/soptq/Dynamic_Batch-Size_DistributedDNN
cd Dynamic_Batch-Size_DistributedDNN
```

### Installation
```
pip install -r requirements.txt
```

### Dataset Preparation
```
python prepare_data.py
```

### Run DBS
Here we run DBS with DenseNet-121 in 4 workers' distributed environment where worker 0-2 use GPU:0 and worker 3 use GPU:1 to simulate the unbalanced performance of different workers.

Additionally, the total batchsize of the entire cluster is set to 512, other arguments remain default.

```
python dbs.py -d false -ws 4 -b 512 -m densenet
```

Details of other arguments can be referred in `parser.py`

## Citation
```

```