# DLB

Official Pytorch implementation of "DLB: A Dynamic Load Balance Strategy For Robust Distributed Deep Neural Network Training".

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
python dbs.py -d false -ws 4 -b 512 -m densenet -gpu 0,0,0,1
```

Details of other arguments can be referred in `parser.py`

## Citation
```
@inproceedings{Ye2020DBSDB,
  title={DBS: Dynamic Batch Size For Distributed Deep Neural Network Training},
  author={Qing Ye and Yuhao Zhou and Mingjia Shi and Yanan Sun and Jiancheng Lv},
  year={2020}
}
```
