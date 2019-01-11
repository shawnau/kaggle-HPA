## Human Protein Atlas Image Classification
75th solution for kaggle [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) by pytorch 1.0

requirements: 
 - pytorch 1.0
 - [yacs](https://github.com/rbgirshick/yacs)
 - python-opencv
 - [imgaug 0.2.7](https://github.com/aleju/imgaug/releases)

### Dataset Preprocessing

See `tools/preprocessing.py`.
1. `combine_dataset` for combining train with external data
2. `train_test_split` to split train and valid dataset while keeping class distribution using [Multilabel Stratification](https://github.com/trent-b/iterative-stratification)
3. `create_class_weight` to assign weights for each class for weighted BCE loss
4. `create_sample_weight` to assign each sample to balancing occurences of each class (linearly)
5. `calc_statistics`: calculate std and mean for datasets

### Train
using config-based system. to train model
```bash
python3 train_net.py --config-file config/res18_cv0.yaml
```
model will be dumped into `dump/res18_cv0` folder

### Test
```bash
python3 test_net.py --config-file config/res18_cv0.yaml
```

### Evaluation & summission

```bash
python3 evaluation.py --config-file config/res18_cv0.yaml
```

### Other Useful tools
1. TTA: edit [`build_tta_transforms`](https://github.com/shawnau/kaggle-HPA/blob/2f58e4b7a4739b29f74e988c4b554774fdff1cd4/dl_backbone/data/transforms/build.py#L74) to insert wanted tta and set [`TTA`](https://github.com/shawnau/kaggle-HPA/blob/2f58e4b7a4739b29f74e988c4b554774fdff1cd4/dl_backbone/config/defaults.py#L115) to 'on'
2. [macro f1 loss](https://github.com/shawnau/kaggle-HPA/blob/2f58e4b7a4739b29f74e988c4b554774fdff1cd4/dl_backbone/model/loss.py#L25)