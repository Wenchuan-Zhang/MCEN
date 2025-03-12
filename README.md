# MCEN Deep Learning Algorithm(Updating)

Based on the Mamba framework, the MCEN algorithm is a cutting-edge deep learning approach designed for efficient and scalable model training and inference. 

## TODO

- [x] Add the code for the preprocessing of the datasets.
- [x] Add the code for the training of the model.
- [ ] Add the code for the testing of the model.
- [ ] Improving README document.
- [ ] Improving the code structure.

## Preprocessing
- We adjusted the preprocessing steps in the [CLAM](https://github.com/mahmoodlab/CLAM) repository.
- CONCH model and weight can be found in [this link](https://github.com/mahmoodlab/CONCH).

## Training
Train the model:
```bash
python main.py --project=conch_mcen --datasets=mydataset --dataset_root=./h5_feature/conch_feature --model_path=./result --cv_fold=5 \
--model=MCEN --pool=attn --n_trans_layers=2 --da_act=tanh --title=conch_MCEN \
--epeg_k=9 --crmsa_k=3 --all_shortcut --input_dim=512 --seed=2024 --label_path ./label.csv --only_rrt_enc --config ./config/conch_mcen.yml
```
