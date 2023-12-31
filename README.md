# SS HW1
#### Implemented by: Pistsov Georgiy 202

You can find report here: [wandb report](https://wandb.ai/goshanice/ss_project/reports/-DLA-SS-Homework--Vmlldzo1OTQ4MTQz?accessToken=724noxivesjdk0w1rkq4ad9e9pbeby2hsytbnerniy4277j3lpfkkal3asjhkkt7)

## Installation guide

Current repository is for Linux

(optional, not recommended) if you are trying to install it on macos run following before install:
```shell
make switch_to_macos
```

Then you run:

```shell
make install
```


## Download checkpoint:

```shell
make download_checkpoint
```
The file "model_best.pth" will be in default_test_model/

## Train model:

```shell
make train
```
Config for training you can find in src/config.json


## Test model:

### On test-clean:

```shell
make test_model
```

The file "output_test_clean.json" with results will be in the root on repository


## Run any other python script:

If you want to run any other custom python script, you can just start it with "poetry run"
For example:

Instead of:

```shell
python test.py -r default_test_model/model_best.pth
```

You can use:

```shell
poetry run python test.py -r default_test_model/model_best.pth
```

## How to train my model

```shell
poetry run python test.py -c src/configs/config_SpexPlusChangeMixture.json
```

Then you will need to resume from latest checkpoint

```shell
poetry run python test.py -c src/configs/config_SpexPlusChangeMixturePost.json -r PATH/TO/CHECKPOINT
```


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.