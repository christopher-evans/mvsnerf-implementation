# MVSNeRF implementation

This project re-implements the MVSNeRF model ([paper](https://arxiv.org/abs/2103.15595) and [code](https://github.com/apchenstu/mvsnerf)).

This is both for the understanding of the author, and to update to [PyTorch Lightning 2](https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html) API.


## Python

Target  Python versions are 3.10 and 3.11.  With one of these installed, set up a virtual environment:

```bash
➜  mkdir venv-mvsnerf               
➜  python -m venv venv-mvsnerf 
➜  source venv-mvsnerf/bin/activate
➜  which python
/home/r2tp/Repos/venv-mvsnerf/bin/python
➜  python -V
Python 3.11.8
```

Then install the project dependencies:

```bash
➜  pip install -r requirements.txt
```


## Project structure

The source code is organised as follows:
* `src/data`: scripts for data processing
* `src/datasets`: torch `Dataset` wrappers
* `src/engines`: lightning modules configuring train, test and validation
* `src/models`: torch models
* `src/scripts`: scripts for running train, test, validation and inference
* `src/utils`: helper functions
* `test`: unit tests

Additional, configurable directories are:
* `.data`: location for datasets
* `.configs`: configuration files, used for selection of train/test/validation splits
* `.experiments`: location for tensorboard logs and checkpoints


## Tensorboard

To display tensorboard logs:

```bash
tensorboard --logdir .experiments
```

This command should provide a local URL for viewing the run details.

