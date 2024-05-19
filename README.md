# MVSNeRF implementation

This project re-implements the MVSNeRF model ([paper](https://arxiv.org/abs/2103.15595) and [code](https://github.com/apchenstu/mvsnerf)).

The aims are to:
* Document and provide tools to process training data, run experiments and evaluate the model
* Increase the range of parameters and input augmentations which can be adjusted on the model
* Support batch sizes greater than one during training
* Remove dependency on depth data for training images
* Update to the [PyTorch Lightning 2](https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html) API.
* Improve test coverage of the codebase and improve code quality


## Status

This repository is in the early stages of development; features will be listed here as they are completed.
[Pull requests](https://github.com/christopher-evans/mvsnerf-implementation/pulls) are welcome!

## Python

Target  Python versions are 3.10, 3.11 and 3.12.  With one of these installed, set up a virtual environment:

```bash
➜  mkdir venv-src               
➜  python -m venv venv-src 
➜  source venv-src/bin/activate
➜  which python
/path/to/venv-src/bin/python
➜  python -V
Python 3.11.8
```

Then install the project dependencies:

```bash
➜  pip install -r requirements.txt
```

Finally, install inplace_abn manually:
```bash
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python3 setup.py install
```


## Running

Ensure the venv is loaded and the script is executable:
```bash
➜  which python
/path/to/venv-src/bin/python
➜  chmod +x mvsnerf.sh
```

Run the bash script `mvsnerf.sh` with one of `train`, `validate`, `fine_tune` or `infer` as the first argument.
Running with `-h` flag shows all required and available arguments.
```bash
./mvsnerf.sh train -h
```

## Datasets

### DTU

See [src/datasets/dtu](src/datasets/dtu) for documentation.


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

## Code quality

### Tests

Tests use [pytest](https://docs.pytest.org/en/8.2.x/) and are configured with the `pytest.ini` file.  To run
tests:
```bash
pytest
```
from the source directory. To generate a coverage report, run
```bash
pytest --cov-report term --cov=src test/
```

Tests are located in the `test` directory, with file structure corresponding to the `src` directory.

### Linting

Linting uses [pylint](https://pypi.org/project/pylint/) and is configured by the `.pylintrc` file.
To lint the source files, run:
```bash
pylint ./src
```
from the repository root.

Similarly for the test directory:
```bash
pylint ./test
```

## Tensorboard

To display tensorboard logs:

```bash
tensorboard --logdir .experiments
```

This command should provide a local URL for viewing the run details.

