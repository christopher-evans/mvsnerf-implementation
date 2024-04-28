# MVSNeRF implementation

This project re-implements the MVSNeRF model ([paper](https://arxiv.org/abs/2103.15595) and [code](https://github.com/apchenstu/mvsnerf)).

This is both for the understanding of the author, and to update to [PyTorch Lightning 2](https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html) API.


## Python

Target  Python versions are 3.10 and 3.11.  With one of these installed, set up a virtual environment:

```bash
➜  mkdir venv-src               
➜  python -m venv venv-src 
➜  source venv-src/bin/activate
➜  which python
/home/r2tp/Repos/venv-src/bin/python
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

## Datasets

### DTU

See [src/datasets/dtu/README.md](src/datasets/dtu/README.md) for documentation.


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
"""
   Send a message to a recipient.

   :param str sender: The person sending the message
   :param str recipient: The recipient of the message
   :param str message_body: The body of the message
   :param priority: The priority of the message, can be a number 1-5
   :type priority: integer or None
   :return: the message id
   :rtype: int
   :raises ValueError: if the message_body exceeds 160 characters
   :raises TypeError: if the message_body is not a basestring
   """

## Tensorboard

To display tensorboard logs:

```bash
tensorboard --logdir .experiments
```

This command should provide a local URL for viewing the run details.

