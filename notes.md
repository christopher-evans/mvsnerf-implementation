

# setup environment

➜  mkdir venv-mvsnerf               
➜  python3 -m venv venv-mvsnerf 
➜  source venv-mvsnerf/bin/activate
➜  which python
/home/r2tp/Repos/venv-mvsnerf/bin/python
➜  python -V
Python 3.11.8

# setup pytorch and cuda

pip install torch torchvision torchaudio
python -m pip install lightning
pip install tensorboard


or use
pip install -r requirements. txt 

# tensorboard

tensorboard --logdir ./.experiments


PyTest : https://stackoverflow.com/questions/19672138/how-do-i-mock-the-filesystem-in-python-unit-tests

export PYTHONPATH=/home/r2tp/Repos/mvsnerf-implementation/src:$PYTHONPATH


For docs: export PYTHONPATH=/home/r2tp/Repos/mvsnerf-implementation:$PYTHONPATH 
Build docs: sphinx-build -M html docs/source/ docs/build/
TODO: use sphinx argparse
TODO: deploy docs with github actions
TODO: write docs
TODO: upload code coverage to codeclimate

Argparse: https://realpython.com/command-line-interfaces-python-argparse/

## Parameters

- select bottom viewpoints instead of top during training
- down sample
- scale factor
- cropping / resizing, could jitter
- feature extraction net -- trade off with down sample
- positional encoding parameters
- colume encoding net params
- lighting conditions
- types of supervision : depth / image / ...
- optimizer params
- padding on interpolation
- depth resolution
- rendering sample at random
- rendering sample points
- rendering no. rays
- rendering batch size
- rendering use depth data for better sampling
- rendering use importance sampling
- rendering re-use depth candidates or not
- add reconstruction cost for source images (regularization ? )
- jitter the offsets for ray marching to sub-pixels and interpolate


- use resnet and more layers in feature extraction
- increase depth and neurons in MLP
- float16/32
- bigger 3d resnet?
- add the masks as inputs to MLP (as in original implementation)
- increase depth resolution