# DTU dataset

## Description

As described on the [DTU homepage](https://roboimagedata.compute.dtu.dk/):

> The data set consist of 124 different scenes, where 80 of them have been used in the evaluation of the above-mentioned paper.
> The remaining 44 consist mainly of scenes that have been rotatated and scanned four times with 90 degree intervals, which enables 360 degree models. 
> A few have been removed from the evaluation due to low quality.

> The scenes include a wide range of objects in an effort to span the MVS problem.
> At the same time, the data set also include scenes with very similar objects, e.g. model houses, such that intra class variability can be explored.
> Each scene has been taken from 49 or 64 position, corresponding to the number of RGB images in each scene or scan. The image resolution is 1600 x 1200.
> The camera positions and internal camera parameters have been found with high accuracy, via the matlab calibration toolbox, which is also the toolbox you need to retrieve these parameters.
> Lastly, the scenes have been recorded in all 49 or 64 scens with seven different lighting conditions from directional to diffuse.


## Obtaining data

The pre-processed DTU [training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view), [test data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) and [depth maps](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) are available from the authors of [MVSNet](https://github.com/YoYo000/MVSNet).


## Data format

The training data consists of:
* Camera parameters stored in `txt` files
* Scene images stored as `png` files
* Depth maps stored as `pfm` files


### Images

An image is indexed by a _scan id_ (i.e. scene identifier), a _view id_ (i.e. viewing position identifier) which ranges from 1 to 49 or 1 to 64 depending on the number of viewpoints captured.
Finally, a _light index_ represents different lighting conditions for a given scan and view.  For training light conditions 0-6 are used, and for testing light index 3 is used (presumed by the authors of MVSNeRF to be the brightest).
The view ID is padded with zeros to three digits, so `14` becomes `014` and the corresponding file is stored at `Rectified/{scan_id}_train/rect_{view_id}_{light_index}_r5000.png`.


### Camera parameters

The camera parameters are stored in a `txt` file with the 4×4 extrinsic parameters on lines 2-5, the 3×3 intrinsic parameters on lines 8-10 and the near and far bounds for the camera frustrum on line 12.
Each file is indexed by a _view id_ which for our use is an index from 0 to 48 for 49 camera positions used in practice.  This identifier is padded to 8 digits, so `14` becomes `00000014`.
The corresponding file is located in `Cameras/train/{view_id}_cam.txt`.


### Depth maps

A depth map is indexed by a _scan id_ (i.e. scene identifier) and a _view id_ (i.e. viewing position identifier) which ranges from 0 to 48 or 0 to 63 depending on the number of viewpoints captured.
The view ID is padded with zeros to four digits, so `14` becomes `0014` and the corresponding file is stored at `Depths/{scan_id}/depth_map_{view_id}.pfm`.

The `pfm` file format used for depth maps is documented [here](https://www.pauldebevec.com/Research/HDR/PFM/).  The script [src/utils/pfm.py](../../utils/pfm.py) is used to parse these files. 


## Configuration format

### Train, test and validation splits

For train, test and validation stages a list of _scan ids_ (i.e. scene identifiers) identifies the scans to be used.  These are configured with text files `train.txt`, `validation.txt` and `test.txt` containing one scan id per line.


### Training image pairing

When training the MVSNeRF network, a set of MVS views are mapped to a single reference view.  For each possible reference viewpoint from 0 to 48 the possible source views are scored and sorted, and the top 10 selected.
The score is the view selection score, which is used to indicate the match quality between the two views. Details of this calculation are given in the [MVSNet paper](https://arxiv.org/abs/1804.02505).

This data is stored in a file `image_pairing.txt` in the configuration directory.  The first line of this file is the number of viewpoints; subsequent pairs of lines contain the reference view ID on the first line
followed by source view data on the next line. Each line of source view data is space separated data containing the number of views, followed by pairs indicating the source view ID and the score.


## Example file structure

An example file structure is available with the authors' [MVSNeRF implementation](https://1drv.ms/u/s!AjyDwSVHuwr8zhAAXh7x5We9czKj?e=oStQ48).
