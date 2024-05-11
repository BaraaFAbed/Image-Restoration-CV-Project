
# Perceptual Image Enhancement for Smartphone Real-Time Applications - CV Project

This is a Computer Vision course project where we train the [LPIENet](https://arxiv.org/pdf/2210.13552) model on the [GoPro_Large](https://seungjunnah.github.io/Datasets/gopro.html) dataset.


## Authors

- [Baraa Abed](https://github.com/BaraaFAbed)
- [Ahmad Awwad](https://github.com/41686d6564)
- [Mohammed Adnan](https://github.com/Hamody23)
- [Yousef Mureb](https://github.com/yousefmureb)


## Installation

Use the `environment.yaml` file, or follow the these steps:
*  Create an enviroment with python=3.9 
*  Install PyTorch with Cuda using [this link](https://pytorch.org/get-started/locally/).
*  Run the following command to install the appropriate packages:

```bash
  pip install opencv-python matplotlib numpy jupyter tqdm

```

## Usage/Examples

Clone the project

```bash
  git clone https://github.com/BaraaFAbed/Image-Restoration-CV-Project.git
```

Go to the project directory

```bash
  cd Image-Restoration-CV-Project
```

Make sure to download the [GoPro Dataset](https://seungjunnah.github.io/Datasets/gopro.html) and extract it to the `GoPro/` directory. The directory tree should look like this:

```bash
GoPro
├───test
│   └─── ...
└───train
    └─── ...
```

### Training

Running train.py will start training the model. There are multiple arguments that can be used:

* `--lr LEARNING_RATE` (default = 1e-4)
* `--batch-size BATCH_SIZE` (default = 4)
* `--num-epochs NUMBER_OF_EPOCHS` (default = 500)
* `--seed SEED` (default = 42)

An example run could be:

```bash
  python train.py --lr 2e-4 --batch-size 16 --num-epochs 300 --seed 123
```

A checkpoint after every epoch of training as well as the checkpoint for the overall best model (`best.pth`) are saved in `Saves/Model_Saves`.  The loss convergence graph and the PSNR convergence graph are stored in `Saves/Graphs/` while the arrays for training loss, validation loss, and PSNR are stored in `Saves/Arrays`. 

### Testing

Running test.py will test the model using the weights from `best.pth` (Found in `Saves/Model_Saves`). There is only one argument: `--batch-size`. An example run could be:

```bash
  python test.py --batch-size 64
```

### Notebook

The `LPIENet.ipynb` notebook contains everything from training and testing, to visualizations of the results. If you would like to run any of that on a notebook, check it out.  


## Acknowledgements

* The model used was obtained from [here](https://github.com/mv-lab/AISP). All credits go to the authors. 
* Some code (such the functions in `utils.py`) were obtained from [this repository](https://github.com/yjn870/SRCNN-pytorch/) and then modified. Check out the original code if you are interested.  

## Contact

If you have any question, please feel free to contact us via `b00088000@aus.edu`.


