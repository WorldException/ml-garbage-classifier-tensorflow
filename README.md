## My TensorFlow Sandbox

I'm going through some TensorFlow tutorials here and just exploring the framework. There's nothing particularly interesting to see here.

#### Setup

Virtualenv:
```
virtualenv venv
source /venv/bin/activate
```

Dependencies:
```
pip install matplotlib
pip install Pillow
```

CPU-only dependencies:
```
pip install tensorflow
```

For GPU, installation instructions are [here](https://www.tensorflow.org/install/install_linux) and [here](https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/tensorflow/).
Note that for cuDNN, its a .dpkg file instead of a tarball.

Unzip Trash Dataset:

```
unzip src/trash_data/dataset.zip
```

Get CIFAR-10 Dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).


#### Reference
[Pillow Docs](http://pillow.readthedocs.io/en/3.4.x/reference/Image.html)

#### Blender

I'm trying to use Blender to generate new training data. I'm using predef files for editing Blender scripts in an external editor, as described [here](http://jameskersey.com/2013/09/11/python_editing_for_blender_part_one).

#### Acknowledgements:
Trash Dataset: https://github.com/garythung/trashnet