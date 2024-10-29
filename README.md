# IEFENet


### The directory where the yaml files for the four innovations of the thesis are located:

###### models/innovations/..





#### The paper's four innovation point codes are listed in the table of contents:

###### models/innovations.py



###### Install

install  requirements.txt in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```
cd IEFENet
pip install -r requirements.txt  # install
```

###### set up WUDD.yaml

The data set configuration file of data/WUDD.yaml defines 1) train/val/test image directory of root directory and relative path of data set. 2) Category and category name. 

The dataset can be accessed at the following URL:  https://doi.org/10.6084/m9.figshare.27178071.v1.

```python
path: ../datasets/WUDD
train: images/train  
val: images/val  
test: images/test               
nc: 3  
names: ['scallop','seacucumber', 'seaurchin']  
```

###### Train

The IEFENet model is trained on WUDD by specifying data set, batch size,  image size and pre-training weights. The batch size is set to 32 and the  image size is 640×640, and the pre-training weight is not used.

```python
parser.add_argument('--weights', type=str, default='', help='initial weights path')
parser.add_argument('--cfg', type=str, default='models/innovations/xxx.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default=ROOT / 'data/WUDD.yaml', help='dataset.yaml path')
parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml')
parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640)
```

