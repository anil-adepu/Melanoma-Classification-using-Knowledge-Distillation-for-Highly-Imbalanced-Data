### Code for **Plain Augment** methods mentioned in the paper is present in <sub><sup>`plain_augmentation`</sub></sup> directory above
* Use `1_plain_augment_64k.py` to generate ~32k augmented images from 581 melanoma images to balance the classes.
* `2_stratify_prepare_dataset.py` prepares the final stratified jpg image dataset (available [here](https://www.kaggle.com/datasets/aniladepu/ablation-class-balanced-isic-2020)) which is later converted to tfrecords (available [here](https://www.kaggle.com/datasets/aniladepu/ablation-stratified-cb-64k-isic2020-tfrecs)) to train the models


### Code for **CycleGAN Augment** methods mentioned in the paper is present in <sub><sup>`gan_synthesis`</sub></sup> directory
