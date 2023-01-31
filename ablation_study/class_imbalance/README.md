### Code for **Plain Augment** methods mentioned in the paper is present in <sub><sup>`plain_augmentation`</sub></sup> directory above
* Use `1_plain_augment_64k.py` to generate ~32k augmented images from 581 melanoma images to balance the classes.
* `2_stratify_prepare_dataset.py` prepares the final stratified jpg image dataset (available [here](https://www.kaggle.com/datasets/aniladepu/ablation-class-balanced-isic-2020)) which is later converted to tfrecords to train the models
    * TFRECORDS: [here](https://www.kaggle.com/datasets/aniladepu/ablation-stratified-cb-64k-isic2020-tfrecs)
    * Code to create tfrecs: [here](https://www.kaggle.com/code/aniladepu/ablation-create-tfrecs-64k-augmented-isic2020/notebook)


### Code for **CycleGAN Augment** methods mentioned in the paper is present in <sub><sup>`gan_synthesis`</sub></sup> directory
* Use `1_train_cycle_gan.ipynb` to train a CycleGAN which outputs NM2M.h5 to translate and synthesize melanoma images from non-melanoma samples.
    * Dataset to train: [here](https://www.kaggle.com/datasets/aniladepu/isic2020-gan-train-512)
* Use `2_cyclegan-upsample-melanoma.ipynb` to generate ~32k melanoma samples from each non-melanoma sample to balance the classes. This generates an augmented dataset of ~64k images.
* `3_stratify_prepare_jpg_dataset.ipynb` helps stratify and prepare the final stratified jpg image dataset (available [here](https://www.kaggle.com/datasets/aniladepu/ablation-stratified-gan-syn-jpg-256)) which is used to create tfrecords.
    * TFRECORDS: [here](https://www.kaggle.com/datasets/aniladepu/ablation-stratified-gan-syn-64k-isic2020-tfrecs)
    * Code to create tfrecs: [here](https://www.kaggle.com/code/aniladepu/ablation-create-tfrecs-64k-gan-syn-isic2020)
