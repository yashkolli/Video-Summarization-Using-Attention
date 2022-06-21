# Video Summarization using Attention

Video Summarization is a challenging problem with wide potential. Prior approaches in this field focussed on sampling useful frames and assembling them as summaries. Also most of them were unsupervised approaches.

## Video Summarization as a Supervised Subset Selection Problem

This implementation considers Video Summarization as a _supervised subset selection_ problem. Formulated as a _sequence - to - sequence learning_ problem, Video Summarization has the input as a sequence of original video frames and output as the keyshot sequence. This model was experimented on the TVSum and SumMe datasets.

## Dataset Preprocessing

The original datasets are preprocessed. The codes for both TVSum and SumMe are given in *tvsum_md.py* and *summe_md.py* files respectively. The videos are uniformly downsampled to 320 frames. And KTS algorithm is used to get the change points. The detailed preprocessing algorithms and processes can be understood by referring to the code.

## Model

The framework consists of two components: an encoder-decoder model and a keyshot selection model. The encoder-decoder part measures the importance of each frame. The key shots selection model helps us to convert frame-level importance scores into shot-level scores and generating summary accounting to the threshold budget which we specify.

More details of the model can be known by skimming through the code.

## Training the model

Modify the config file with the path of the dataset. And run,
`
python train.py
`

For every epoch, the model weights will be saved and these will be used while creating the summary of a test video. Evaluation results will be automatically printed on your screen while the model is being trained.

There are different kinds of models available in the *train.py* file. The code is properly structured and all the comments are in place. Please refer to the comments and kindly make the corresponding changes required to run different models.

The various options available to choose are:

* _dataloader.py_ options -> you can change the data that is fabricated from the h5 file to pass to the models. This will change depending on the deep learning model you want to use

* _train.py_ options
    1. You can choose the dataset on which the model will be trained
    2. You can choose which model to use

Accordingly comment and uncomment the lines of code.

## Generating Summary

To generate the summary of a video, run
```
python gen_summary.py --h5_path --json_path --data_root --save_dir 
```
