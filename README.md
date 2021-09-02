# Video Summarization using Attention

Video Summarization is a challenging problem with wide potential. Prior approaches in this field focussed on sampling useful frames and assembling them as summaries. Also most of them were unsupervised approaches.

## Video Summarization as a Supervised Subset Selection Problem

This implementation considers Video Summarization as a _supervised subset selection_ problem. Formulated as a _sequence - to - sequence learning_ problem, Video Summarization has the input as a sequence of original video frames and output as the keyshot sequence. This model was experimented on the TVSum dataset.

Link to the preprocessed dataset :
[TVSum](https://drive.google.com/file/d/1SfImsAvUpT_HsiqdEmeyYipQUnFFlDbV/view?usp=sharing)

## Model

The framework consists of two components: an encoder-decoder model and a keyshot selection model. The encoder-decoder part measures the importance of each frame. The key shots selection model helps us to convert frame-level importance scores into shot-level scores and generating summary accounting to the threshold budget which we specify.

More details of the model can be known by skimming through the code.

## Training the model

Modify the config file with the path of the dataset. And run,
`
python train.py
`

For every epoch, the model weights will be saved and these will be used while creating the summary of a test video. Evaluation results will be automatically printed on your screen while the model is being trained.

## Generating Summary

To generate the summary of a video, run
```
python gen_summary.py --h5_path --json_path --data_root --save_dir --bar
```
