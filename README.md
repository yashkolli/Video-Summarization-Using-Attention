# Video-Summarization-Using-Attention
Video summarization is a challenging problem with great application potential. Whereas prior approaches, largely unsupervised in nature, focus on sampling useful frames and assembling them as summaries, this implementation considers video summarization as a supervised subset selection problem. This problem of supervised video summarization is tackled by formulating it as a sequence-to-sequence learning problem, where the input is a sequence of original video frames, the output is a keyshot sequence. Extensive experiments are conducted on two video summarization benchmark datasets, i.e., SumMe, and TVSum.
# Dataset
TVSum and SumMe datasets (downsampled to 320 frames) and preprocessed by extracting the features from the last pooling layer of the  GoogleNet Model are used (hd5 file). Python Script for preparing the datasets will be uploaded shortly.
# Train
Config.py file contains the paths to the datasets. Make sure to change the paths to where your datasets are saved, and run
`python train.py`<br />
For every epoch, the model weights will be saved and these will be used while creating the summary of a test video. Evaluation results will be automatically printed on your screen while the model is being trained.
# Generate Summary
To generate the summary of a video, run <br />
`python gen_summary.py --h5_path --json_path --data_root --save_dir --bar`
