# NGDM-SE: Noise-Guided Dual-Memory xLSTM for Single Channel Speech Enhacement (Submitted in ICASSP 2027 for Possible Publication) #

This repo contains the implementation of the NGDM-SE: Noise-Guided Dual-Memory xLSTM for Single Channel Speech Enhacement paper. 

# Model #
![Description of the image](images/SE.png)

# Test Audio example and Saved Model Weights #
The test audio samples on the reverb. and out-of-domain datasets can be found [here](Test_samples/).

# Datasets #
To download and extract the dataset is available here: [EARS Data](https://sp-uhh.github.io/ears_dataset/) and [VoiceBAnk+Demands](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) and MUSAN Noise Data can be found [here](https://www.openslr.org/17/)

# Acknowledgements #
We thank the authors of the following baseline paper used for comparison with our proposed method: [ConvTasNet](https://ieeexplore.ieee.org/document/8707065), [DCCRN](Dccrn: Deep complex convolution recur-
rent network for phase-aware speech enhancemet), [MPSENet](Mp-senet: A speech enhancement model with parallel denoising of magnitude and phase spectra), [Demucs](Hybrid transformers for music source separation), and [xLSTM-SENet](xlstm-senet: xlstm for single-channel speech enhancement).
