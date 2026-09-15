# NGDM-SE: Noise-Guided Dual-Memory xLSTM for Single Channel Speech Enhacement (Submitted in ICASSP 2027 for Possible Publication) #

This repo contains the implementation of the NGDM-SE: Noise-Guided Dual-Memory xLSTM for Single Channel Speech Enhacement paper. 

# Model #
![Description of the image](images/SE.png)

# Test Audio example and Saved Model Weights #
The audio samples on the reverb. and out-of-domain test can be found [here](test_audio_samples/).

# Datasets #
To download and extract the dataset is available here: [EARS Data](https://sp-uhh.github.io/ears_dataset/) and [VoiceBAnk+Demands](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) and MUSAN Noise Data can be found [here](https://www.openslr.org/17/)

# Acknowledgements #
We thank the authors of the following baseline papers used for comparison with our proposed method: [ConvTasNet](https://ieeexplore.ieee.org/document/8707065), [DCCRN](https://arxiv.org/abs/2008.00264), [MPSENet](https://arxiv.org/abs/2305.13686), [Demucs](https://arxiv.org/abs/2211.08553), and [xLSTM-SENet](https://arxiv.org/abs/2501.06146).
