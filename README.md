# CLD-FD
Code implementation of ICLR 2023 paper entitled "[Cross-Level Distillation and Feature Denoising for Cross-Domain Few-Shot Classification](https://openreview.net/forum?id=kCeP36h9c2T&referrer=%5Bthe%20profile%20of%20Hao%20ZHENG%5D(%2Fprofile%3Fid%3D~Hao_ZHENG4))"

## Updates on 2023-05-23
We firstly release the code of training with Cross-Level Distillation (CLD)

Train the model with the command:

```
python oho.py --dir tmp/dir_you_want --target_dataset EuroSAT --target_subset_split splits/EuroSAT_unlabeled_20.csv --alpha 2 --bsize 32 --coef 1
```

BTW, if you want to Train the SimCLR baseline, just modify the line 260 in ```methods/student9.py``` by removing the distillation loss. We will release the formal code soon!


## TODO on 2023-05-23

*Releasing the code of testing stage*, including Feature Denoising (FD). We are really sorry that we are still sorting. If you are in a hurry, you can also contact me in advance by sending emails to [me](zheng.h.ad@m.titech.ac.jp). You are always welcome to contact me!!!
