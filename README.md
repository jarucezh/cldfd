# Cross-Level Distillation and Feature Denoising for Cross-Domain Few-Shot Classification
Code implementation of ICLR 2023 paper entitled "[Cross-Level Distillation and Feature Denoising for Cross-Domain Few-Shot Classification](https://openreview.net/forum?id=kCeP36h9c2T&referrer=%5Bthe%20profile%20of%20Hao%20ZHENG%5D(%2Fprofile%3Fid%3D~Hao_ZHENG4))"

**English version:**
## Updates on 2023-05-23
We firstly release the code of training with Cross-Level Distillation (CLD)

Train the model with the command:

```
python oho.py --dir tmp/dir_you_want --target_dataset EuroSAT --target_subset_split splits/EuroSAT_unlabeled_20.csv --alpha 2 --bsize 32 --coef 1
```

BTW, if you want to Train the SimCLR baseline, just modify the line 260 in ```methods/student9.py``` by removing the distillation loss. We will release the formal code soon!


## TODO on 2023-05-23

**Releasing the code of testing stage**, including Feature Denoising (FD). We are really sorry that we are still sorting. If you are in a hurry, you can also contact me in advance by sending emails to [me](hao.zheng@riken.jp). You are always welcome to contact us!!!  

********

**中文版：**
## 2023年5月23日更新
我们首先开源了训练Cross-Level Distillation (CLD)的代码

你只需要：
```
python oho.py --dir tmp/dir_you_want --target_dataset EuroSAT --target_subset_split splits/EuroSAT_unlabeled_20.csv --alpha 2 --bsize 32 --coef 1
```

如果你想训练SimCLR baseline的话，只需要修改```methods/student9.py```中第260行，把蒸馏loss去掉就可以了。后续我们也会开放正式的代码！！！

## 2023年5月23日TODO
**开放测试阶段的代码**，包括Feature Denoising (FD). 我很抱歉我们还在整理！如果你着急的话，也可以给[我](hao.zheng@riken.jp)发邮件。随时欢迎交流沟通！！！
