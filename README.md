# oppo 6g ai 大赛 A榜 0.7+方案
~

python trainer.py 训练任务一 python trainer2.py 训练任务二 训练好的模型文件保存在 submit_pt文件夹中

模型方案 模型方案为AutoEncoder，中间的表征为48bits量化表征

Encoder和Decoder均采用标准Transformer（Attention is All You Need）结构，生成时，仅使用Decoder即可

量化方案参考CSINet
