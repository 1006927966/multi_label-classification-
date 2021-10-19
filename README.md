# multi_label-classification-

### 工程环境
  pytorch 环境

### 工程入口
  正常入口为 train.py文件  
  多机多卡入口为  train_multi_machine.py（超参LOADERNAME="multi_machine"  必须如此设置）
  coco标签数据入口  train_coco.py
  单标签训练入口 train_single.py
### 超参定义
  所有的超参定义在config/value_config文件中。 
  
