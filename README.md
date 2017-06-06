# ALL CONVOLUTIONAL NET

## 依赖环境
```
tensorflow r1.1 或 r1.2
numpy
cv2
```

## 代码结构
```
cifar10_input.py   读取数据
cifar10.py         定义模型
cifar10_train.py   训练模型
cifar10_eval.py    测试模型（使用cifar10测试集）
cifar10_test.py    测试模型（使用用户指定的图片集，测试过JPGE和PNG格式）
```

## 运行方法
训练模型：`python cifar10_train.py`。程序会检测`train_data`目录下是否有`cifar-10-binary.tar.gz`文件，如果没有，会自动下载并解压。程序运行过程中会定期向`cifar10_train`目录下写入checkpoint。
```bash
$ python cifar10_train.py 
Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
total params: 1369738
2017-06-05 23:52:23.098317: step 0, loss = 2.71 (72.1 examples/sec; 1.776 sec/batch)
2017-06-05 23:52:43.690872: step 10, loss = 2.47 (62.2 examples/sec; 2.059 sec/batch)
2017-06-05 23:53:04.070340: step 20, loss = 2.31 (62.8 examples/sec; 2.038 sec/batch)
......
```

测试模型：`python cifar10_eval.py`。程序会读取`cifar10_train`目录下的checkpoint，然后测试模型在10000张测试集上的准确率。也可使用命令行参数`--checkpoint_dir dir`更改默认的checkpoint位置
```bash
$ python cifar10_eval.py
python cifar10_eval.py --run_once --checkpoint_dir save_cifar10_train 
2017-06-05 23:55:34.964539  53861: precision @ 1 = 0.930
```

测试模型：`python cifar10_test.py --images path1 path2 ... pathN`，会从`saved_cifar10_train`目录下读取checkpoint,使用`path1, path2, ..., pathN`来测试模型，`pathX`是图片路径，也可以是只包括图片的目录，图片可以是JPEG和PNG格式。如果图片文件名是形如`[0-9]_xxx.jpg`的格式，其中`[0-9]`是标记，那么程序会给出准确率。
```bash
$  python cifar10_test.py --images test
airplane    : 1036/10000
automobile  : 1009/10000
bird        :  959/10000
cat         :  968/10000
deer        : 1011/10000
dog         : 1009/10000
frog        : 1039/10000
horse       :  987/10000
ship        : 1004/10000
truck       :  978/10000
9290/10000, 92.9%
```

### 注：
本程序使用了Tensorflow官方教程中的cifar10代码，代码地址为[models/tutorials/image/cifar10](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)。本程序仅使用了其数据读取代码，完全重写了神经网络模型，教程代码的准确率仅为86%，本程序的准确率达到92.9%。