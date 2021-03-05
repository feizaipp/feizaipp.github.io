---
layout:     post
title:      faster-rcnn源代码分析(一) 数据加载
#subtitle:  
date:       2019-12-23
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - DeepLeaning
    - AI
    - Object Detective
---

> [我的博客](http://feizaipp.github.io)

# 1. 概述
&#160; &#160; &#160; &#160;本文记录下 Faster-RCNN 网络加载数据集的实现，以 PASCAL VOC 数据集为例。

# 2. PASCAL VOC 介绍
&#160; &#160; &#160; &#160;首先我们将从网上下载下来的数据集解压，然后进入 VOCdevkit 目录，执行 tree -d 命令查看目录结构。

```
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   ├── Action
    │   ├── Layout
    │   ├── Main
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```
## 2.1 VOC2012
&#160; &#160; &#160; &#160;数据集是 2012 年的版本。

## 2.2 Annotations
&#160; &#160; &#160; &#160;存放 XML 文件，存放图像的标注信息，与 JPEGImages 中的图片一一对应。XML前面部分声明图像数据来源，大小等元信息。常用的标注信息如下：
> segmented: 为 1 表示有分割标注，为 0 表示没有分割标注。  
> object: 标注目标的详细信息，用于目标检测。常用字段如下：  
* name: 表示对象类别。
* pose: 表示采用是从什么视角，常见的有Left、Right、Frontal、rear。
* difficult: 是否被标记为很难识别对称，0表示不是，1表示是。
* truncated: 是否被标记为截断，0表示没有，1表示是。
* bndbox: 标签描述box框在图像上的 位置。

## 2.3 ImageSets
&#160; &#160; &#160; &#160;存放数据集的文件列表信息。 Action 中是所有具有 Action 标注信息图像文件名的 txt 文件列表； Layout 中的 txt 文件表示包含 Layout 标注信息的图像文件名列表； Segmentation 中是包含语义分割信息图像文件的列表； Main 文件夹中包含 20 个类别每个类别一个 txt 文件，每个 txt 文件都是包含该类别的图像文件名称列表，还包括已经划分好的数据集， train.txt 表示是的训练数据集合； val.txt 表示验证集数据； trainval.txt 表示训练与验证集数据； test.txt 表示测试集数据。

## 2.3 JPEGImages
&#160; &#160; &#160; &#160;所有的原始图像文件，格式必须是JPG格式。

## 2.4 SegmentationClass
&#160; &#160; &#160; &#160;所有分割的图像标注，分割图像安装每个类别标注的数据。

## 2.5 SegmentationObject
&#160; &#160; &#160; &#160;所有分割的图像标注，分割图像按照每个类别每个对象不同标注的数据。

# 3. 数据集预处理
&#160; &#160; &#160; &#160;对于训练集，首先将数据转会为 Tensor 类型，然后 50% 的概率对训练数据进行随机水平翻转。
```
data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
```

# 4. 数据解析
&#160; &#160; &#160; &#160;数据加载的实现在 VOC2012DataSet 类中。

&#160; &#160; &#160; &#160;init 函数，有三个参数：
* voc_root: 数据集所在目录
* transforms: 数据预处理接口
* train_set: 数据集类型
```
class VOC2012DataSet(Dataset):
    def __init__(self, voc_root, transforms, train_set=True):
        self.root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file
        if train_set:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "train.txt")
        else:
            txt_list = os.path.join(self.root, "ImageSets", "Main", "val.txt")
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines()]

        # read class_indict
        try:
            json_file = open('pascal_voc_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms
```

&#160; &#160; &#160; &#160;init函数主要是根据划分好的数据集加载每一张图像的标注文件，并将标注文件的路径保存到 self.xml_list 变量中。

&#160; &#160; &#160; &#160;加载 pascal_voc_classes.json 文件，该文件存储着每种类别对应的标签，并保存到 self.class_dict 变量中，以字典的形式存储。

&#160; &#160; &#160; &#160;实现 __len__ 函数，让实例类对象支持 len 方法，返回数据集的长度。
```
def __len__(self):
        return len(self.xml_list)
```

&#160; &#160; &#160; &#160;实现 __getitem__ 函数，可以让实例类对象支持根据索引值访问元素，这里返回的是图像信息和标注信息。
```
def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
```
&#160; &#160; &#160; &#160;根据 parse_xml_to_dict 函数解析 xml 文件，这里只关心 object 标签，一张图片可能有多个 object ，将每个 object 信息以列表的形式存储，然后将信息以字典的形式存储到 target 中。

&#160; &#160; &#160; &#160;parse_xml_to_dict 函数，参数是 etree._ElementTree 对象，递归调用 parse_xml_to_dict 函数，直到将所有 etree._ElementTree 对象都遍历完成，遍历到底层后，直接返回 tag 对应的信息。因为 object 可能有多个，所以需要将所有 object 的信息 append 到链表，每一个 tag 以字典的形式存储。
```
def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
```

&#160; &#160; &#160; &#160;get_height_and_width 函数返回当前索引对应图像的高和宽。
```
def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width
```

# 4. 数据加载
&#160; &#160; &#160; &#160;加载数据集，以下是加载训练集的实现：
```
train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], True)
```
&#160; &#160; &#160; &#160;调用 pytorch 内的 DataLoader 接口， DataLoader 接口的参数如下：
* train_data_set: 可迭代的数据加载器
* batch_size: 批处理大小，根据自己的硬件设备实际情况而定
* num_workers: 加载数据线程数
* collate_fn: 如何取样本，我们可以定义自己的函数来准确地实现想要的功能，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch

```
train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=utils.collate_fn)

def collate_fn(batch):
    return tuple(zip(*batch))
```
&#160; &#160; &#160; &#160;训练模型时加载数据：
```
utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, train_loss=train_loss, train_lr=learning_rate,
                              print_freq=50, warmup=True)
```
&#160; &#160; &#160; &#160;train_one_epoch 函数中通过调用 metric_logger.log_every 来加载数据。
```
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    train_loss=None, train_lr=None, warmup=False):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if isinstance(train_loss, list):
            # 记录训练损失
            train_loss.append(loss_value)

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if isinstance(train_lr, list):
            train_lr.append(now_lr)
```

&#160; &#160; &#160; &#160;MetricLogger 将数据集与训练输出信息结合在了一起，方便统计训练过程中时间参数。 MetricLogger 使用方式如下：
```
metric_logger = MetricLogger(delimiter="  ")
metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
for images, targets in metric_logger.log_every(data_loader, print_freq, header):
```

```
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,

                                                         total_time / len(iterable)))
```
&#160; &#160; &#160; &#160;上面一大段代码跟加载数据相关的就下面这两行。
```
for obj in iterable:
    yield obj
```

&#160; &#160; &#160; &#160;至此，数据解析和加载就介绍完了。
