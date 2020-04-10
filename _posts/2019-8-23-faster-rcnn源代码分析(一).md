---
layout:     post
title:      faster-rcnn源代码分析(一)
#subtitle:  
date:       2019-8-23
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
&#160; &#160; &#160; &#160;学习目标检测有一段时间了，看过几个经典的目标检测算法原理，如 faster-rcnn 、 YOLO 、 SSD 等。但我认为光了解原理是远远不够的，还要通过分析源码加深对算法的理解，我打算先分析 faster-rcnn 的源码，从训练到测试，将整个算法的原理与源码串起来，后续在分析 YOLO 和 SSD 的源码。

&#160; &#160; &#160; &#160;本文是 faster-rcnn 源代码分析的第一篇，介绍数据集的加载。了解数据集的加载对理解模型的训练过程尤为重要。

```
#py-faster-rcnn/tools/train_faster_rcnn_alt_opt.py
def get_solvers(net_name):
    # Faster R-CNN Alternating Optimization
    n = 'faster_rcnn_alt_opt'
    # Solver for each training stage
    solvers = [[net_name, n, 'stage1_rpn_solver60k80k.pt'],
               [net_name, n, 'stage1_fast_rcnn_solver30k40k.pt'],
               [net_name, n, 'stage2_rpn_solver60k80k.pt'],
               [net_name, n, 'stage2_fast_rcnn_solver30k40k.pt']]
    solvers = [os.path.join(cfg.MODELS_DIR, *s) for s in solvers]
    # Iterations for each training stage
    max_iters = [80000, 40000, 80000, 40000]
    # max_iters = [100, 100, 100, 100]
    # Test prototxt for the RPN
    rpn_test_prototxt = os.path.join(
        cfg.MODELS_DIR, net_name, n, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt
```

&#160; &#160; &#160; &#160;get_solvers 函数根据网络名称返回训练时各个阶段使用的 .prototxt 文件路径、每个阶段的最大循环次数、 rpn 测试用的 .prototxt 文件路径。函数传入参数 net_name ，比如是 VGG16 , 那么函数返回指为：
```
solvers:
['models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt',
'models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage1_fast_rcnn_solver30k40k.pt',
'models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage2_rpn_solver60k80k.pt',
'models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage2_fast_rcnn_solver30k40k.pt']

max_iters:
[80000, 40000, 80000, 40000]

rpn_test_prototxt:
'models/pascal_voc/VGG16/faster_rcnn_alt_opt/rpn_test.pt'
```
&#160; &#160; &#160; &#160;get_roidb 函数, imdb_name 的默认值是 voc_2007_trainval 。
```
#py-faster-rcnn/tools/train_faster_rcnn_alt_opt.py
def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb
```
&#160; &#160; &#160; &#160;get_imdb 函数，该函数返回 __sets 字典的 voc_2007_trainval 健值对应的值。下面来看看 __sets 字典是怎么初始化的。
```
#py-faster-rcnn/lib/datasets/factory.py
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()
```
&#160; &#160; &#160; &#160;从下面代码可知， __sets 是由 pascal_voc 类初始化的，那么对应 voc_2007_trainval 健值的初始化为 pascal_voc('2007', 'trainval') ，下面在看 pascal_voc 的构造函数。
```
#py-faster-rcnn/lib/datasets/factory.py
__sets = {}
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
```
&#160; &#160; &#160; &#160; pascal_voc 类继承自 imdb 类， 该类的构造函数开始对数据集进行初始化。
```
#py-faster-rcnn/lib/datasets/pascal_voc.py
class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
```
&#160; &#160; &#160; &#160;调用构造函数之后，就是 get_imdb 函数的返回值 imdb ，此时该值的内容如下：
```
self._name = voc_2007_trainval
self._num_classes = 21
self._obj_proposer = 'selective_search'
self._year = 2007
self._image_set = trainval
self._devkit_path = data/VOCdevkit2007 #这个目录是一个链接文件
self._data_path = data/VOCdevkit2007/VOC2007
self._classes = xxx #数据集的类别数，算上背景一共 20 种物体
self._class_to_ind = xxx #字典，以类别为健，以序号为值
self._image_ext = '.jpg'
self._image_index = xxx #从 data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt 加载训练集编号列表，该变量中保存着训练集的编号，根据这个编号可以得到图片以及 groud-truth 信息。
self._roidb_handler = self.selective_search_roidb
self._salt = str(uuid.uuid4())
self._comp_id = 'comp4'
self.config = {'cleanup' : True, 'use_salt' : True, 'use_diff' : False, 'matlab_eval' : False, 'rpn_file' : None, 'min_size' : 2}
```

```
class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}
```
&#160; &#160; &#160; &#160;get_imdb 函数只是返回数据集的信息，还未加载数据。

&#160; &#160; &#160; &#160;在回到 get_roidb 函数，调用 imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD), cfg.TRAIN.PROPOSAL_METHOD 值为 'gt'
```
#py-faster-rcnn/lib/datasets/imdb.py
def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method
```
&#160; &#160; &#160; &#160;method 为执行 self.gt_roidb 函数的返回值。
```
def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
```
&#160; &#160; &#160; &#160;第一次加载肯定不存在 data/cache/voc_2007_trainval_gt_roidb.pkl 文件，所以从数据集中读取，读取后缓存到上述文件中。根据数据集编号读取，之前已经将这些编号保存到 self.image_index 变量中了。
```
def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}
```
&#160; &#160; &#160; &#160;根据编号找到 data/VOCdevkit2007/VOC2007/Annotations/{index}.xml 文件，该文件中存放着 groud-truth 信息。首先获得 xml 文件中的 object 标签，条件是 difficult 值为 0 。然后遍历这些 objs 得到 boxes 存储的是 groud-truth 的左上角和右下角的坐标， gt_classes 存储着类别的序号， overlaps 存储着 groud-truth 的 IOU 为 1 ， seg_areas 存储着 groud-truth 的面积。最终返回字典 {'boxes' : boxes, 'gt_classes': gt_classes,'gt_overlaps' : overlaps,     'flipped' : False, 'seg_areas' : seg_areas} ，最终返回的 gt_roidb 是一个字典列表，存储的每一个参加训练的数据集的 groud-truth 信息。这个字典的列表最终赋值给了 imdb 对象的 roidb_handler 属性。

&#160; &#160; &#160; &#160;接着，返回到 get_roidb 函数，调用 get_training_roidb 函数。
```
#py-faster-rcnn/lib/fast_rcnn/train.py
def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb
```
&#160; &#160; &#160; &#160;在 config.py 文件中定义了 cfg.TRAIN.USE_FLIPPED=True ，所以接近着调用 append_flipped_images 函数，该函数对图像进行旋转，实际上是沿着 y 轴翻转， x 轴保持不变。
```
#py-faster-rcnn/lib/datasets/imdb.py
def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
```
&#160; &#160; &#160; &#160;num_images 为参与训练的图片的列表，实际上为 len(self.image_index) 。这里用到了装饰器 @property 和 @xxx_setter ，其实就是获取和设置属性。 widths 为一个列表，返回所有参加训练图像的宽度。然后遍历每一个训练集，将翻转后的图像信息添加到 roidb 中，这样就相当与参加训练的数据翻倍了，使得训练出的模型更好。主要是 bounding-box 的坐标值要进行变换。

&#160; &#160; &#160; &#160;返回到 get_training_roidb 函数，继续分析 prepare_roidb 函数。
```
#py-faster-rcnn/lib/roi_data_layer/roidb.py
def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
```
&#160; &#160; &#160; &#160;sizes 为每个训练集的大小；继续初始化 roidb 变量。
