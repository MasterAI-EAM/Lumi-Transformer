_base_ = [
    '../_base_/models/swin_transformer_v2/large_384.py',
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg = dict(
            #frozen_stages = 3,
            type='Pretrained', 
            checkpoint="checkpoints/swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth", 
            prefix='backbone')
    ),
    head=dict(
        num_classes=2,
        topk = (1, )
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5)
    ])
    )

img_norm_cfg = dict(
     mean=[124.508, 116.050, 106.438],
     std=[58.577, 57.310, 57.437],
     to_rgb=True)

dataset_type = 'CustomDataset'
classes = ['0', '1']  

data = dict(
    # 每个 gpu 上的 batch size 和 num_workers 设置，根据计算机情况设置
    samples_per_gpu = 4,
    workers_per_gpu=2,
    # 指定训练集路径
    train = dict(
        #data_prefix = 'data/cats_dogs_dataset/training_set/training_set',
        #classes = 'data/cats_dogs_dataset/classes.txt'
        
        data_prefix = 'data/poly/train',
        classes = 'data/mono/class.txt'
    ),
    # 指定验证集路径
    val = dict(
        #data_prefix = 'data/cats_dogs_dataset/val_set/val_set',
        #ann_file = 'data/cats_dogs_dataset/val.txt',
        #classes = 'data/cats_dogs_dataset/classes.txt'

        data_prefix = 'data/mono/val',
        ann_file = 'data/mono/val.txt',
        classes = 'data/mono/class.txt'
    ),
    # 指定测试集路径
    test = dict(
        #data_prefix = 'data/cats_dogs_dataset/test_set/test_set',
        #ann_file = 'data/cats_dogs_dataset/test.txt',
        #classes = 'data/cats_dogs_dataset/classes.txt'

        data_prefix = 'data/mono/test',
        ann_file = 'data/mono/test.txt',
        classes = 'data/mono/class.txt'
    )
)
# 修改评估指标设置
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(lr=5e-4 * 4 * 1 / 512)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-1,
    warmup='linear',
    warmup_ratio=1e-1,
    warmup_iters=100,
    warmup_by_epoch=False)

runner = dict(max_epochs=40)

