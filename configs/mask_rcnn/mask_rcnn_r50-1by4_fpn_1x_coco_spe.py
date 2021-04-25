_base_ = ['mask_rcnn_r50_fpn_1x_coco_spe.py']

# model settings
base_channels = 16
model = dict(
    type='MaskRCNN',
    pretrained=  # noqa
    '/mnt/lustre/share/lixinran/ckpt/mmspe/faster_resnet50_1by4.pth',  # noqa
    backbone=dict(
        type='FasterResNet',
        faster=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        base_channels=base_channels,
        stem_channels=base_channels,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256 * 2**i // (64 // base_channels) for i in range(4)],
        out_channels=64,
        num_outs=5),
    rpn_head=dict(type='RPNHead', in_channels=64, feat_channels=64),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=256,
            roi_feat_size=7,
            num_classes=80),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead', in_channels=64, conv_out_channels=64)))
