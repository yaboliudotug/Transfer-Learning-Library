# dataset
1. detectron采用注册器机制，有两个主要的注册器，MetadataCatalog和DatasetCatalog，前者存储每个数据集对应的类别信息，后者存储数据集的图像路径和标注信息
2. 在tllib中，定义自定义数据集类别，在初始化数据集类别时，通过调用register_pascal_voc将该数据集注册至MetadataCatalog和DatasetCatalog，其中DatasetCatalog注册的不是全部的数据，而是数据加载函数，这样做的目的是，不在项目启动时就将大量的数据加载而消耗数据，而是在确实使用到该数据时，通过调用数据加载函数再将需要的数据加载至内存。
args.sources = utils.build_dataset(args.sources[::2], args.sources[1::2])返回的就是该注册器内的注册对象
DatasetCatalog的注册的数据集格式list(dict)：
r = {
    "file_name": jpeg_file,
    "image_id": fileid,
    "height": int(tree.findall("./size/height")[0].text),
    "width": int(tree.findall("./size/width")[0].text),
    "annotations"：{
        "category_id": class_names.index(cls),
         "bbox": ["xmin", "ymin", "xmax", "ymax"], 
         "bbox_mode": BoxMode.XYXY_ABS}
}

将DatasetCatalog中注册得到的dict格式数据集，传入detectron2 build_detection_test/train_loader方法，来构建使用的数据集，通过mapper类将dict数据重新组织，使用该形式，组件dataloader，用于检测模型的训练或测试
r = {
    "image": img_np,
    "height": int(tree.findall("./size/height")[0].text),
    "width": int(tree.findall("./size/width")[0].text),
    "instances": {
        gt_boxes,
        gt_classes,
        gt_masks,
        gt_keypoints
    }
}

7. ProposalGenerator。用于处理decouple roi-head输出的result和background_result，
{
    self.image_id = image_id
    self.filename = filename
    self.pred_boxes = pred_boxes
    self.pred_classes = pred_classes
    self.pred_scores = pred_scores
    self.gt_classes = gt_classes
    self.gt_boxes = gt_boxes
    self.gt_ious = gt_ious
    self.gt_fg_classes = gt_fg_classes
    self.all_gt_classes = all_gt_classes
    self.all_gt_boxes = all_gt_boxes
}
2. ProposalDataset, 用于category和bbox的adaptor训练，定义于tllib，每次get_item以图像为单位，随机选取一个img内的1个proposal。返回内容为
    {    
        img, 
        {
            "image_id": proposal.image_id,
            "filename": proposal.filename,
            "pred_boxes": proposal.pred_boxes.astype(np.float),
            "pred_classes": proposal.pred_classes.astype(np.long),
            "pred_scores": proposal.pred_scores.astype(np.float),
            "gt_classes": proposal.gt_classes.astype(np.long),
            "gt_boxes": proposal.gt_boxes.astype(np.float),
            "gt_ious": proposal.gt_ious.astype(np.float),
            "gt_fg_classes": proposal.gt_fg_classes.astype(np.long),
            "width": image_width,
            "height": image_height
        }
    }
！！！！！！此处需要修改，在测试时，ProposalDataset返回也为随机选择，不合理，应该全部测试，或者固定选取个数
同时应该测试target_test的GT，而不仅仅是target—trian-proposal，可能导致过拟合

2. 模型使用的同样为注册器机制，位于detectorn2/modeling/build中META_ARCH_REGISTRY，使用fvcore.common.registry实现了一个Register类，在每一个实现的模型类前使用装饰器将模型注册@META_ARCH_REGISTRY.register()，当detectorn工程启动时，META_ARCH_REGISTRY将所有的实现模型注册。对于tllib，通过初始化调用所自定义模型的方法，将模型注册。
3. 模型构建方式。detectron实现了configurable方法，装饰在模型的init方法前，通过该函数，可以将模型类的init参数首先传递到类内的from_config参数中，再使用该函数的返回值来init对象。在from_config方法中，模型定义了build_backbone等各模型构建方法，从而实现从配置文件到torch.mm.module的构建。
4. backbone fpn head等模块使用了同样类似的注册机制。与mmdetection类似
5. detectron2检测模型forward方法，首先会判断当前模型的traininng状态，若为training则直接返回losses字典（包含proposal和head损失），若不是training状态，则调用inference方法，返回检出result
6. 作者提出的decouple模型，对roi-head进行了创新，返回result和background_result，同样inference方法也返回两种result。其中backgroud_result是将roi-head最终预测结果中的背景样本进行采样返回，能够进入到roi-head阶段的proposal，算是难例背景。





