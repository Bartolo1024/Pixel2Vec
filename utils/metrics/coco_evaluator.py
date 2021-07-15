import json

from ignite import metrics
from pycocotools import coco, cocoeval


class COCOEvaluator(metrics.Metric):
    """Convert predictions and targets to coco annotations, dump it to json and compute mAP using cocotools"""
    def __init__(self, class_names, output_transform):
        self.categories = [{
            'supercategory': name,
            'id': idx + 1,
            'name': name
        } for idx, name in enumerate(class_names)]
        self.pred_annotations = {}
        self.gt_annotations = {}
        self.img_idx = 0
        self.gt_ann_idx = 0
        self.pred_ann_idx = 0
        super().__init__(output_transform)

    def reset(self):
        self.pred_annotations = self._init_annotations()
        self.gt_annotations = self._init_annotations()

    def _init_annotations(self):
        return {
            'info': {},
            'licenses': {},
            'images': [],
            'annotations': [],
            'categories': self.categories
        }

    def update(self, output):
        (pred_conts, prec_classes), annotations_batch = output
        for contours, classes, annotations in zip(pred_conts, prec_classes,
                                                  annotations_batch):
            image = {
                'file_name': annotations['file_name'],
                'width': annotations['width'],
                'height': annotations['height'],
                'id': self.img_idx
            }
            for cont, cls in zip(contours, classes):
                ann = {
                    'segmentation':
                    [[int(val) for cont in contours for val in cont]],
                    'iscrowd':
                    0,
                    'image_id':
                    self.img_idx,
                    'category_id':
                    int(cls),
                    'score':
                    1.,
                    'area':
                    1.,
                    'id':
                    self.pred_ann_idx
                }
                self.pred_annotations['annotations'].append(ann)
                self.pred_annotations['images'].append(image)
                self.pred_ann_idx += 1
            for gt_ann in annotations['annotations']:
                ann = {
                    'segmentation': gt_ann['segmentation'],
                    'iscrowd': 0,
                    'image_id': self.img_idx,
                    'category_id': gt_ann['category_id'],
                    'area': 1.,
                    'id': self.gt_ann_idx
                }
                self.gt_annotations['annotations'].append(ann)
                self.gt_annotations['images'].append(image)
                self.gt_ann_idx += 1
            self.img_idx += 1

    def compute(self):
        with open('pred_annos_tmp.json', 'w') as f:
            json.dump(self.pred_annotations, f)
        with open('gt_annos_tmp.json', 'w') as f:
            json.dump(self.gt_annotations, f)
        coco_dt = coco.COCO('pred_annos_tmp.json')
        coco_gt = coco.COCO('gt_annos_tmp.json')
        evaluator = cocoeval.COCOeval(coco_gt, coco_dt)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        out = {
            'AP': evaluator.stats[0],
            'AP50': evaluator.stats[1],
            'AP75': evaluator.stats[2],
            'APs': evaluator.stats[3],
            'APm': evaluator.stats[4],
            'APl': evaluator.stats[5],
            'AR_max_dets_1': evaluator.stats[6],
            'AR_max_dets_10': evaluator.stats[7],
            'AR_max_dets_100': evaluator.stats[8],
            'ARs': evaluator.stats[9],
            'ARm': evaluator.stats[10],
            'ARl': evaluator.stats[11],
        }
        return out
