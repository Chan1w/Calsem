# -*- coding: utf-8 -*-
"""数据模块：VOC 2012 数据集 + 语义失败评估。"""
from calsem.data.voc import VOCDataset, VOC_CLASSES  # noqa: F401
from calsem.data.metrics import multilabel_f1, semantic_failure  # noqa: F401
