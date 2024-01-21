# -*- coding: utf-8 -*-


# 参数设置
class MyConfig(object):
    def __init__(self):
        self.config_dict = {
            # 数据路径
            "data_path": {
                "train_path": "/root/code/bert_cls2/data/data1/train.tsv",
                "valid_path": "/root/code/bert_cls2/data/data1/valid.tsv",
                "test_path": "/root/code/bert_cls2/data/data1/test.tsv",
            },
            # 使用预训练模型
            "pre_train_model": {
                "model_path": "/root/code/bert_cls2/prev_trained_model/chinese-bert-wwm/pytorch_model.bin",
                "config_path": "/root/code/bert_cls2/prev_trained_model/chinese-bert-wwm/config.json",
                "vocab_path": "/root/code/bert_cls2/prev_trained_model/chinese-bert-wwm/vocab.txt",
            },
            # 训练和测试参数配置
            "training_config": {
                "epoch": 3,
                # "max_length": 300,
                # "hidden_dropout_prob": 0.1,  # 0.1, 0.3, 0.5
                "num_labels": 2,  # 几分类个数
                "learning_rate": 1e-5,  # 1e-3, 1e-5
                "weight_decay": 1e-3,  # 0.001
                "batch_size": 32,  #
                "is_clean": True,
            },
            # 结果保存
            "result": {
                "save_path": "/root/code/bert_cls2/result",
                "model_save_path": "/root/code/bert_cls2/result/model.bin",
                "config_save_path": "/root/code/bert_cls2/result/model_config.json",
                "vocab_save_path": "/root/code/bert_cls2/result/model_vocab.txt",
            },
        }

    # 获取配置方法
    def get(self, section, name):
        return self.config_dict[section][name]
