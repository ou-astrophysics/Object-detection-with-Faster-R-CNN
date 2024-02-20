# Object-detection-with-Faster-R-CNN
Most things are setup via a config-file, e.g. `Zoobot-backbone-finetuned5_5channels_config.json`:
```javascript
{
    // name used for model checkpoints
    "model_name": "Zoobot-backbone-finetuned5_5channels",
    // descriptive, not used
    "model_type": "Zoobot",
    // defines which ResNet-blocks of the backbone will be unfreezed for training
    // 0 - none, 1 - layer 1, ..., 5 - all blocks
    "trainable_layers": 0,
    // descriptive, not used
    "description": "3-channel ResNet50 initialised with weights from Zoobot, transfer learning mode",
    // output directory for logs and checkpoints
    "log_dir": "./models/Zoobot-backbone-finetuned5_5channels/",
    // path to pre-trained checkpoint for the ResNet backbone, should be a Zoobot-checkpoint
    "pretrained_ckpt": "./pretrained_models/Zoobot-evo-epoch=23-step=92160_5channel.ckpt",
    // path to dataframe (parquet) with bbox and label data
    "data_table": "./data/HSC_crossmatch_model_training_5channel.gzip",
    // path to directory containing the subdirectories /pngs and /masks
    "image_dir": "./data/",
    // batch size
    "batch_size": 16,
    // number of classes for the detector to classify. Should always includes +1 for background.
    "num_classes": 6,
    // number of epochs to traine
    "epochs": 150,
    "channels": 5,
    "bands": 'IRGZY,
    // mean pixel values for each channel
    "image_mean":
    [
        0.1752493,
        0.20197728,
        0.216239,
        0.23456442,
        0.25831807
    ],
    "image_std":
    [
        0.15935956,
        0.18195586,
        0.19402547,
        0.20103182,
        0.2069739
    ],
    // not used here
    "k_folds": 0,
    "validation_split": 0.2,
    "test_split": 0.0,
    // name of the final model checkpoint in folder log_dir for validation and inference
    "final_model_ckpt": "Zoobot-backbone-finetuned5_5channels_FRCNN_epoch_18.pth"
}
```