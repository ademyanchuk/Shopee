Config = {
    # data
    "image_id_col": "image",
    "target_col": "target",
    "img_size": 384,  # resize
    "crop_size": None,  # if none == img_size
    # model bacbone
    "arch": "resnet50d",
    "pretrained": True,
    "global_pool": "avg",
    "embed_size": 512,
    "drop_rate": 0.0,
    "model_kwargs": {"drop_path_rate": None},
    "bn_momentum": 0.1,  # default 0.1
}
