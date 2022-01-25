# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

def build_model(args):
    if args.model_name == "hoi_ts":
        from models.ts.hoi_share_qpos_eobj_cos_kl import build
    else:
        from models.detr import build
    return build(args)
