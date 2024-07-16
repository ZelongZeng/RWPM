# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder

from global_cache import global_cache
import torch

def print_stats(x, name):
    
    print(f"{name}: (Min, Max, Mean, STD) ", 
        x.min().cpu().item(),
        x.max().cpu().item(),
        x.mean().cpu().item(),
        x.std().cpu().item(),
    )

@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        self.RWPM = global_cache.RWPM
        self.TT = global_cache.TT
        self.temperture = global_cache.temperture
        self.n_patrion = global_cache.n_patrion
        self.alpha = global_cache.alpha

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == 'simple_transformer_decoder':
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)
    
    def softmax(self, x, tau):
        s = torch.nn.Softmax(dim=1)
        return s(x/tau)
    

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":

            if self.RWPM:
                _, d_f, size_f_h, size_f_w = mask_features.shape
                
                sub_output_list = []
                size_p_h = int(size_f_h/self.n_patrion)
                size_p_w = int(size_f_w/self.n_patrion)

                for i in range(self.n_patrion):
                    for j in range(self.n_patrion):
                        f_map_sub = mask_features[:,:,size_p_h*i:size_p_h*(i+1),size_p_w*j:size_p_w*(j+1)]

                        
                        f_proto = f_map_sub.squeeze(0).reshape(d_f, -1).T       
                        f_proto_norm = torch.nn.functional.normalize(f_proto, dim=-1)

                        similarity = f_proto_norm.mm(f_proto_norm.T)

                        masks = torch.eye(f_proto.size(0), device='cuda').half()

                        W = similarity - masks * 1e9 # + local_mask

                        W = self.softmax(W, self.temperture)

                        f_t = self.alpha * W @ f_proto + (1-self.alpha) * f_proto
                        for tt in range(self.TT):
                            f_t = self.alpha * W @ f_t + (1-self.alpha) * f_proto      

                        f_proto_rw = f_t  
                        f_proto_rw = f_proto_rw.T

                        sub_output_list.append(f_proto_rw.view(d_f, size_p_h, size_p_w).unsqueeze(0).to(torch.float32))

                output_list = []
                for i in range(self.n_patrion):
                    output_list.append(torch.cat(sub_output_list[i*self.n_patrion:(i+1)*self.n_patrion], dim=3))

                mask_features = torch.cat(output_list, dim=2)
            

            predictions = self.predictor(multi_scale_features, mask_features, mask)
        elif self.transformer_in_feature == 'simple_transformer_decoder':
            predictions = self.predictor(mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions
