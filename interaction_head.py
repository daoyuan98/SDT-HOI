import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
import torch.distributed as dist
import pocket

from fnda import TransformerEncoderLayer as TELayer
from fnda import BertConnectionLayer as CrossAttentionEncoderLayer
import torchvision.ops.boxes as box_ops
import numpy as np

from hoi_mixup import TokenMixer

class LongShortDistanceEncoderLayer(nn.Module):
    def __init__(self, 
        d_model: int = 256, nhead: int = 8, 
        dim_feedforward: int = 512, dropout: float = 0.1, use_sp=False):
        super().__init__()
        self.use_sp = use_sp

        self.encoder_layer_a = TELayer(d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout_prob=dropout, use_sp=self.use_sp)
        
        if not use_sp:
            self.encoder_layer_b = TELayer(d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout_prob=dropout, use_sp=False)

    def forward(self, x, dist, boxes, hi, oi, mask_a, mask_b):
        if self.use_sp:
            x, attn = self.encoder_layer_a(x, dist, boxes, hi, oi, None)
            attn = [attn]
        else:
            x, attn1 = self.encoder_layer_a(x, dist, boxes, hi, oi, mask_a)
            x, attn2 = self.encoder_layer_b(x, dist, boxes, hi, oi, mask_b)
            attn = [attn1, attn2]
        return x, attn 

class ModifiedEncoder(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, num_layers: int = 2,
        dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([LongShortDistanceEncoderLayer(
            d_model=hidden_size, nhead=num_heads, 
            dim_feedforward=representation_size,
            dropout=dropout_prob,
            use_sp=i==0
        ) for i in range(num_layers)])


    def forward(self, x: Tensor, dist:Tensor, boxes:Tensor, hi: Tensor, oi: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        x = x.unsqueeze(0)
        
        mask_a, mask_b = self.generate_mask(dist)

        for i, layer in enumerate(self.mod_enc):
            x, attn = layer(x, dist, boxes, hi, oi, mask_a, mask_b)

            if isinstance(attn, list):
                attn_weights.extend(attn)

        x = x.squeeze(0)
        return x, attn_weights

    def generate_mask(self, pairwise_dist):
        n = pairwise_dist.shape[0]

        mask_a = torch.ones_like(pairwise_dist)
        mask_b = torch.ones_like(pairwise_dist)

        sorted, indices = torch.sort(pairwise_dist, dim=-1)

        split = n // 2

        tau_index = indices[:, split]
        tau_dist  = pairwise_dist[torch.arange(n), tau_index].unsqueeze(1)

        x_near, y_near = torch.nonzero(pairwise_dist <= tau_dist).unbind(1)
        x_far , y_far  = torch.nonzero(pairwise_dist >  tau_dist).unbind(1)

        mask_a[x_far , y_far]  = 0
        mask_b[x_near, y_near] = 0

        # can always attend to itself
        mask_a[torch.arange(n), torch.arange(n)] = 1
        mask_b[torch.arange(n), torch.arange(n)] = 1

        return mask_a, mask_b


class CompEncoder(nn.Module):
    def __init__(self, hidden_size, return_weights, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
        pocket.models.TransformerEncoderLayer(
            hidden_size=hidden_size,
            return_weights=return_weights
        ) for _ in range(num_layers)])

    def forward(self, x):
        weights = []
        for i in range(self.num_layers):
            x, w = self.layers[i](x)
            weights.append(w)
        return x, weights[-1]


class ObjectRelation(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
        nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=512
        ) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int
        Size of the object features
    representation_size: int
        Size of the human-object pair features
    num_channels: int
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """
    def __init__(self,
        box_pair_predictor: nn.Module,
        hidden_state_size: int, representation_size: int,
        num_channels: int, num_classes: int, human_idx: int,
        object_class_to_target_class: List[list], args
    ) -> None:
        super().__init__()

        self.box_pair_predictor = box_pair_predictor

        self.hidden_state_size = hidden_state_size
        self.representation_size = representation_size

        self.num_classes = num_classes
        self.human_idx = human_idx
        self.object_class_to_target_class = object_class_to_target_class

        self.args = args

        self.token_mixer = TokenMixer()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.global_feature_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.coop_layer = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=args.encoder_layer,
            num_heads=args.num_heads,
            return_weights=True
        )
        self.comp_layer = CompEncoder(
            hidden_size=representation_size, 
            return_weights=True, 
            num_layers=args.comp_layer
        )
        self.object_relation = ObjectRelation(
            hidden_size=256,
            num_layers=1
        )
        

    def get_pairwise_spatial(self, b1, b2, eps=1e-8):
        
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x, c1_y, c2_x, c2_y,
            # Relative box width and height
            b1_w, b1_h, b2_w, b2_h,
            # Relative box area
            b1_w * b1_h, b2_w * b2_h,
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)
        return torch.cat([f, torch.log(f + eps)], 1)


    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])


    def get_pairwise_dist(self, boxes1, boxes2):
        cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2.
        cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2.

        cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2.
        cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2.

        dist = ((cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)).sqrt()
        
        return dist


    def scale_box(self, boxes, shape):
        h, w = shape
        scaled_boxes = boxes.detach().clone()
        scaled_boxes[:, 0] /= w
        scaled_boxes[:, 1] /= h
        scaled_boxes[:, 2] /= w
        scaled_boxes[:, 3] /= h

        areas = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) * (scaled_boxes[:, 3] - scaled_boxes[:, 1])
        return scaled_boxes, areas


    def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """

        device = features.device
        global_features = self.global_feature_head(self.avg_pool(features).flatten(start_dim=1))

        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        pairwise_tokens_collated = []
        attn_maps_collated = []; nh_list = []; no_list = []
        original_labels = []

        for b_idx, props in enumerate(region_props):
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                pairwise_tokens_collated.append(torch.zeros(
                    0, self.representation_size,
                    device=device)
                )
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()

            # Compute spatial features
            # Reshape the spatial features
            scaled_boxes, areas = self.scale_box(boxes, image_shapes[b_idx])
            original_unary_tokens = unary_tokens # n x d
            
            # invalid_index = torch.arange(len(#torch.where(torch.logical_and(scores < self.args.score_thres, areas < 0.1))[0]
            # valid_index = torch.where(torch.logical_or(scores >= self.args.score_thres, areas>=0.1))[0]
            # print(invalid_index)
            # print("invalid:", len(invalid_index), "valid:", len(valid_index))
            retrieved_tokens = self.token_mixer.retrieve(labels, k=self.args.k, training=self.training) # n x c x d
            if retrieved_tokens is not None:
                unary_tokens_in = torch.cat([unary_tokens.clone().unsqueeze(1), retrieved_tokens], dim=1)
                unary_tokens_out = self.object_relation(unary_tokens_in)[:, 0, :]
                
                if not self.training:
                    invalid_index = torch.where(torch.logical_or(scores > self.args.score_thres, areas < 0.2))[0]
                    unary_tokens[invalid_index] = unary_tokens_out[invalid_index]
                else:
                    unary_tokens = unary_tokens_out
            else:
                pass

            if self.training:
                idx = torch.where(scores>self.args.score_thres)[0]
                if len(idx):
                    self.token_mixer.enqueue(original_unary_tokens[idx], labels[idx], areas[idx], scores[idx])


            pairwise_dist = self.get_pairwise_dist(boxes[x], boxes[y])
            pairwise_dist = pairwise_dist.reshape(n, n)

            unary_tokens, unary_attn = self.coop_layer(unary_tokens, pairwise_dist, scaled_boxes, x, y)

            pairwise_tokens = torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1)
            global_feature = global_features[b_idx, None].repeat(len(pairwise_tokens), 1)
            pairwise_tokens, pairwise_attn = self.comp_layer(pairwise_tokens + global_feature)

            pairwise_tokens_collated.append(pairwise_tokens)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            attn_maps_collated.append((unary_attn, pairwise_attn))
            no_list.append(n)
            nh_list.append(n_h)
            original_labels.append(labels)

        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)        
        logits = self.box_pair_predictor(pairwise_tokens_collated)

        return logits, prior_collated, \
            boxes_h_collated, boxes_o_collated, object_class_collated, \
            attn_maps_collated, nh_list, no_list, original_labels
