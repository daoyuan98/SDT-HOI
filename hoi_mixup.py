import torch
import torch.nn as nn 
import numpy as np

class TokenMixer(nn.Module):

    def __init__(self, num_classes=80, dim=256, buffer_size=128):
        super().__init__()
        self.buffer = nn.Parameter(torch.zeros((num_classes, buffer_size, dim)), requires_grad=False)
        self.areas  = nn.Parameter(torch.zeros((num_classes, buffer_size)), requires_grad=False)
        self.scores = nn.Parameter(torch.zeros_like(self.areas), requires_grad=False)
        self.pointer = torch.zeros(num_classes)
        self.buffer_size = buffer_size

    def enqueue(self, tokens, labels, areas, scores):
        # self.buffer_tokens
        for token, token_label, area, score in zip(tokens, labels, areas, scores):
            token_label = token_label.item()
            if self.pointer[token_label] < self.buffer_size - 1:
                self.pointer[token_label] += 1
                idx = self.pointer[token_label].long()
                self.buffer[token_label, idx] = token.clone()
                self.areas[token_label,  idx] = area
                self.scores[token_label, idx] = score
            else:
                i = np.random.randint(self.buffer_size)
                self.buffer[token_label, i, :]  = token.clone()
                self.areas[token_label, i]      = area
                self.scores[token_label, i]     = score


    def mix_tokens(self, tokens, token_labels, alpha=0.5):
        counts = self.pointer[token_labels]
        invalid = torch.where(counts == 0)[0]

        device = tokens.device
        tokens2mix = self.buffer[token_labels, 0, :].to(device)

        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)

        mixed_tokens = tokens2mix
        mixed_tokens[invalid] = tokens[invalid]
        return mixed_tokens

    def retrieve(self, obj_labels, k, training):
        if training:
            counts = self.pointer[obj_labels]
        else:
            counts = torch.zeros(len(obj_labels))
            for oi, obj in enumerate(obj_labels):
                for i in range(self.buffer_size):
                    if self.areas[obj][i] == 0:
                        break
                counts[oi] = i - 1

        valid = (counts > k).all()
        if not valid:
            return None
        else:
            topv, topi = torch.topk(self.areas[obj_labels], k, dim=1)
            r_list = []
            for i, idx in enumerate(topi):
                r_list.append(self.buffer[obj_labels[i], idx])
            if len(r_list):
                return torch.stack(r_list).to(obj_labels.device)
            else:
                return None

