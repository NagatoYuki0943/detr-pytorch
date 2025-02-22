# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#-----------------------------#
#   对backbone的输出进行编码
#-----------------------------#
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        #-------------------------------------------------------------#
        #   src:                    [625, B, 256]
        #   src_mask:               None
        #   src_key_padding_mask:   [B, 625]
        #   pos:                    [625, B, 256] => ...(x6)... => [625, B, 256]
        #-------------------------------------------------------------#
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output

#-----------------------------#
#   自注意力
#-----------------------------#
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Self-Attention模块
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN模块
        # Implementation of Feedforward model
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model)

        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        Args:
            src (Tensor): x                                                                     [625, B, 256]
            src_mask (Optional[Tensor], optional): _description_. Defaults to None.             None
            src_key_padding_mask (Optional[Tensor], optional): _description_. Defaults to None. [B, 625]
            pos (Optional[Tensor], optional): _description_. Defaults to None.                  [625, B, 256]

        Returns:
            Tensor: [625, B, 256]
        """
        # q和k添加位置信息
        # [625, B, 256] + [625, B, 256] => [625, B, 256]
        q = k = self.with_pos_embed(src, pos)
        # 使用自注意力机制模块
        # [625, B, 256] => [625, B, 256]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # 添加FFN结构
        # [625, B, 256] => [625, B, 2048] => [625, B, 256]
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

#-----------------------------#
#   对backbone的输出进行编码
#-----------------------------#
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        #-----------------------------------------------------------#
        #   Decoder
        #   tgt:                    [100, B, 256]
        #   tgt_mask:               None
        #   memory_mask:            None
        #   tgt_key_padding_mask:   None
        #   memory:                 [625, B, 256]
        #   memory_key_padding_mask:[B, 625]
        #   pos:                    [625, B, 256]
        #   query_pos:              [100, B, 256] => ...(x6)... => [6, 100, B, 256]   6指的是TransformerDecoderLayer的6个layer的输出
        #-----------------------------------------------------------#

        # 中间层输出    [100, B, 256] * 6
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,   # [100, B, 256] => [100, B, 256]
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))  # 保存中间层的输出

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop() # 删除最后一个,添加norm最后一个的结果
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate) # [100, B, 256] * 6 => [6, 100, B, 256]    6指的是TransformerDecoderLayer的6个layer的输出

        return output.unsqueeze(0)

#-----------------------------#
#   自注意力+注意力
#-----------------------------#
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        # q自己做一个self-attention
        self.self_attn      = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # q、k、v联合做一个self-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN模块
        # Implementation of Feedforward model
        self.linear1        = nn.Linear(d_model, dim_feedforward)
        self.dropout        = nn.Dropout(dropout)
        self.linear2        = nn.Linear(dim_feedforward, d_model)

        self.norm1          = nn.LayerNorm(d_model)
        self.norm2          = nn.LayerNorm(d_model)
        self.norm3          = nn.LayerNorm(d_model)
        self.dropout1       = nn.Dropout(dropout)
        self.dropout2       = nn.Dropout(dropout)
        self.dropout3       = nn.Dropout(dropout)

        self.activation         = _get_activation_fn(activation)
        self.normalize_before   = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        Args:
            tgt (Tensor): shape like query_pos, init all zeros                                      [100, B, 256]
            memory (Tensor): encoder's output                                                       [625, B, 256]
            tgt_mask (Optional[Tensor], optional): _description_. Defaults to None.                 None
            memory_mask (Optional[Tensor], optional): _description_. Defaults to None.              None
            tgt_key_padding_mask (Optional[Tensor], optional): _description_. Defaults to None.     None
            memory_key_padding_mask (Optional[Tensor], optional): _description_. Defaults to None.  [B, 625]
            pos (Optional[Tensor], optional): position code. Defaults to None.                      [625, B, 256]
            query_pos (Optional[Tensor], optional): query_pos. Defaults to None.                    [100, B, 256]
                                                    query_pos的输入值为query_embed,应该为查询向量
                                                    相比之下tgt更像位置编码,query_pos更像查询向量
        Returns:
            Tensor: [6, 100, B, 256]   6指的是TransformerDecoderLayer的6个layer的输出
        """
        #---------------------------------------------#
        #   q自己做一个self-attention
        #---------------------------------------------#
        # q和k添加位置信息,是独立的位置编码
        # [100, B, 256] + [100, B, 256] => [100, B, 256]
        q = k = self.with_pos_embed(tgt, query_pos)
        # 3 * [100, B, 256] => [100, B, 256]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        #---------------------------------------------#
        #   q、k、v联合做一个attention,
        #   query是tgt+query_pos,key和value是memory
        #   q = [100, B, 256]
        #   k = [625, B, 256]
        #   v = [625, B, 256] => [100, B, 256]
        #---------------------------------------------#
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),   # 添加位置信息
                                   key=self.with_pos_embed(memory, pos),        # 添加位置信息,和q的位置编码不同
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        #---------------------------------------------#
        #   做一个FFN
        #---------------------------------------------#
        # [100, B, 256] => [100, B, 2048 ]=> [100, B, 256]
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # 定义用到的transformer的encoder层，然后定义使用到的norm层
        encoder_layer   = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm    = nn.LayerNorm(d_model) if normalize_before else None
        # 构建Transformer的Encoder，一共有6层
        self.encoder    = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 定义用到的transformer的decoder层，然后定义使用到的norm层
        decoder_layer   = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm    = nn.LayerNorm(d_model)
        # 构建Transformer的Decoder，一共有6层
        self.decoder    = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model    = d_model
        self.nhead      = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src (Tensor): x                 [B, 256, 25, 25]
            mask (Tensor): x'mask           [B, 25, 25]
            query_embed (Tensor): 查询向量   [100, 256]
            pos_embed (Tensor):   位置编码   [B, 256, 25, 25]

        Returns:
            Tensor: [6, B, 100, 256]  6指的是TransformerDecoderLayer的6个layer的输出
        """
        bs, c, h, w = src.shape
        # [B, 256, 25, 25] => [B, 256, 625] => [625, B, 256]
        src         = src.flatten(2).permute(2, 0, 1)
        # [B, 256, 25, 25] => [B, 256, 625] => [625, B, 256]
        pos_embed   = pos_embed.flatten(2).permute(2, 0, 1)
        # [100, 256] => [100, 1, 256] => [100, B, 256] decoder中的query
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # [B, 25, 25] => [B, 625]
        mask        = mask.flatten(1)

        #-----------------------------------------------------------#
        #   [100, B, 256] 创建了一个与query_embed一样shape的矩阵
        #-----------------------------------------------------------#
        tgt         = torch.zeros_like(query_embed)

        #-----------------------------------------------------------#
        #   Encoder
        #   src:                    [625, B, 256]
        #   src_key_padding_mask:   [B, 625]
        #   pos:                    [625, B, 256] => [625, B, 256]
        #-----------------------------------------------------------#
        memory      = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        #-----------------------------------------------------------#
        #   Decoder
        #   tgt:                    [100, B, 256]
        #   memory:                 [625, B, 256]
        #   memory_key_padding_mask:[B, 625]
        #   pos:                    [625, B, 256]
        #   query_pos:              [100, B, 256] => [6, 100, B, 256]   6指的是TransformerDecoderLayer的6个layer的输出
        #-----------------------------------------------------------#
        hs          = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)

        # [6, 100, B, 256] => [6, B, 100, 256], [625, B, 256] => [B, 256, 625] => [B, 256, 25, 25]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

def build_transformer(hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048, enc_layers=6, dec_layers=6, pre_norm=True):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )

