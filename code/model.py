import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from math import sqrt
from torch.autograd import Variable
from torch import nn, einsum
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout,style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x):
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)

                x = rearrange(x, '1 b (n d) -> b n d', n = n)

        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        return x


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = x.float()
        x = self.layers(x)
        return x

class Contrast_MultiHeadSelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int
    num_heads: int
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(Contrast_MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x, y):
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        # x = x.to(torch.float32)
        # y = y.to(torch.float32)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(y).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist_a = 1 - dist # contrastive weight
        dist_a = torch.softmax(dist_a, dim=-1)
        c_att = torch.matmul(dist_a, v)  # batch, nh, n, dv
        c_att = c_att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        dist = torch.softmax(dist, dim=-1)
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return c_att,att

class ModVAR(nn.Module):
  def __init__(self):
    super(ModVAR,self).__init__()
    self.output_dim = 128
    self.rank = 4

    self.fc_bert1 = nn.Linear(768,768)
    self.fc_bert2 = nn.Linear(768,128)
    self.dna_cmattn = Contrast_MultiHeadSelfAttention(768,300,768,num_heads = 3)
    self.contrast_ln = nn.LayerNorm(normalized_shape=[768],eps=1e-6)
    self.dna_ln = nn.LayerNorm(normalized_shape=[128],eps=1e-6)

    self.tab_ln1 = nn.LayerNorm(normalized_shape=[226,4],eps = 1e-6)
    # self.tab_ln1 = nn.LayerNorm(normalized_shape=[143,4],eps = 1e-6) #remove cancer specific fea
    self.tab_ln2 = nn.LayerNorm(normalized_shape=[128],eps = 1e-6)

    self.tab_fc1 = nn.Linear(4*226,226)
    # self.tab_fc1 = nn.Linear(4*143,226) #remove cancer specific fea
    self.tab_fc2 = nn.Linear(226,128)
    self.tab_f = nn.Flatten()

    #aa 3d
    self.aa_fc1 = nn.Linear(1280,960)
    self.aa_fc2 = nn.Linear(960,512)
    self.aa_fc3 = nn.Linear(512,128)
    self.aa_cmattn = Contrast_MultiHeadSelfAttention(1280,960,1280,num_heads = 4)
    self.aa_contrastln =  nn.LayerNorm(normalized_shape=[1280],eps=1e-6)
    self.aa_ln = nn.LayerNorm(normalized_shape=[128],eps=1e-6)

    self.seq_factor = Parameter(torch.Tensor(self.rank, 128 + 1, self.output_dim))
    self.aa_factor = Parameter(torch.Tensor(self.rank, 128 + 1, self.output_dim))
    self.tab_factor = Parameter(torch.Tensor(self.rank, 128 + 1, self.output_dim))
    self.tool_factor = Parameter(torch.Tensor(self.rank,  4 + 1, self.output_dim))

    self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
    self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

    # init teh factors
    xavier_normal_(self.seq_factor)
    xavier_normal_(self.aa_factor)
    xavier_normal_(self.tool_factor)
    xavier_normal_(self.tab_factor)
    xavier_normal_(self.fusion_weights)
    self.fusion_bias.data.fill_(0)

    self.fc1_ = nn.Linear(128,512)
    self.fc2_ = nn.Linear(512,256)
    self.fc3_ = nn.Linear(256,128)
    self.fc4_ = nn.Linear(128,64)
    self.fc5_ = nn.Linear(64,1)


  def forward(self,dna_x,aa_x,tool_x=None,tab_x =None,dna_switch = True,aa_switch = True,tab_switch = True):
    if dna_switch:
        half_size = int(dna_x.shape[1]/2)
        splits = torch.split(dna_x,(half_size,half_size),dim = 1)
        ref_x = splits[0]
        alt_x = splits[1]
        ref_x = torch.reshape(ref_x,(ref_x.shape[0],1,ref_x.shape[1]))
        alt_x = torch.reshape(alt_x,(alt_x.shape[0],1,alt_x.shape[1]))
        x_contrast,x_att = self.dna_cmattn(ref_x,alt_x)
        x_contrast = torch.squeeze(x_contrast)
        dna_x = self.contrast_ln(x_contrast)
        dna_x = self.fc_bert1(dna_x)
        dna_x = self.fc_bert2(dna_x)
        dna_x = self.dna_ln(dna_x)

    #aa 3d
    if aa_switch:
        half_size = int(aa_x.shape[1]/2)
        splits = torch.split(aa_x,(half_size,half_size),dim = 1)
        ref_aax = splits[0]
        alt_aax = splits[1]
        ref_aax = torch.reshape(ref_aax,(ref_aax.shape[0],1,ref_aax.shape[1]))
        alt_aax = torch.reshape(alt_aax,(alt_aax.shape[0],1,alt_aax.shape[1]))
        aa_x_contrast,aa_x_att = self.aa_cmattn(ref_aax,alt_aax)
        aa_x_contrast = torch.squeeze(aa_x_contrast)
        aa_x = self.aa_contrastln(aa_x_contrast)
        aa_x = self.aa_fc1(aa_x)
        aa_x = self.aa_fc2(F.relu(aa_x))
        aa_x = self.aa_fc3(F.relu(aa_x))
        aa_x = self.aa_ln(aa_x)

    #tabular
    if tab_switch:
        #tab_pretrain
        # tab_x = tab_x.to(torch.float32)
        tab_x = self.tab_ln1(tab_x)
        tab_x = self.tab_f(tab_x)
        tab_x = self.tab_fc1(tab_x)
        tab_x = F.relu(tab_x)
        tab_x = self.tab_fc2(tab_x)
        tab_x = self.tab_ln2(tab_x)

    #LMF
    if dna_x.is_cuda:
        DTYPE = torch.cuda.FloatTensor
    else:
        DTYPE = torch.FloatTensor

    if len(dna_x.shape) == 1:
        dna_x = torch.unsqueeze(dna_x,0)
        aa_x = torch.unsqueeze(aa_x,0)

    batch_size = dna_x.shape[0]

    if dna_switch:
        _seq_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), dna_x), dim=1)
    if aa_switch:
        _aa_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), aa_x), dim=1)

    _tool_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), tool_x), dim=1)

    if tab_switch:
        _tab_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), tab_x), dim=1)
    if dna_switch:
        fusion_seq = torch.matmul(_seq_h, self.seq_factor)
    if aa_switch:
        fusion_aa = torch.matmul(_aa_h, self.aa_factor)
    if tab_switch:
        fusion_tab = torch.matmul(_tab_h, self.tab_factor)

    # _tool_h = _tool_h.to(torch.float32)
    fusion_tool = torch.matmul(_tool_h, self.tool_factor)

    first = True
    if dna_switch:
        fusion_zy = fusion_seq
        first = False
    if aa_switch:
        if not first:
            fusion_zy = fusion_zy * fusion_aa
        else:
            fusion_zy = fusion_aa
            first = False
    if tab_switch:
        if not first:
            fusion_zy = fusion_zy * fusion_tab
        else:
            fusion_zy = fusion_tab
            first = False
    if not first:
        fusion_zy = fusion_zy * fusion_tool
    else:
        fusion_zy = fusion_tool
    output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
    output = output.view(-1, self.output_dim)

    x = self.fc1_(output)
    x = F.relu(x)
    x = torch.sigmoid(x)
    x = self.fc2_(x)
    x = F.relu(x)
    x = self.fc3_(x)
    x = F.relu(x)
    x = self.fc4_(x)
    x = F.relu(x)
    res = self.fc5_(x)
    return torch.sigmoid(res)
