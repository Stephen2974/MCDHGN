import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from concurrent.futures import ThreadPoolExecutor
import dgl
from dgl.nn.pytorch import GATConv, GATv2Conv, GraphConv, GCN2Conv
from dgl.sampling import RandomWalkNeighborSampler


def assign_metapath_devices(num_meta_paths, device_ids, edge_counts=None, primary_device_id=0):
    """
    Assign each metapath index to a CUDA device.

    When >=2 GPUs are available, keep ALL metapath experts off the primary GPU
    (primary holds features / semantic fusion / classifier). Load-balance the
    remaining GPUs by edge count.
    """
    if device_ids is None or len(device_ids) == 0:
        return None
    devices = [
        torch.device(f"cuda:{i}" if not str(i).startswith("cuda") else i)
        for i in device_ids
    ]
    primary = torch.device(f"cuda:{primary_device_id}")

    # Prefer worker GPUs only; fall back to all devices if primary is the only one.
    worker_idx = [j for j, d in enumerate(devices) if d != primary]
    if not worker_idx:
        worker_idx = list(range(len(devices)))
    workers = [devices[j] for j in worker_idx]

    if edge_counts is None:
        return [workers[i % len(workers)] for i in range(num_meta_paths)]

    order = sorted(range(num_meta_paths), key=lambda i: edge_counts[i], reverse=True)
    load = [0] * len(workers)
    assignment = [None] * num_meta_paths
    for idx in order:
        j = min(range(len(workers)), key=lambda k: load[k])
        assignment[idx] = workers[j]
        load[j] += int(edge_counts[idx])
    return assignment


def place_metapath_graphs(gs, metapath_devices, primary_device=None):
    """
    Move each metapath block to its assigned device.
    If metapath_devices is None, move all graphs to primary_device (legacy behavior).
    """
    if metapath_devices is None:
        if primary_device is None:
            return gs
        return [g.to(primary_device) for g in gs]
    assert len(gs) == len(metapath_devices)
    return [g.to(metapath_devices[i]) for i, g in enumerate(gs)]


class _ChunkedGATv2Scores(torch.autograd.Function):
    """e_ij = a^T LeakyReLU(Wh_i + Wh_j), computed in edge chunks (low peak memory)."""

    @staticmethod
    def forward(ctx, feat_src, feat_dst, attn, src, dst, slope, chunk_size):
        ctx.slope = float(slope)
        ctx.chunk_size = int(chunk_size)
        ctx.save_for_backward(feat_src, feat_dst, attn, src, dst)
        num_edges = src.shape[0]
        scores = feat_src.new_empty(num_edges, attn.shape[1], 1)
        chunk = ctx.chunk_size
        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            ee = feat_src[src[start:end]] + feat_dst[dst[start:end]]
            ee = F.leaky_relu(ee, ctx.slope)
            scores[start:end] = (ee * attn).sum(dim=-1, keepdim=True)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        feat_src, feat_dst, attn, src, dst = ctx.saved_tensors
        slope = ctx.slope
        chunk = ctx.chunk_size
        grad_feat_src = torch.zeros_like(feat_src)
        grad_feat_dst = torch.zeros_like(feat_dst)
        grad_attn = torch.zeros_like(attn)
        num_edges = src.shape[0]
        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            s_idx = src[start:end]
            d_idx = dst[start:end]
            ee = feat_src[s_idx] + feat_dst[d_idx]
            ee_act = F.leaky_relu(ee, slope)
            gs = grad_scores[start:end]  # (C, H, 1)
            grad_attn += (gs * ee_act).sum(dim=0, keepdim=True)
            # grad w.r.t. pre-activation: attn * gs, then LeakyReLU'
            # Inplace scale negatives; no extra (C,H,D) clone/where buffer.
            grad_ee = gs * attn
            grad_ee[ee < 0] *= slope
            grad_feat_src.index_add_(0, s_idx, grad_ee)
            grad_feat_dst.index_add_(0, d_idx, grad_ee)
            del ee, ee_act, gs, grad_ee
        return grad_feat_src, grad_feat_dst, grad_attn, None, None, None, None


class _ChunkedAttnAggregate(torch.autograd.Function):
    """rst[d] += feat_src[s] * alpha_sd, in edge chunks (avoids full (E,H,D) messages)."""

    @staticmethod
    def forward(ctx, feat_src, alpha, src, dst, num_dst, chunk_size):
        ctx.chunk_size = int(chunk_size)
        ctx.num_dst = int(num_dst)
        ctx.save_for_backward(feat_src, alpha, src, dst)
        num_edges = src.shape[0]
        rst = feat_src.new_zeros(ctx.num_dst, feat_src.shape[1], feat_src.shape[2])
        chunk = ctx.chunk_size
        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            msg = feat_src[src[start:end]] * alpha[start:end]
            rst.index_add_(0, dst[start:end], msg)
        return rst

    @staticmethod
    def backward(ctx, grad_rst):
        feat_src, alpha, src, dst = ctx.saved_tensors
        chunk = ctx.chunk_size
        grad_feat_src = torch.zeros_like(feat_src)
        grad_alpha = torch.zeros_like(alpha)
        num_edges = src.shape[0]
        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            s_idx = src[start:end]
            d_idx = dst[start:end]
            fs = feat_src[s_idx]
            a = alpha[start:end]
            g = grad_rst[d_idx]
            grad_feat_src.index_add_(0, s_idx, g * a)
            grad_alpha[start:end] = (g * fs).sum(dim=-1, keepdim=True)
        return grad_feat_src, grad_alpha, None, None, None, None


class MemEfficientGATv2Conv(GATv2Conv):
    """
    Memory-efficient GATv2 (same math as DGL GATv2Conv).

    Avoids materializing full (E, H, D) for both attention scores and
    message passing by computing them in edge chunks.
    """

    def __init__(self, *args, edge_chunk_size=65536, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(self, graph, feat, get_attention=False):
        from dgl.base import DGLError
        from dgl.ops import edge_softmax

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "Adding self-loop or set allow_zero_in_degree=True."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]

            src, dst = graph.edges()
            e = _ChunkedGATv2Scores.apply(
                feat_src,
                feat_dst,
                self.attn,
                src,
                dst,
                self.leaky_relu.negative_slope,
                self.edge_chunk_size,
            )
            # (E, H, 1)
            alpha = self.attn_drop(edge_softmax(graph, e))
            num_dst = graph.number_of_dst_nodes()
            rst = _ChunkedAttnAggregate.apply(
                feat_src, alpha, src, dst, num_dst, self.edge_chunk_size
            )

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, alpha
            return rst

#自写注意力层，但实际用的注意力层是dgl.nn.pytorch.GATConv
class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation, feat_drop_rate, attn_drop_rate):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop_rate)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.activation = activation

        for i in range(num_heads):
            self.heads.append(self.build_head())

        self.weight = nn.Parameter(torch.Tensor(in_feats, num_heads * out_feats))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)

    def build_head(self):
        return nn.Linear(self.in_feats, self.out_feats, bias=False)

    def forward(self, graph, inputs):
        h = self.feat_drop(inputs)
        outputs = []
        for i, attention_head in enumerate(self.heads):
            # compute attention coefficients
            a = self.attn_drop(attention_head(h))

            # compute softmax
            graph.ndata['attn'] = F.softmax(torch.sum(a * h, dim=-1) / (self.out_feats ** 0.5), dim=0)
            # compute the linear combination of neighborhood features
            h_prime = dgl.prop_nodes_topo(graph, message_func=dgl.function.u_mul_e('h', 'attn'),
                                          reduce_func=dgl.function.sum('m', 'neigh'))
            outputs.append(h_prime)

        h = torch.cat(outputs, dim=-1)
        h = torch.matmul(h, self.weight)

        return self.activation(h)
#元路径怎么融合
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1),beta[0]  # (N, D * K)


class MySampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_list.append(
                RandomWalkNeighborSampler(
                    G=g,
                    num_traversals=1,
                    termination_prob=0,
                    num_random_walks=num_neighbors,
                    num_neighbors=num_neighbors,
                    metapath=metapath,
                )
            )

    def sample_blocks(self, seeds):
        block_list = []
        for sampler in self.sampler_list:
            frontier = sampler(seeds)
            # add self loop
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(seeds.clone().detach() , seeds.clone().detach() )
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim/2), bias=True)
        self.fc2 = nn.Linear(int(dim/2), 2, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class MyLayer(nn.Module):
    """
    Model layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    use_gatv2 : if True, use GATv2Conv; otherwise use original GATConv
    share_weights : GATv2 only; share W_left and W_right when True
    use_checkpoint : recompute GATv2 edge feats in backward to save GPU memory
                     (recommended for GATv2; default True when use_gatv2=True)
    metapath_devices : optional list[device], one device per metapath (model parallel)
    primary_device : device for semantic attention fusion / output
    parallel_metapaths : if True, overlap different GPUs in waves (faster, riskier);
                         default False (sequential, safer for GATv2 memory)

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs (already placed on metapath_devices if multi-GPU)
    h : tensor
        Input features (on primary_device)

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self,
        num_meta_paths,
        in_size,
        out_size,
        layer_num_heads,
        dropout,
        use_gatv2=False,
        share_weights=False,
        use_checkpoint=None,
        metapath_devices=None,
        primary_device=None,
        parallel_metapaths=None,
    ):
        super(MyLayer, self).__init__()

        # One GAT / GATv2 layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.use_gatv2 = use_gatv2
        # MemEfficientGATv2 already chunked; outer checkpoint often hurts fragmentation.
        if use_checkpoint is None:
            use_checkpoint = False
        self.use_checkpoint = use_checkpoint
        self.metapath_devices = metapath_devices
        self.primary_device = primary_device
        # Default False: safer for GATv2 memory (one (E,H,D) live globally).
        # Set True to overlap different GPUs in waves (faster, higher peak risk).
        if parallel_metapaths is None:
            parallel_metapaths = False
        self.parallel_metapaths = parallel_metapaths
        for i in range(num_meta_paths):
            if use_gatv2:
                # GATv2 enhancement (chunked scores so 256x4 fits on 11GB)
                self.gat_layers.append(
                    MemEfficientGATv2Conv(
                        in_size,
                        out_size,
                        layer_num_heads,
                        dropout,
                        dropout,
                        activation=F.elu,
                        share_weights=share_weights,
                        edge_chunk_size=65536,
                    )
                )
            else:
                # Original MCDHGN path: GATConv
                self.gat_layers.append(
                    GATConv(
                        in_size,
                        out_size,
                        layer_num_heads,
                        dropout,
                        dropout,
                        activation=F.elu,
                    )
                )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def place_devices(self, metapath_devices=None, primary_device=None):
        """Put each GAT/GATv2 expert on its metapath GPU; fusion on primary."""
        if metapath_devices is not None:
            self.metapath_devices = metapath_devices
        if primary_device is not None:
            self.primary_device = primary_device
        if self.metapath_devices is None:
            if self.primary_device is not None:
                self.to(self.primary_device)
            return self
        assert len(self.metapath_devices) == len(self.gat_layers)
        for i, layer in enumerate(self.gat_layers):
            layer.to(self.metapath_devices[i])
        prim = self.primary_device or self.metapath_devices[0]
        self.primary_device = prim
        self.semantic_attention.to(prim)
        return self

    def _run_one_metapath(self, i, g, h_primary, return_attention):
        layer = self.gat_layers[i]
        if self.metapath_devices is not None:
            dev = self.metapath_devices[i]
            h = h_primary.to(dev, non_blocking=True)
        else:
            dev = h_primary.device
            h = h_primary

        need_attn = return_attention
        if self.use_checkpoint and self.training and not need_attn:
            def _run(feat, _layer=layer, _g=g):
                return _layer(_g, feat)

            embeddings = checkpoint(_run, h, use_reentrant=False)
            alpha_weight = None
        elif need_attn:
            embeddings, alpha_weight = layer(g, h, get_attention=True)
        else:
            embeddings = layer(g, h, get_attention=False)
            alpha_weight = None

        prim = self.primary_device or embeddings.device
        embeddings = embeddings.flatten(1).to(prim, non_blocking=True)
        if alpha_weight is not None:
            alpha_weight = alpha_weight.to(prim, non_blocking=True)
        return embeddings, alpha_weight

    def forward(self, gs, h, return_attention=False):
        n = len(gs)
        semantic_embeddings = [None] * n
        gat_attention_weights = [None] * n if return_attention else []

        def _compute(i):
            emb, alpha = self._run_one_metapath(i, gs[i], h, return_attention)
            return i, emb, alpha

        if self.parallel_metapaths and self.metapath_devices is not None:
            # One wave: at most one metapath per physical GPU.
            device_to_indices = {}
            for i, dev in enumerate(self.metapath_devices):
                device_to_indices.setdefault(str(dev), []).append(i)
            max_len = max(len(v) for v in device_to_indices.values())
            for wave in range(max_len):
                jobs = [
                    indices[wave]
                    for indices in device_to_indices.values()
                    if wave < len(indices)
                ]
                if len(jobs) == 1:
                    results = [_compute(jobs[0])]
                else:
                    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                        results = list(ex.map(_compute, jobs))
                for i, emb, alpha in results:
                    semantic_embeddings[i] = emb
                    if return_attention:
                        gat_attention_weights[i] = alpha
        else:
            for i in range(n):
                emb, alpha = self._run_one_metapath(i, gs[i], h, return_attention)
                semantic_embeddings[i] = emb
                if return_attention:
                    gat_attention_weights[i] = alpha
                # Do NOT call torch.cuda.empty_cache() here: mid-forward cache
                # flushes often surface async CUDA errors and can abort the kernel.

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)
        gat_out, beta = self.semantic_attention(semantic_embeddings)
        return gat_out, gat_attention_weights, beta  # (N, D * K)


class MCDHGN(nn.Module):
    def __init__(
        self,
        num_meta_paths,
        in_size,
        hidden_size,
        out_size,
        num_heads,
        dropout,
        use_gatv2=False,
        share_weights=False,
        use_checkpoint=None,
        metapath_device_ids=None,
        primary_device_id=0,
        edge_counts=None,
        parallel_metapaths=None,
    ):
        super(MCDHGN, self).__init__()

        # use_gatv2=False -> original GATConv (default, unchanged behavior)
        # use_gatv2=True  -> GATv2Conv enhancement experiment
        # metapath_device_ids e.g. [0,1,2]: place different metapath experts on different GPUs
        self.use_gatv2 = use_gatv2
        if use_checkpoint is None:
            use_checkpoint = False
        self.use_checkpoint = use_checkpoint

        if metapath_device_ids is not None:
            self.metapath_devices = assign_metapath_devices(
                num_meta_paths,
                metapath_device_ids,
                edge_counts=edge_counts,
                primary_device_id=primary_device_id,
            )
            self.primary_device = torch.device(f"cuda:{primary_device_id}")
        else:
            self.metapath_devices = None
            self.primary_device = None

        layer_kwargs = dict(
            use_gatv2=use_gatv2,
            share_weights=share_weights,
            use_checkpoint=use_checkpoint,
            metapath_devices=self.metapath_devices,
            primary_device=self.primary_device,
            parallel_metapaths=parallel_metapaths,
        )
        self.layers = nn.ModuleList()
        self.layers.append(
            MyLayer(
                num_meta_paths,
                in_size,
                hidden_size,
                num_heads[0],
                dropout,
                **layer_kwargs,
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                MyLayer(
                    num_meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                    **layer_kwargs,
                )
            )
        dim = hidden_size*num_heads[-1]
        
 
        self.predict = linear_module(dim=dim)
        if self.metapath_devices is not None:
            self.place_devices()
        #self.predict=nn.Linear(hidden_size * num_heads[-1], out_size)
        #self.conv1 = GraphConv(dim, 128, norm='both', weight=True, bias=True)
        #self.conv2 = GraphConv(128,2)
        #self.conv = GCN2Conv(dim,layer=1,alpha = 0.8,project_initial_features=True, allow_zero_in_degree=True)

    def place_devices(self, metapath_devices=None, primary_device=None):
        if metapath_devices is not None:
            self.metapath_devices = metapath_devices
        if primary_device is not None:
            self.primary_device = primary_device
        for layer in self.layers:
            layer.place_devices(self.metapath_devices, self.primary_device)
        if self.primary_device is not None:
            self.predict.to(self.primary_device)
        return self

    def forward(self, g, h, return_attention=False):
        alpha, beta = None, None
        if self.primary_device is not None and h.device != self.primary_device:
            h = h.to(self.primary_device)
        for i, gnn in enumerate(self.layers):
            # Only the first layer's attention is returned (same as before).
            need_attn = return_attention and (i == 0)
            h, alpha_i, beta_i = gnn(g, h, return_attention=need_attn)
            if i == 0:
                alpha, beta = alpha_i, beta_i
            else:
                beta = beta_i
        
        return h,self.predict(h),alpha,beta
