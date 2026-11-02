"""Validate actual DGL graph and cached blocks on local CPU or the training server.

This command does not train or select models. A nonzero exit blocks use of the inputs.
"""
import argparse
import json
import os
import platform
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, VALIDATION_CODE, log, now, sha256, verify_frozen, write_json
from .audit import aligned_features, load_pickle, ordered_genes


def validate(root, out, output):
    os.environ.setdefault('DGLBACKEND', 'pytorch')
    import dgl
    import torch
    verify_frozen(root, out)
    mapping = load_pickle(root / 'data/step1/gene_toids.pkl')
    expected = aligned_features(mapping, load_pickle(root / 'data/step1/gene_feature_bio.pkl'))
    graphs, labels = dgl.load_graphs(str(root / 'data/network/hetero/new_9nodes_graph.bin'))
    if len(graphs) != 1:
        raise ValueError('Expected exactly one training heterograph')
    graph = graphs[0]
    actual = graph.nodes['Gene'].data['feature'].cpu()
    if graph.num_nodes('Gene') != len(mapping) or actual.shape != expected.shape:
        raise ValueError('Graph/map/feature dimensions disagree')
    if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
        raise ValueError('Graph features differ from explicitly aligned original features')
    paths = [['gg'], ['ga','ag'], ['gb','bg'], ['gc','cg'], ['gd','dg'], ['ge','eg'], ['gf','fg'], ['gh','hg'], ['gi','ig']]
    from run_ablation import global_graph
    import run_ablation
    run_ablation.torch = torch
    run_ablation.dgl = dgl
    summaries = []
    for i, path in enumerate(paths):
        filename = root / f'Intermediate/blocks/{i}gs.bin'
        block = load_pickle(filename).to('cpu')
        if block.is_block:
            for side, data, count in [('src',block.srcdata,block.num_src_nodes()), ('dst',block.dstdata,block.num_dst_nodes())]:
                if dgl.NID not in data:
                    raise ValueError(f'Block {i} missing {side} original node IDs')
                ids = data[dgl.NID].cpu()
                if ids.ndim!=1 or len(ids)!=count or ids.dtype not in (torch.int32,torch.int64):
                    raise ValueError(f'Block {i}: invalid {side} ID tensor')
                if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(range(len(mapping))):
                    raise ValueError(f'Block {i}: {side} IDs do not cover all genes uniquely')
        full = global_graph(block, len(mapping))
        src, dst = full.edges(order='eid')
        if full.num_edges()!=block.num_edges():
            raise ValueError(f'Block {i}: edges lost during ID conversion')
        # Structural coverage is verified. Historical RNG provenance cannot be recovered from the cache.
        # Ensure every non-self sampled pair is reachable through the named raw metapath.
        import scipy.sparse as sp
        relation = None
        for etype in path:
            src_type, _, dst_type = graph.to_canonical_etype(etype)
            a,b = graph.edges(etype=etype)
            matrix = sp.csr_matrix((__import__('numpy').ones(len(a),dtype=bool),(a.numpy(),b.numpy())),
                                   shape=(graph.num_nodes(src_type),graph.num_nodes(dst_type)))
            relation = matrix if relation is None else relation @ matrix
        mask = src != dst
        nonself_src, nonself_dst = src[mask].numpy().copy(), dst[mask].numpy().copy()
        # The sampler creates neighbor -> seed edges; walk reachability runs seed -> neighbor.
        invalid = int((~__import__('numpy').asarray(relation[nonself_dst, nonself_src]).reshape(-1).astype(bool)).sum())
        summaries.append(dict(index=i,metapath='-'.join(path),cache_sha256=sha256(filename),is_block=bool(block.is_block),
                              nodes=full.num_nodes(),edges=full.num_edges(),self_loops=int((src==dst).sum()),
                              zero_in_degree_nodes=int((full.in_degrees()==0).sum()),
                              pairs_not_reachable_by_metapath=invalid,
                              degree_min=int(full.in_degrees().min()),degree_max=int(full.in_degrees().max())))
    report = dict(created_at=now(),status='passed' if not any(r['pairs_not_reachable_by_metapath'] for r in summaries) else 'failed',
                  baseline_manifest_sha256=sha256(out/'baseline_manifest.json'),
                  validation_code_sha256={name:sha256(ROOT/name) for name in VALIDATION_CODE},
                  host_platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,dgl=dgl.__version__,
                  num_gene_nodes=len(mapping),feature_shape=list(actual.shape),graph_features_exact_match=True,
                  maximum_feature_difference=float((actual-expected).abs().max()),blocks=summaries,
                  limitations=['Historical sampling seed and neighbor budget are not proven by cached graphs.',
                               'This is a structural/data audit, not server training or a test performance evaluation.'])
    write_json(output,report)
    log('06 实际图与九份缓存核验',report['status'],json.dumps(report,ensure_ascii=False),out)
    if report['status']!='passed':
        raise ValueError('Cached edges do not match their designated metapaths; see report')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    output=args.report or args.out/'graph_validation_local.json'
    try:
        validate(args.root.resolve(),args.out.resolve(),output.resolve())
    except Exception as exc:
        write_json(output.with_name(output.stem+'_error.json'),dict(status='failed',error_type=type(exc).__name__,error=str(exc),created_at=now()))
        log('06 实际图与九份缓存核验','failed',f'{type(exc).__name__}: {exc}',args.out)
        raise


if __name__=='__main__':
    main()
