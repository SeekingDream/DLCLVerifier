"""Test actual pretrained utils.py models, without downloading datasets.

PYTHONPATH=src python test/test_reconstruction_models.py --tasks 0 1 2 3
CUDA_VISIBLE_DEVICES=7 python test/test_reconstruction_models.py --tasks 2 \
    --device cuda --compiler triton --image-size 64

With --compiler triton|tilelang the program is the compiled kernel path: its GEMM-shaped calls
are lowered to explicit FMA chains (dlclv.lowering) and the rest is preserved (eager-kernel) or,
with --mode primitive --probe-kernels, lowered from the behaviour of the library kernels.
With --compiler tvm every kernel is executed from its scheduled TIR (dlclv.tir_lowering):
CUDA_VISIBLE_DEVICES=7 python test/test_reconstruction_models.py --tasks 2 --device cuda \
    --compiler tvm --image-size 64
Weights must be cached unless --allow-download is passed. Agreement is empirical.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))


IMAGE_SIZE = 224


def inputs_for(task, model, generator):
    import torch
    import utils
    if task.dataset == 'imagenet_val':
        return utils.image_domain(IMAGE_SIZE).sample(1, generator)
    if task.model == 'whisper_tiny_encoder':
        cfg = model.encoder.config
        return (torch.randn(1, cfg.num_mel_bins, cfg.max_source_positions * 2,
                            generator=generator),)
    cfg = model.net.config
    domain = utils.token_domain(cfg.vocab_size, task.seq_len,
                                token_type=task.model == 'bert_large_squad')
    inputs = list(domain.sample(1, generator))
    inputs[1].fill_(1)
    if len(inputs) == 3:
        inputs[2].zero_()
    return tuple(inputs)


def run_task(task_id, samples, mode='eager-kernel', image_edges=False, device='cpu',
             compiler='eager', options=None, probe_kernels=False):
    import torch
    import utils
    from dlclv.reconstruction import reconstruct, UnsupportedReconstruction, ExecutionMismatch
    task = utils.get_task(task_id)
    row = {'task': task_id, 'model': task.model, 'mode': mode, 'compiler': compiler,
           'options': options or {}}

    def draw(generator):
        return tuple(x.to(device) for x in inputs_for(task, model, generator))

    try:
        model = utils.load_model(task_id, device=device)
        generator = torch.Generator().manual_seed(20261031)
        inputs = draw(generator)
        program = model
        if compiler != 'eager':
            program = utils.load_compiler(compiler, **(options or {}))(model, inputs)
    except Exception as exc:
        return dict(row, status='BLOCKED', stage='load_model/inputs/compile',
                    reason=f'{type(exc).__name__}: {exc}')
    try:
        reconstructed = reconstruct(program, inputs, mode=mode, probe_kernels=probe_kernels)
    except (UnsupportedReconstruction, ExecutionMismatch) as exc:
        return dict(row, status='UNSUPPORTED', stage='reconstruction', reason=str(exc)[:2000])
    except Exception as exc:
        return dict(row, status='ERROR', stage='reconstruction',
                    reason=f'{type(exc).__name__}: {exc}')
    try:
        cases = [inputs] + [draw(generator) for _ in range(samples - 1)]
        if image_edges and task.dataset == 'imagenet_val':
            cases.extend((torch.full_like(inputs[0], value),) for value in (0., 0.5, 1.))
            checker = torch.arange(IMAGE_SIZE).view(-1, 1) + torch.arange(IMAGE_SIZE)
            cases.append(((checker % 2).to(inputs[0]).expand_as(inputs[0]).contiguous(),))
        evidence = reconstructed.compare(program, cases)
        report = reconstructed.report
        lowered = report['lowered_kernels']
        return dict(row, status='EMPIRICAL_AGREEMENT' if evidence['bitwise_equal'] else 'MISMATCH',
                    evidence=evidence, assumptions=report['assumptions'],
                    preserved_kernels=report['preserved_kernels'],
                    lowered_kernels=len(lowered), lowering_levels=report['lowering_levels'],
                    lowering_failures=report['lowering_failures'],
                    lowered_examples=[{k: str(v)[:400] for k, v in op.items()}
                                      for op in lowered[:2]],
                    lowered_primitive_ops=report['lowered_primitive_ops'],
                    universal_certificate=report['universal_certificate'])
    except Exception as exc:
        return dict(row, status='ERROR', stage='comparison', reason=f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--mode', choices=['primitive', 'eager-kernel'], default='eager-kernel')
    parser.add_argument('--image-edges', action='store_true',
                        help='also compare black, gray, white and checkerboard images')
    parser.add_argument('--output', type=Path, help='save environment and results as JSON')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compiler', choices=['eager', 'triton', 'tilelang', 'tvm'],
                        default='eager')
    parser.add_argument('--tvm-fuse', action='store_true', help='tvm only: FuseOps + FuseTIR')
    parser.add_argument('--split-k', type=int, default=1, help='kernel paths only')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--probe-kernels', action='store_true',
                        help='also lower eager library kernels from behaviour: linear, sum/mean, '
                             'ReLU/clamps, max pooling, inference batch norm')
    args = parser.parse_args()
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size
    options = {'split_k': args.split_k} if args.compiler in ('triton', 'tilelang') else {}
    if args.compiler == 'tvm':
        options = {'fuse': args.tvm_fuse}
        args.mode = 'primitive'                    # a tvm program has no eager kernels
    kw = dict(device=args.device, compiler=args.compiler, options=options,
              probe_kernels=args.probe_kernels)
    if args.samples < 1:
        parser.error('--samples must be positive')
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    metadata = {'torch': torch.__version__, 'python': sys.version.split()[0],
                'device': args.device, 'compiler': args.compiler, 'options': options,
                'image_size': args.image_size, 'pretrained_weights': True, 'mode': args.mode,
                'seed': 20261031, 'random_samples': args.samples, 'image_edges': args.image_edges}
    print(json.dumps(metadata), flush=True)
    def no_download(*args, **kwargs):
        raise RuntimeError('weights are not cached; use --allow-download to download')
    if args.allow_download:
        rows = [run_task(i, args.samples, args.mode, args.image_edges, **kw) for i in args.tasks]
    else:
        with patch('torch.hub.download_url_to_file', no_download):
            rows = [run_task(i, args.samples, args.mode, args.image_edges, **kw) for i in args.tasks]
    for row in rows:
        print(json.dumps(row))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(metadata, rows=rows), indent=2) + '\n')
    return 0 if all(r['status'] == 'EMPIRICAL_AGREEMENT' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
