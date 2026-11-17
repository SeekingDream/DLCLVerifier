"""Offline scope regression for the actual normalization wrapper in utils.py.

Identity isolates normalization; it is not a substitute for a benchmark network.
"""
from pathlib import Path
import sys
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from dlclv.reconstruction import reconstruct, UnsupportedReconstruction, ExecutionMismatch
from dlclv.bitwise import bitwise_equal


@pytest.mark.parametrize('value', [0., 0.5, 1.])
def test_utils_normalized_reconstructs_division(value):
    from torch.fx.experimental.proxy_tensor import make_fx
    model = utils.Normalized(torch.nn.Identity(), utils.IMAGENET_MEAN,
                             utils.IMAGENET_STD).eval()
    x = torch.full((1, 3, 8, 8), value)
    with torch.no_grad():
        graph = make_fx(model)(x)
    rebuilt = reconstruct(graph, (x,), format='aten')
    assert bitwise_equal(rebuilt(x), model(x))


def test_division_preserves_exceptional_bits():
    from torch.fx.experimental.proxy_tensor import make_fx
    x = torch.tensor([0., -0., 1., -1., float('inf'), float('nan'), 2**-149])
    y = torch.tensor([-1., 1., 0., 0., float('inf'), 1., 3.])
    graph = make_fx(lambda x, y: x / y)(x, y)
    rebuilt = reconstruct(graph, (x, y), format='aten')
    assert bitwise_equal(rebuilt(x, y), x / y)


def small_cnn():
    return utils.Normalized(torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Hardtanh(0, 6),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2),
    ), utils.IMAGENET_MEAN, utils.IMAGENET_STD).eval().requires_grad_(False)


def test_cnn_kernel_reconstruction_and_snapshot():
    model = small_cnn()
    cases = [utils.image_domain(8).sample(1, torch.Generator().manual_seed(seed))
             for seed in range(3)]
    cases += [(torch.zeros_like(cases[0][0]),), (torch.ones_like(cases[0][0]),)]
    with pytest.raises(UnsupportedReconstruction, match='convolution|conv2d'):
        reconstruct(model, cases[0])
    rebuilt = reconstruct(model, cases[0], mode='eager-kernel')
    assert rebuilt.compare(model, cases)['bitwise_equal']
    assert rebuilt.report['mode'] == 'eager-kernel'
    assert rebuilt.report['preserved_kernels']
    assert not rebuilt.report['universal_certificate']
    assert not rebuilt.report['empirical']['is_proof']
    before = rebuilt(*cases[0])
    model.net[-1].bias.add_(10)
    assert bitwise_equal(before, rebuilt(*cases[0]))
    assert not rebuilt.compare(model, cases[:1])['bitwise_equal']


def test_kernel_configuration_change_rejected():
    model = small_cnn()
    x = torch.rand(1, 3, 8, 8)
    rebuilt = reconstruct(model, (x,), mode='eager-kernel')
    original = torch.backends.mkldnn.enabled
    try:
        torch.backends.mkldnn.enabled = not original
        with pytest.raises(ExecutionMismatch, match='configuration'):
            rebuilt(x)
    finally:
        torch.backends.mkldnn.enabled = original


def test_kernel_mode_still_rejects_unknown_ops_and_tensor_getitem():
    import operator
    for target, args in [(torch.ops.aten.sin.default, ()), (operator.getitem, (0,))]:
        graph = torch.fx.Graph()
        x = graph.placeholder('x')
        y = graph.call_function(target, (x, *args))
        graph.output(y)
        gm = torch.fx.GraphModule({}, graph)
        with pytest.raises(UnsupportedReconstruction):
            reconstruct(gm, (torch.ones(3),), format='aten', mode='eager-kernel')
