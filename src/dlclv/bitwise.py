"""Storage-bit comparisons; never convert floating point values to another dtype."""
import numpy as np


def bitwise_equal(a, b):
    if hasattr(a, 'detach') or hasattr(b, 'detach'):
        import torch
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            # Existing ONNX fidelity checks compare NumPy outputs with torch outputs.
            def as_numpy(x):
                if isinstance(x, torch.Tensor):
                    if x.dtype == torch.bfloat16:
                        return None  # NumPy has no matching native bfloat16 dtype.
                    return x.detach().cpu().numpy()
                return np.asarray(x)
            a, b = as_numpy(a), as_numpy(b)
            return a is not None and b is not None and bitwise_equal(a, b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        # reshape before dtype view supports scalars and noncontiguous tensors.
        a = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(a, b)
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
