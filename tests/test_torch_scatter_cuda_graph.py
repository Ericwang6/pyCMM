import torch
import pytest

from torch_scatter import segment_csr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_segment_csr_cuda_graph():
    device = torch.device("cuda")

    # Input tensor and CSR indptr defining 4 segments of equal length.
    src = torch.arange(16, dtype=torch.float32, device=device)
    indptr = torch.tensor([0, 4, 8, 12, 16], dtype=torch.long, device=device)

    # Eager baseline result.
    expected = segment_csr(src, indptr, reduce="sum")

    # Warmup to ensure kernels and memory pools are initialized before capture.
    for _ in range(3):
        _ = segment_csr(src, indptr, reduce="sum")

    static_src = src.clone()
    graph = torch.cuda.CUDAGraph()

    # Capture the segment_csr call into a CUDA graph.
    with torch.cuda.graph(graph):
        out = segment_csr(static_src, indptr, reduce="sum")

    # Replaying the graph multiple times should not error and should
    # produce the same values as the eager baseline.
    for _ in range(3):
        graph.replay()
        assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mask_cuda_graph():
    device = torch.device("cuda")

    # Fixed mask so that indexing and reduction have static shape (graph-able).
    torch.manual_seed(42)
    # mask = torch.rand(1000, device=device) > 0.5  # random True/False, fixed pattern
    mask = torch.tensor([0, 1, 2, 6, 9], device=device)

    # Eager baseline: sum of src where mask is True.
    src = torch.randn(1000, device=device)
    expected = torch.sum(src[mask])

    # Pre-allocate static tensors for capture (no dynamic shape or allocation inside graph).
    static_src = torch.empty(1000, device=device, dtype=src.dtype)
    static_mask = mask.clone()

    # Warmup.
    for _ in range(3):
        _ = torch.sum(src[mask])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.sum(static_src[static_mask])

    # Replay: copy current src into static buffer, replay graph, check result.
    static_src.copy_(src)
    graph.replay()
    assert torch.allclose(out, expected)

    # Replay again with different data to ensure graph is reusable.
    src2 = torch.randn(1000, device=device)
    expected2 = torch.sum(src2[mask])
    static_src.copy_(src2)
    graph.replay()
    assert torch.allclose(out, expected2)