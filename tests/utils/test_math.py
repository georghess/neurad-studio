import pytest
import torch

from nerfstudio.utils.math import components_from_spherical_harmonics


@pytest.mark.parametrize("components", list(range(1, 5 + 1)))
def test_spherical_harmonics(components):
    torch.manual_seed(0)
    N = 1000000

    dx = torch.normal(0, 1, size=(N, 3))
    dx = dx / torch.linalg.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(components, dx)
    matrix = (sh.T @ sh) / N * 4 * torch.pi
    torch.testing.assert_close(matrix, torch.eye(components**2), rtol=0, atol=1.5e-2)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 10])
@pytest.mark.parametrize("normalize", [False, True])
def test_chamfer_distance_normalization_is_independent_of_chunking(chunk_size, normalize):
    from nerfstudio.utils.math import chamfer_distance

    source = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]])
    target = torch.tensor([[1.0, 0, 0], [7.0, 0, 0]])
    # Squared nearest distances: source -> target = 1 + 1 + 9; reverse = 1 + 9.
    expected = 21.0 / (len(target) if normalize else 1)
    result = chamfer_distance(source, target, chunk_size=chunk_size, normalize_with_target=normalize)
    assert result.item() == pytest.approx(expected)
