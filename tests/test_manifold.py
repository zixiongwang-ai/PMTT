import torch
from pmtt.manifold import unit, residuals, reference, coefficients, error
from pmtt.representation import observed_states

def test_same_visible_information_and_tangent_factor():
    generator = torch.Generator().manual_seed(93)
    center = torch.randn(3, 24, generator=generator)
    visible = unit(torch.randn(3, 16, 24, generator=generator))
    _, tangent, _ = residuals(center, visible)
    factor = reference(tangent)['a']
    assert factor.shape == (3, 24, 8)
    assert torch.linalg.matrix_rank(factor).max() <= 8
    torch.testing.assert_close(unit(center)[:, None] @ factor, torch.zeros(3, 1, 8), atol=2e-6, rtol=0)
    observed = observed_states(center, visible)
    torch.testing.assert_close(observed[:, 1:], visible)
    torch.testing.assert_close(observed[:, 0], unit(center))

def test_bounded_projection_and_basis_invariance():
    factor = torch.zeros(1, 12, 8, dtype=torch.float64)
    factor[0, :8] = torch.eye(8, dtype=torch.float64) * .2
    delta = torch.ones(1, 5, 12, dtype=torch.float64)
    coordinates = coefficients(factor, delta)
    assert torch.all(coordinates.norm(dim=-1) <= 1 + 1e-12)
    original = error(factor, delta)[0]
    reflected = error(factor.flip(-1), delta)[0]
    torch.testing.assert_close(original, reflected, atol=1e-12, rtol=1e-12)
