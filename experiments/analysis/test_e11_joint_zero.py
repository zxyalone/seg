"""Verify the joint-zero hook on real PFCR forwards without a GPU."""
from types import SimpleNamespace
import torch

import e11_joint_zero as J
from nnunetv2.training.nnUNetTrainer.phaser.modules import PhaseFrequencyCoherenceRouting


def main():
    torch.manual_seed(917)
    torch.set_num_threads(1)
    module = PhaseFrequencyCoherenceRouting(channels=2, use_prior=True).eval()
    with torch.no_grad():
        module.router[0].bias.copy_(torch.linspace(-.3, .7, 8))
        module.router[2].weight.normal_(0, .2)
        module.router[2].bias.copy_(torch.tensor([.2, -.4, .1]))
        module.residual_gate.fill_(.35)
    weights = {k: v.clone() for k, v in module.state_dict().items()}
    model = SimpleNamespace(name='test', network=SimpleNamespace(pfcr_modules=[module]), has_pfcr=True)
    x, prior = torch.randn(1, 2, 9, 10, 11), torch.rand(1, 1, 9, 10, 11)
    with torch.no_grad():
        normal, normal_route = module(x, prior)
        J.joint_intervention(model, J.CONDITION)
        joint, joint_route = module(x, prior)
        _, other_route = module(x * 2, 1 - prior)
        torch.testing.assert_close(joint_route, other_route, rtol=0, atol=0)
        assert not torch.allclose(joint_route, torch.full_like(joint_route, 1 / 3))
        assert not torch.allclose(normal_route, joint_route)
        J.joint_intervention(model, 'none')
        restored, restored_route = module(x, prior)
        torch.testing.assert_close(normal, restored, rtol=0, atol=0)
        torch.testing.assert_close(normal_route, restored_route, rtol=0, atol=0)
        J.ORIGINAL_SET(model, 'pc_zero')
        manual, manual_route = module(x, torch.zeros_like(prior))
        torch.testing.assert_close(joint, manual, rtol=0, atol=0)
        torch.testing.assert_close(joint_route, manual_route, rtol=0, atol=0)
    assert all(torch.equal(v, weights[k]) for k, v in module.state_dict().items())
    assert J.AUDIT['test/level0']['max_spatial_range'] == 0
    print('PASS: both channels zero; joint hook matches manual zero PC/prior; learned route constant and nonuniform; normal restored; weights unchanged')


if __name__ == '__main__':
    main()
