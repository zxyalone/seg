#!/usr/bin/env python3
"""E11 follow-up: zero PC and prior together at every trained A6 router input.

This analysis-only hook preserves network code, weights, feature transforms,
residual gates and the original E11 measurements. Baseline predictions/evals are
reused by symlink; baseline utilities are copied. Only 63 new predictions run.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

import e11_interventions as E

CONDITION = 'pc_prior_zero'
AUDIT = {}
ORIGINAL_SET = E.C.set_intervention


def joint_intervention(model, mode, levels=None):
    for module in model.network.pfcr_modules:
        for handle in getattr(module, '_joint_zero_handles', []):
            handle.remove()
        module._joint_zero_handles = []
    ORIGINAL_SET(model, 'none' if mode == CONDITION else mode, levels)
    if mode != CONDITION:
        return
    for level, module in enumerate(model.network.pfcr_modules):
        if levels is not None and level not in levels:
            continue
        assert module.use_prior and module.router[0].in_channels == 2
        key = f'{model.name}/level{level}'

        def zero_input(router, args, key=key):
            assert len(args) == 1 and args[0].shape[1] == 2
            record = AUDIT.setdefault(key, {'calls': 0})
            record['calls'] += 1
            return (torch.zeros_like(args[0]),)

        def check_constant(router, args, logits, key=key, module=module):
            record = AUDIT[key]
            if 'route' in record:
                return
            route = logits.softmax(1).float().flatten(2)
            spatial_range = (route.amax(-1) - route.amin(-1)).amax().item()
            first, last = module.router[0], module.router[2]
            expected_logits = torch.mv(last.weight.detach().float().flatten(1), first.bias.detach().float().relu()) + last.bias.detach().float()
            expected = expected_logits.softmax(0)
            actual = route[0, :, 0]
            assert spatial_range < 1e-6, 'Joint-zero router is not spatially constant'
            assert (actual - expected).abs().max().item() < .002, 'Joint-zero router differs from its trained bias mixture'
            record.update(route=actual.tolist(), bias_route_fp32=expected.tolist(),
                          max_spatial_range=spatial_range, dtype=str(logits.dtype),
                          residual_gate_tanh=module.residual_gate.detach().float().tanh().item())

        module._joint_zero_handles = [module.router.register_forward_pre_hook(zero_input),
                                      module.router.register_forward_hook(check_constant)]


def prepare_baseline(out, source):
    assert out.resolve() != source.resolve(), 'Follow-up must not overwrite original E11'
    for seed in range(3):
        run = f'A6_s{seed}'
        for kind in ('pred', 'eval'):
            target = out / kind / run / 'none'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                original = source / kind / run / 'none'
                assert original.is_dir(), original
                target.symlink_to(original.resolve(), target_is_directory=True)
        for original in (source / 'utility' / run).glob('*.json'):
            target = out / 'utility' / run / original.name
            if not target.exists():
                data = json.loads(original.read_text())
                data['loss'] = {'none': data['loss']['none']}
                E.C.write_json(data, target)
            data = json.loads(target.read_text())
            prediction = out / 'pred' / run / CONDITION / (original.stem + '.nii.gz')
            if prediction.exists() and CONDITION not in data['loss']:
                # Retain interrupted exports, but do not let the original E11
                # case-level skip treat missing utility as a complete condition.
                prediction.replace(prediction.with_name(prediction.name + '.incomplete'))


def main():
    minimal = argparse.ArgumentParser(add_help=False)
    minimal.add_argument('command', choices=('run', 'summarize'))
    minimal.add_argument('--out', type=Path, default=E.C.RESULTS_ROOT / '_analysis/E11_followup')
    minimal.add_argument('--baseline-source', type=Path, default=E.C.RESULTS_ROOT / '_analysis/E11')
    args, rest = minimal.parse_known_args()
    prepare_baseline(args.out, args.baseline_source)
    E.CONDITIONS[CONDITION] = (CONDITION, None)
    E.LABELS[CONDITION] = 'Zero PC and prior together (trained constant route)'
    E.C.set_intervention = joint_intervention
    sys.argv = [sys.argv[0], args.command, '--out', str(args.out), '--conditions',
                *(['none', CONDITION] if args.command == 'summarize' else [CONDITION]), *rest]
    E.main()
    if args.command == 'run' and AUDIT:
        E.C.write_json({'condition': CONDITION, 'router_inputs': ['PC=0', 'prior=0'],
                       'scope': 'all five levels; learned weights and residual gates unchanged',
                       'checks': AUDIT}, args.out / 'joint_zero_router_audit.json')


if __name__ == '__main__':
    main()
