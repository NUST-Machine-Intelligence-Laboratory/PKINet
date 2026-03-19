import argparse
import os
import torch
import torch.nn as nn
from mmcv import Config
from mmrotate.models import build_detector
from mmcv.runner import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PKINet++ training model to deployment model (Re-parameterization)')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='training checkpoint file path (.pth)')
    parser.add_argument('output', help='output deployment checkpoint file path (.pth)')
    
    parser.add_argument('--device', default='cpu', help='device used for conversion')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1. Load config
    print(f'Loading config from {args.config}...')
    cfg = Config.fromfile(args.config)

    # Keep training-time structure before conversion (auto_reparam=False).
    # We need to load multi-branch weights first, then run manual conversion.
    if hasattr(cfg.model.backbone, 'deploy'):
        cfg.model.backbone.deploy = False
    if hasattr(cfg.model.backbone, 'auto_reparam'):
        cfg.model.backbone.auto_reparam = False

    # 2. Build model
    print('Building model...')
    model = build_detector(cfg.model)
    model.to(args.device)

    # 3. Load training checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    load_checkpoint(model, args.checkpoint, map_location=args.device)

    # 4. Run re-parameterization
    print('Switching to deployment mode (Kernel Fusion)...')
    # Call switch_to_deploy recursively when needed.
    # Backbone should expose switch_to_deploy in current implementation.
    if hasattr(model.backbone, 'switch_to_deploy'):
        model.backbone.switch_to_deploy()
    else:
        # If backbone does not expose the interface directly, fall back to recursive search.
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

    print('Re-parameterization done.')

    # 5. Verify converted structure (optional)
    # Print the first block after conversion and check fusion-related attributes.
    print('\nVerifying structure of first block...')
    try:
        # Adjust this access path if backbone internals change.
        sample_block = model.backbone.block1[0].attn.spatial_gating_unit
        print(sample_block)
        if hasattr(sample_block, 'fused_dw_conv') and not hasattr(sample_block, 'dw_branches'):
            print('SUCCESS: Structure looks correct (Fused).')
        else:
            print('WARNING: Structure might not be fully fused.')
    except Exception as e:
        print(f"Verification skipped: {e}")

    # 6. Save converted checkpoint
    print(f'\nSaving deployment checkpoint to {args.output}...')
    torch.save(model.state_dict(), args.output)
    print('All done!')


if __name__ == '__main__':
    main()
