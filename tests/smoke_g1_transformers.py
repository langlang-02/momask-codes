"""Real-data, full-model GPU smoke test without touching production runs."""

import argparse
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2MotionDataset
from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer, ResidualTransformerTrainer
from utils.transformer_training import prepare_g1_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoints-dir', required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='momask-transformer-smoke-') as temporary:
        out = Path(temporary)
        names = []
        for name in (root / 'train.txt').read_text().splitlines():
            motion = np.load(root / 'merged_joint_vecs' / f'{name}.npy')
            if 40 <= len(motion) < 200:
                names.append(name)
            if len(names) == 4:
                break
        assert len(names) == 4
        split = out / 'split.txt'
        split.write_text('\n'.join(names) + '\n')
        for stage, model_class, trainer_class in (
                ('mask', MaskTransformer, MaskTransformerTrainer),
                ('res', ResidualTransformer, ResidualTransformerTrainer)):
            run = out / stage
            (run / 'model').mkdir(parents=True)
            opt = SimpleNamespace(
                dataset_name='bones_g1', data_root=str(root),
                motion_dir=str(root / 'new_joint_vecs'), text_dir=str(root / 'texts'),
                checkpoints_dir=args.checkpoints_dir, vq_name='bones_g1_rvq_rot6d68_v1_nq6',
                vq_checkpoint='frozen_latest_ep0499_it8998000.tar',
                representation_config='configs/representations/g1_local_rot6d68_v1.yaml',
                device=torch.device('cuda'), is_main_process=True, is_train=True,
                save_root=str(run), model_dir=str(run / 'model'), log_dir=str(run / 'log'),
                max_motion_length=196, batch_size=2, lr=2e-4, gamma=0.1,
                warm_up_epochs=5, decay_epochs=[200, 400], warm_up_iter=2000,
                milestones=[50000], is_continue=False, max_epoch=1, log_every=1,
                save_latest=1, save_every_e=1, eval_every_e=10,
                cond_drop_prob=0.1 if stage == 'mask' else 0.2, force_mask=False,
            )
            entry = importlib.import_module('train_t2m_transformer' if stage == 'mask' else 'train_res_transformer')
            entry.opt = opt
            entry.dim_pose = prepare_g1_inputs(opt)
            assert entry.dim_pose == 68
            vq, vq_opt = entry.load_vq_model()
            opt.unit_length = vq_opt.stride_t ** vq_opt.down_t
            opt.num_tokens = vq_opt.nb_code
            opt.num_quantizers = vq_opt.num_quantizers
            meta = Path(args.checkpoints_dir) / 'bones_g1' / opt.vq_name / 'meta'
            dataset = Text2MotionDataset(opt, np.load(meta / 'mean.npy'), np.load(meta / 'std.npy'), split)
            assert len(dataset) >= 4
            indices = opt.representation['indices']
            for name in names:
                np.testing.assert_array_equal(dataset.data_dict[name]['motion'],
                    np.load(root / 'new_joint_vecs' / f'{name}.npy')[:, indices])
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            kwargs = dict(code_dim=vq_opt.code_dim, cond_mode='text', latent_dim=384,
                          ff_size=1024, num_layers=8, num_heads=6, dropout=0.2,
                          clip_dim=512, cond_drop_prob=opt.cond_drop_prob,
                          clip_version='ViT-B/32', opt=opt)
            if stage == 'res':
                kwargs.update(shared_codebook=vq_opt.shared_codebook, share_weight=True)
            model = model_class(**kwargs)
            trainer = trainer_class(opt, model, vq)
            before = {k: v.clone() for k, v in vq.state_dict().items()}
            trainer.train(loader, loader, None, None, None)
            saved = torch.load(run / 'model/latest.tar', map_location='cpu', weights_only=False)
            assert saved['ep'] == 1 and saved['next_batch'] == 0
            assert (run / 'model/ep0001.tar').is_file()
            assert np.isfinite(saved['best_loss'])
            assert all(torch.equal(v.cpu(), vq.state_dict()[k].cpu()) for k, v in before.items())
            # Save/resume the actual optimizer and continue for a second epoch.
            opt.is_continue = True
            opt.max_epoch = 2
            trainer.train(loader, loader, None, None, None)
            final = torch.load(run / 'model/latest.tar', map_location='cpu', weights_only=False)
            assert final['ep'] == 2 and final['total_it'] == 2 * len(loader)
            assert final['best_loss'] <= saved['best_loss']
            assert (run / 'model/ep0002.tar').is_file()
            print(json.dumps(dict(stage=stage, status='PASS', samples=len(dataset),
                                  feature_dim=68, frozen_rvq_epoch=499, resume_epoch=2)))
            trainer.logger.close()
            del trainer, model, vq, before, saved, final
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
