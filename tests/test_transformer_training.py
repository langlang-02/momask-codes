import unittest
from types import SimpleNamespace

import torch

from utils.transformer_training import make_scheduler, check_resume_binding


class TransformerTrainingTests(unittest.TestCase):
    def test_epoch_schedule_and_resume(self):
        opt = SimpleNamespace(warm_up_epochs=5, decay_epochs=[200, 400],
                              gamma=0.1, milestones=[50000])
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler = make_scheduler(optimizer, opt, 2)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2e-5)
        for step in range(1, 801):
            optimizer.step()
            scheduler.step()
            if step in (10, 399, 400, 800):
                expected = {10: 2e-4, 399: 2e-4, 400: 2e-5, 800: 2e-6}[step]
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)
        other = make_scheduler(optimizer, opt, 2)
        other.load_state_dict(scheduler.state_dict())
        self.assertEqual(other.last_epoch, 800)

    def test_resume_rejects_changed_vq(self):
        check_resume_binding({'training_binding': {'sha256': 'a'}}, {'sha256': 'a'})
        with self.assertRaises(ValueError):
            check_resume_binding({'training_binding': {'sha256': 'a'}}, {'sha256': 'b'})
        with self.assertRaises(ValueError):
            check_resume_binding({}, {'sha256': 'a'})


if __name__ == '__main__':
    unittest.main()
