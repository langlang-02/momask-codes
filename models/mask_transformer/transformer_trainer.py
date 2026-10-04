import os
import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *
from utils.distributed import barrier, load_state_dict_compat, reduce_sum, set_loader_epoch, unwrap_model

from utils.transformer_training import make_scheduler, check_resume_binding

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.next_batch = 0
        self.is_main = getattr(args, 'is_main_process', True)

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir) if self.is_main else None


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        with torch.no_grad():
            code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // self.opt.unit_length

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        if not self.is_main:
            return
        t2m_trans_state_dict = unwrap_model(self.t2m_transformer).state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'next_batch': self.next_batch,
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'training_binding': getattr(self.opt, 'training_binding', None),
            'schedule_config': self.schedule_config,
        }
        torch.save(state, file_name + '.tmp')
        os.replace(file_name + '.tmp', file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device, weights_only=False)
        check_resume_binding(checkpoint, getattr(self.opt, 'training_binding', None))
        if checkpoint.get('schedule_config', self.schedule_config) != self.schedule_config:
            raise ValueError('Resume learning-rate schedule or batch size changed')
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.next_batch = checkpoint.get('next_batch', 0)
        missing_keys, unexpected_keys = load_state_dict_compat(
            self.t2m_transformer, checkpoint['t2m_transformer'], strict=False
        )
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except Exception:
            if getattr(self.opt, 'training_binding', None) is not None:
                raise
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.schedule_config = dict(steps_per_epoch=len(train_loader), batch_size=self.opt.batch_size,
                                    lr=self.opt.lr, gamma=self.opt.gamma,
                                    warm_up_epochs=self.opt.warm_up_epochs, decay_epochs=self.opt.decay_epochs,
                                    warm_up_iter=self.opt.warm_up_iter, milestones=self.opt.milestones)
        self.scheduler = make_scheduler(self.opt_t2m_transformer, self.opt, len(train_loader))

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            if self.is_main:
                print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_main:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        native_g1 = self.opt.dataset_name == 'bones_g1'
        if not native_g1 and self.is_main:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                self.opt.save_root, eval_val_loader, unwrap_model(self.t2m_transformer), unwrap_model(self.vq_model), self.logger, epoch,
                best_fid=100, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        barrier()

        while epoch < self.opt.max_epoch:
            set_loader_epoch(train_loader, epoch)
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                if i < self.next_batch:
                    continue
                it += 1
                if self.opt.warm_up_epochs is None and it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                self.next_batch = i + 1
                if self.is_main:
                    logs['loss'] += loss
                    logs['acc'] += acc
                    logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if self.is_main and it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.next_batch = 0
            epoch += 1

            if self.is_main:
                print('Validation time:')
            set_loader_epoch(val_loader, epoch)
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            val_totals = reduce_sum([sum(val_loss), sum(val_acc), len(val_loss)], self.device)
            val_count = max(val_totals[-1], 1.0)
            mean_val_loss = val_totals[0] / val_count
            mean_val_acc = val_totals[1] / val_count

            if self.is_main:
                print(f"Validation loss:{mean_val_loss:.3f}, accuracy:{mean_val_acc:.3f}")
                self.logger.add_scalar('Val/loss', mean_val_loss, epoch)
                self.logger.add_scalar('Val/acc', mean_val_acc, epoch)

            if native_g1 and self.is_main and mean_val_loss < self.best_loss:
                self.best_loss = mean_val_loss
                self.save(pjoin(self.opt.model_dir, 'net_best_val.tar'), epoch, it)
                print(f"Best native G1 validation loss: {self.best_loss:.5f}")

            if self.is_main and mean_val_acc > self.best_acc:
                print(f"Improved accuracy from {self.best_acc:.02f} to {mean_val_acc}!!!")
                self.best_acc = mean_val_acc
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            if self.opt.save_every_e > 0 and epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, f'ep{epoch:04d}.tar'), epoch, it)

            if not native_g1 and self.is_main:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                    self.opt.save_root, eval_val_loader, unwrap_model(self.t2m_transformer), unwrap_model(self.vq_model), self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
                )
            barrier()


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.next_batch = 0
        self.is_main = getattr(args, 'is_main_process', True)

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir) if self.is_main else None
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        with torch.no_grad():
            code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // self.opt.unit_length

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        if not self.is_main:
            return
        res_trans_state_dict = unwrap_model(self.res_transformer).state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'next_batch': self.next_batch,
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'training_binding': getattr(self.opt, 'training_binding', None),
            'schedule_config': self.schedule_config,
        }
        torch.save(state, file_name + '.tmp')
        os.replace(file_name + '.tmp', file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device, weights_only=False)
        check_resume_binding(checkpoint, getattr(self.opt, 'training_binding', None))
        if checkpoint.get('schedule_config', self.schedule_config) != self.schedule_config:
            raise ValueError('Resume learning-rate schedule or batch size changed')
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.next_batch = checkpoint.get('next_batch', 0)
        missing_keys, unexpected_keys = load_state_dict_compat(
            self.res_transformer, checkpoint['res_transformer'], strict=False
        )
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except Exception:
            if getattr(self.opt, 'training_binding', None) is not None:
                raise
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.schedule_config = dict(steps_per_epoch=len(train_loader), batch_size=self.opt.batch_size,
                                    lr=self.opt.lr, gamma=self.opt.gamma,
                                    warm_up_epochs=self.opt.warm_up_epochs, decay_epochs=self.opt.decay_epochs,
                                    warm_up_iter=self.opt.warm_up_iter, milestones=self.opt.milestones)
        self.scheduler = make_scheduler(self.opt_res_transformer, self.opt, len(train_loader))

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            if self.is_main:
                print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        if self.is_main:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        native_g1 = self.opt.dataset_name == 'bones_g1'
        if not native_g1 and self.is_main:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, unwrap_model(self.res_transformer), unwrap_model(self.vq_model), self.logger, epoch,
                best_fid=100, best_div=100,
                best_top1=0, best_top2=0, best_top3=0,
                best_matching=100, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=False, save_anim=False
            )
        barrier()

        while epoch < self.opt.max_epoch:
            set_loader_epoch(train_loader, epoch)
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                if i < self.next_batch:
                    continue
                it += 1
                if self.opt.warm_up_epochs is None and it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                self.next_batch = i + 1
                if self.is_main:
                    logs['loss'] += loss
                    logs["acc"] += acc
                    logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if self.is_main and it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.next_batch = 0
            epoch += 1

            if self.is_main:
                print('Validation time:')
            set_loader_epoch(val_loader, epoch)
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            val_totals = reduce_sum([sum(val_loss), sum(val_acc), len(val_loss)], self.device)
            val_count = max(val_totals[-1], 1.0)
            mean_val_loss = val_totals[0] / val_count
            mean_val_acc = val_totals[1] / val_count

            if self.is_main:
                print(f"Validation loss:{mean_val_loss:.3f}, Accuracy:{mean_val_acc:.3f}")
                self.logger.add_scalar('Val/loss', mean_val_loss, epoch)
                self.logger.add_scalar('Val/acc', mean_val_acc, epoch)

            if self.is_main and mean_val_loss < self.best_loss:
                print(f"Improved loss from {self.best_loss:.02f} to {mean_val_loss}!!!")
                self.best_loss = mean_val_loss
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                if native_g1:
                    self.save(pjoin(self.opt.model_dir, 'net_best_val.tar'), epoch, it)

            if self.is_main and mean_val_acc > self.best_acc:
                print(f"Improved acc from {self.best_acc:.02f} to {mean_val_acc}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                self.best_acc = mean_val_acc

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            if self.opt.save_every_e > 0 and epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, f'ep{epoch:04d}.tar'), epoch, it)

            if not native_g1 and self.is_main:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                    self.opt.save_root, eval_val_loader, unwrap_model(self.res_transformer), unwrap_model(self.vq_model), self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
                )
            barrier()
