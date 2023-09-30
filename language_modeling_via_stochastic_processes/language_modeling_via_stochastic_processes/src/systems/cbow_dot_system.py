import os
import pickle
import math
from language_modeling_via_stochastic_processes.src import constants

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from language_modeling_via_stochastic_processes.src.datasets import (
    wikisection,
    recipe,
    wikihow,
    tm2,
    tm2fixed,
    tm2sixed,
    tm2sep,
    tm2new,
    tm2newsep,
    tickettalk,
    roc_stories
)

import datasets
from language_modeling_via_stochastic_processes.src.models.utils import weights_init

NAME2DATASET = {
    'wikisection': wikisection.WikisectionTriplet,
    'wikisectioncbow': wikisection.WikisectionCBow,
    'wikihowcbow': wikihow.WikihowCBow,
    'recipecbow': recipe.RecipeCBow,
    'recipe': recipe.RecipeTriplet,
    'wikihow': wikihow.WikihowTriplet,
    'roc_storiescbow': roc_stories.ROCStoriesCBow,
    'roc_stories': roc_stories.ROCStoriesTriplet,
    'tm2': tm2.TM2Triplet,
    'tm2cbow': tm2.TM2CBow,
    'tm2fixedcbow': tm2fixed.TM2FixedCBow,
    'tm2sixedcbow': tm2sixed.TM2SixedCBow,
    'tm2sepcbow': tm2sep.TM2SepCBow,
    'tm2newcbow': tm2new.TM2NewCBow,
    'tm2newsepcbow': tm2newsep.TM2NewSepCBow,
    'tickettalk': tickettalk.TicketTalkTriplet,
    'tickettalkcbow': tickettalk.TicketTalkCBow,
}

from language_modeling_via_stochastic_processes.src.models import language
from language_modeling_via_stochastic_processes.src.objectives import brownian_bridge
from torch.utils.data._utils.collate import default_collate
torch.autograd.set_detect_anomaly(True)

def create_dataloader(dataset, config, shuffle=True):
    def get_feats(obs):
        #import pdb; pdb.set_trace()
        input_ids_i, attention_mask_i = dataset.tokenize_caption(
                obs, device=torch.device('cpu'))
        input_ids_i = input_ids_i[:, :dataset.max_length]
        attention_mask_i = attention_mask_i[:, :dataset.max_length]
        #feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return input_ids_i, attention_mask_i

    def collate_fn(examples):
        examples = [ex for ex in examples if ex is not None]
        batch = default_collate(examples)


        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        DEBUG = True
        DEBUG = False
        if DEBUG:
            import pdb; pdb.set_trace()
            sent_0 = dataset.tokenizer.decode(obs_0[0].view(-1))
            sent_t = dataset.tokenizer.decode(obs_t[0].view(-1))
            sent_T = dataset.tokenizer.decode(obs_T[0].view(-1))
            sent_t_distract = dataset.tokenizer.decode(obs_t[1].view(-1))
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        input_ids_0, attention_mask_0 = get_feats(obs_0) # bsz, h
        input_ids_T, attention_mask_T = get_feats(obs_t)
        input_ids_t, attention_mask_t = get_feats(obs_T)

        batch = {'input_ids_0': input_ids_0,
                'attention_mask_0': attention_mask_0,
                'input_ids_T': input_ids_T,
                'attention_mask_T': attention_mask_T,
                'input_ids_t': input_ids_t,
                'attention_mask_t': attention_mask_t,
                }
        return batch

        #feats_0 = self.get_feats(obs_0) # bsz, h
        #feats_T = self.get_feats(obs_t)
        #feats_t = self.get_feats(obs_T)


    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.experiment_params.data_loader_workers,
        collate_fn=collate_fn,
    )
    return loader

class CBowSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._set_dataset()
        self._set_language_encoder()
        #import pdb; pdb.set_trace()
        #self.opt_mlp = nn.Sequential(*[nn.Linear(self.model.latent_dim*3, self.model.hidden_dim*4), nn.ReLU(), nn.Linear(self.model.hidden_dim*4, 1)])
        self.opt_mlp = nn.Sequential(*[nn.Linear(self.model.latent_dim*2, self.model.hidden_dim*4), nn.ReLU(), nn.Linear(self.model.hidden_dim*4, self.model.latent_dim)])
        self.opt_mlp.apply(weights_init)

    def configure_optimizers(self):
        #import pdb; pdb.set_trace()
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def train_dataloader(self):
        print ('SHUFFLE: ', self.config.data_params.shuffle)
        return create_dataloader(self.train_dataset, self.config, shuffle=self.config.data_params.shuffle)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, shuffle=False)
    def _set_dataset(self):
        dname = self.config.data_params.name
        if 'recipe' in dname:
            self.data_dir = constants.PATH2RECIPENLG
            self.all_dataset = datasets.load_dataset("recipe_nlg", data_dir=self.data_dir)['train']
        elif 'wikihow' in dname:
            self.data_name = constants.PATH2WIKIHOW
            with open(self.data_name, 'rb') as f:
                self.all_dataset = pickle.load(f)
        else:
            self.all_dataset = None

        dataset = NAME2DATASET[dname]
        self.train_dataset = dataset(
            train=True,
            seed=self.config.data_params.data_seed,
            all_dataset=self.all_dataset,
            config=self.config
        )
        self.test_dataset = dataset(
            train=False,
            seed=self.config.data_params.data_seed,
            all_dataset=self.all_dataset,
            config=self.config
        )
        self.val_dataset = dataset(
            train='validation',
            seed=self.config.data_params.data_seed,
            all_dataset=self.all_dataset,
            config=self.config
        )

    def set_to_train(self):
        pass

    def _set_language_encoder(self):
        self.model = language.GPT2OUEncoder(
            hidden_dim=self.config.model_params.hidden_size,
            latent_dim=self.config.model_params.latent_dim,
            finetune_gpt2=False,
            single_layer=self.config.model_params.single_layer)

        print (f'FINETUNE GPT2: {self.config.model_params.finetune_gpt2}')
        self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))
        for p in self.model.model.parameters():
            #p.requires_grad = False
            p.requires_grad = self.config.model_params.finetune_gpt2

    def forward(self, input_ids, attention_mask):
        feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_feats(self, obs):
        #import pdb; pdb.set_trace()
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs, device=self.device)
        input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch, batch_idx):
        assert False
        #import pdb; pdb.set_trace()
        #torch.cuda.empty_cache()
        #obs_0 = batch['y_0']
        #obs_t = batch['y_t']
        #obs_T = batch['y_T']
        #t_s = batch['t_'].float()
        #ts = batch['t'].float()
        #Ts = batch['T'].float()
        #feats_0 = self.get_feats(obs_0) # bsz, h
        #feats_T = self.get_feats(obs_t)
        #feats_t = self.get_feats(obs_T)
        feats_0 = self.forward(input_ids=batch['input_ids_0'].cuda(), attention_mask=batch['attention_mask_0'].cuda()) # bsz, h
        feats_T = self.forward(input_ids=batch['input_ids_T'].cuda(), attention_mask=batch['attention_mask_T'].cuda()) # bsz, h
        feats_t = self.forward(input_ids=batch['input_ids_t'].cuda(), attention_mask=batch['attention_mask_t'].cuda()) # bsz, h


        bsz = feats_0.shape[0]

        feats_0_T = torch.cat((feats_0, feats_T), dim=-1) # bsz, h*2
        positive_logits = self.opt_mlp(torch.cat((feats_0_T, feats_t), dim=-1)).squeeze(-1) # bsz, 1
        feats_0_T_expanded = feats_0_T.unsqueeze(1).expand(-1, bsz, -1) # bsz, bsz, h*2
        feats_t_expanded = feats_t.unsqueeze(0).expand(bsz, -1, -1) # bsz, bsz, h
        feats_cat = torch.cat((feats_0_T_expanded, feats_t_expanded), dim=-1).view(bsz*bsz, -1) # bsz, bsz, h*3
        negative_logits = self.opt_mlp(feats_cat).squeeze(-1).view(bsz, bsz)
        logits = positive_logits - torch.logsumexp(negative_logits, dim=-1)
        loss = -logits.mean()


        #positive_logits = self.opt_mlp(
        #log_q_y_tp1 = self.model.get_log_q(feats_t)
        #loss_fn = brownian_bridge.BrownianBridgeLoss(
        #    z_0=feats_0,
        #    z_t=feats_t,
        #    z_T=feats_T,
        #    t_=t_s,
        #    t=ts,
        #    T=Ts,
        #    alpha=0,
        #    var=0,
        #    log_q_y_T=log_q_y_tp1,
        #    loss_type=self.config.loss_params.name,
        #    eps=self.config.model_params.eps,
        #    max_seq_len=batch['total_t'].float(),
        #)
        #loss = loss_fn.get_loss()
        return loss
    def get_losses_for_batch_withacc(self, batch, batch_idx):
        #import pdb; pdb.set_trace()
        #torch.cuda.empty_cache()
        #obs_0 = batch['y_0']
        #obs_t = batch['y_t']
        #obs_T = batch['y_T']
        #t_s = batch['t_'].float()
        #ts = batch['t'].float()
        #Ts = batch['T'].float()
        #feats_0 = self.get_feats(obs_0) # bsz, h
        #feats_T = self.get_feats(obs_t)
        #feats_t = self.get_feats(obs_T)
        feats_0 = self.forward(input_ids=batch['input_ids_0'].cuda(), attention_mask=batch['attention_mask_0'].cuda()) # bsz, h
        feats_T = self.forward(input_ids=batch['input_ids_T'].cuda(), attention_mask=batch['attention_mask_T'].cuda()) # bsz, h
        feats_t = self.forward(input_ids=batch['input_ids_t'].cuda(), attention_mask=batch['attention_mask_t'].cuda()) # bsz, h
        bsz = feats_0.shape[0]

        feats_0_T = torch.cat((feats_0, feats_T), dim=-1) # bsz, h*2
        predicted_vector = self.opt_mlp(feats_0_T) # bsz, h
        diff = feats_t * predicted_vector
        positive_logits = diff.sum(dim=-1) #self.opt_mlp(torch.cat((feats_0_T, feats_t), dim=-1)).squeeze(-1) # bsz, 1
        feats_t_expanded = feats_t.unsqueeze(0).expand(bsz, -1, -1) # bsz, bsz, h
        diff = feats_t_expanded * predicted_vector.unsqueeze(1)
        negative_logits = diff.sum(dim=-1) #self.opt_mlp(torch.cat((feats_0_T, feats_t), dim=-1)).squeeze(-1) # bsz, 1
        predicted_labels = negative_logits.argmax(dim=-1)
        acc = (predicted_labels == torch.arange(0, bsz).long().to(predicted_labels.device)).float().mean()
        logits = positive_logits - torch.logsumexp(negative_logits, dim=-1)
        loss = -logits.mean()


        #positive_logits = self.opt_mlp(
        #log_q_y_tp1 = self.model.get_log_q(feats_t)
        #loss_fn = brownian_bridge.BrownianBridgeLoss(
        #    z_0=feats_0,
        #    z_t=feats_t,
        #    z_T=feats_T,
        #    t_=t_s,
        #    t=ts,
        #    T=Ts,
        #    alpha=0,
        #    var=0,
        #    log_q_y_T=log_q_y_tp1,
        #    loss_type=self.config.loss_params.name,
        #    eps=self.config.model_params.eps,
        #    max_seq_len=batch['total_t'].float(),
        #)
        #loss = loss_fn.get_loss()
        return loss, acc, bsz

    def training_step(self, batch, batch_idx):
        #import pdb; pdb.set_trace()
        loss, acc, batch_size = self.get_losses_for_batch_withacc(batch, batch_idx)
        #wandb.log({'train_loss': loss.cpu().detach().numpy(),
        #           'epoch': self.trainer.current_epoch})
        wandb.log({'train_loss': loss.cpu().detach().item(),
                   'epoch': self.trainer.current_epoch})
        #self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.log("performance", {'train_ppl': loss.exp().cpu().detach().item(), 'train_acc': acc.cpu().detach().item()},  prog_bar=True, on_step=True, batch_size=batch_size)
        return loss
    def validation_step(self, batch, i):
        loss, acc, batch_size = self.get_losses_for_batch_withacc(batch=batch, batch_idx=i)
        val_loss = loss.cpu().item()
        val_bsz = batch_size
        val_ppl = loss.exp().cpu().item()
        val_acc = acc.cpu().item()
        wandb.log({'val_loss': val_loss,
                   'epoch': self.trainer.current_epoch})
        self.log('val_bsz', val_bsz, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('val_loss', val_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('val_ppl', val_ppl, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('val_acc', val_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return {'val_loss': val_loss, 'val_bsz': val_bsz, 'val_ppl': val_ppl, 'val_acc': val_acc}

    def validation_epoch_end(self, items):
        val_losses = [item['val_loss'] for item in items]
        val_accs = [item['val_acc'] for item in items]
        val_bszs = [item['val_bsz'] for item in items]
        total_loss = sum([a*b for a, b in zip(val_losses, val_bszs)])
        total_acc = sum([a*b for a, b in zip(val_accs, val_bszs)])
        total_bsz = sum(val_bszs)
        val_loss = total_loss / total_bsz
        val_acc = total_acc / total_bsz


        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_ppl', math.exp(val_loss), prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, i):
        loss, acc, batch_size = self.get_losses_for_batch_withacc(batch=batch, batch_idx=i)
        test_loss = loss.cpu().item()
        test_bsz = batch_size
        test_ppl = loss.exp().cpu().item()
        test_acc = acc.cpu().item()
        wandb.log({'test_loss': test_loss,
                   'epoch': self.trainer.current_epoch})
        self.log('test_bsz', test_bsz, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('test_loss', test_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('test_ppl', test_ppl, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('test_acc', test_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return {'test_loss': test_loss, 'test_bsz': test_bsz, 'test_ppl': test_ppl, 'test_acc': test_acc}

    def test_epoch_end(self, items):
        test_losses = [item['test_loss'] for item in items]
        test_accs = [item['test_acc'] for item in items]
        test_bszs = [item['test_bsz'] for item in items]
        total_loss = sum([a*b for a, b in zip(test_losses, test_bszs)])
        total_acc = sum([a*b for a, b in zip(test_accs, test_bszs)])
        total_bsz = sum(test_bszs)
        test_loss = total_loss / total_bsz
        test_acc = total_acc / total_bsz


        self.log('test_loss', test_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_ppl', math.exp(test_loss), prog_bar=True, on_step=False, on_epoch=True)

    #def test_step(self, batch, i):
    #    loss, acc = self.get_losses_for_batch_withacc(batch=batch, batch_idx=i)
    #    wandb.log({'test_loss': loss.cpu().detach().item(),
    #               'epoch': self.trainer.current_epoch})
    #    self.log('test_loss', loss.cpu().detach().item(), prog_bar=True, on_step=True)
    #    self.log('test_ppl', loss.exp().cpu().detach().item(), prog_bar=True, on_step=True)
    #    self.log('test_acc', acc.cpu().detach().item(), prog_bar=True, on_step=True)
    #    return loss

    #def save(self, directory):
    #    torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
    #    torch.save(self.opt_mlp.state_dict(), os.path.join(directory, "opt_mlp.pt"))
    #    torch.save(self.model.model.state_dict(), os.path.join(directory, "gpt2.pt"))
    #    torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))
