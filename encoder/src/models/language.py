from encoder.src.models.utils import weights_init
import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers import BertModel
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerFast, AutoModelForCausalLM
from encoder.src.models.utils import weights_init
from tokenizers import Tokenizer

class GPT2OUEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, finetune_gpt2=False, single_layer=False):
        super(GPT2OUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune_gpt2
        self.single_layer = single_layer
        self._init_model(single_layer=single_layer)

    def _init_model(self, single_layer=False):
        self.model = GPT2Model.from_pretrained('gpt2')
        self.model = self.model.eval()
        # turn off all the gradients
        for param in self.model.parameters():
            param.requires_grad = self.finetune
        if single_layer:
            print ('SINGLE LAYER')
            self.feature_extractor = self.create_feature_extractor_single_layer() # data_dim -> hidden_dim
        else:
            print ('NOT SINGLE LAYER')
            self.mlp = nn.Linear(self.model.wte.embedding_dim, self.hidden_dim)
            self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim
            self.mlp.apply(weights_init)
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)

        ## NEW AUG 19, turn off bias training.
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
                               ])
    def create_feature_extractor_single_layer(self):
        return nn.Sequential(*[
            nn.Linear(self.model.wte.embedding_dim, self.latent_dim),
                               ])

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
                               ])

    def get_gpt2_embeddings(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        return gpt_emb

    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, gpt_emb):
        if self.single_layer:
            z = self.feature_extractor(gpt_emb)
        else:
            z = self.mlp(gpt_emb) # 32, 100
            z = self.feature_extractor(z)
        return z

    def forward(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)


class GPT2NeoXOUEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, finetune_gpt2=False, single_layer=False):
        super(GPT2NeoXOUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune_gpt2
        self.single_layer = single_layer
        self._init_model(single_layer=single_layer)

    def _init_model(self, single_layer=False):

        tokenizer_path = '/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/codon_wordlevel_100vocab_added.json'
        print (f'Loading tokenizer from {tokenizer_path}')
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tokenizer_path))
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print (f'New vocab size: {len(tokenizer)}')
        base_config = AutoConfig.from_pretrained('/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/neox_25,290,752.json')
        self.model = AutoModelForCausalLM.from_config(base_config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in self.model.parameters()).values())
        print (f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        a = torch.load('/home/jayfeather/language_modeling_via_stochastic_processes/genslm_model/patric_25m_epoch01-val_loss_0.57_bias_removed.pt')['state_dict']
        b = {}
        for k in a:
            #b[k.replace('module.model.', '')] = a[k]
            b[k.replace('model.', '')] = a[k]
        self.model.load_state_dict(b,False)
        self.model = self.model.gpt_neox
        self.model.resize_token_embeddings(len(tokenizer))

        # self.model = GPT2Model.from_pretrained('gpt2')
        # self.model = self.model.eval()
        # turn off all the gradients
        for param in self.model.parameters():
            param.requires_grad = self.finetune
        if single_layer:
            print ('SINGLE LAYER')
            self.feature_extractor = self.create_feature_extractor_single_layer() # data_dim -> hidden_dim
        else:
            print ('NOT SINGLE LAYER')
            self.mlp = nn.Linear(self.model.embed_in.embedding_dim, self.hidden_dim)
            self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim
            self.mlp.apply(weights_init)
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)

        ## NEW AUG 19, turn off bias training.
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
                               ])
    def create_feature_extractor_single_layer(self):
        return nn.Sequential(*[
            nn.Linear(self.model.embed_in.embedding_dim, self.latent_dim),
                               ])

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
                               ])

    def get_gpt2_embeddings(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        return gpt_emb

    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, gpt_emb):
        if self.single_layer:
            z = self.feature_extractor(gpt_emb)
        else:
            z = self.mlp(gpt_emb) # 32, 100
            z = self.feature_extractor(z)
        return z

    def forward(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)