import random
import numpy as np
from tqdm import tnrange
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]


# there are some things from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
class TrDecoder(nn.Module):
    def __init__(self, dec_embed_dim, dec_depth, dec_n_heads, dec_feedforward_dim):
        super(TrDecoder, self).__init__()

        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_n_heads = dec_n_heads
        self.dec_feedforward_dim = dec_feedforward_dim

        self.transformer_layers = nn.ModuleList([nn.TransformerDecoderLayer(dec_embed_dim,
                                                                            dec_n_heads,
                                                                            dim_feedforward=dec_feedforward_dim,
                                                                            activation='gelu') for i in
                                                 range(dec_depth)])

    def forward(self, embs, mem, seq_mask, pad_mask=None):
        embs = embs.transpose(0, 1)

        for i in range(self.dec_depth):
            embs = self.transformer_layers[i](embs, mem, tgt_mask=seq_mask, memory_key_padding_mask=pad_mask)

        return embs.transpose(0, 1)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class MDVAE(nn.Module):
    def __init__(self, input_size, enc_depth, enc_n_heads, enc_feedforward_dim,
                 dec_num_embeds, dec_embed_dim, dec_depth, dec_n_heads, dec_feedforward_dim,
                 distr_dim, dict_constants, mode='VAE'):
        super(MDVAE, self).__init__()

        # hyperparameters:
        self.input_size = input_size
        self.enc_depth = enc_depth
        self.enc_n_heads = enc_n_heads
        self.enc_feedforward_dim = enc_feedforward_dim
        self.dec_num_embeds = dec_num_embeds
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_n_heads = dec_n_heads
        self.dec_feedforward_dim = dec_feedforward_dim
        self.distr_dim = distr_dim
        self.mode = mode
        self.DEC_SOS_IND = dict_constants['DEC_SOS_IND']
        self.DEC_EOS_IND = dict_constants['DEC_EOS_IND']
        self.MAX_SUMMARY_LEN = dict_constants['MAX_SUMMARY_LEN']
        self.MAX_DOC_LEN = dict_constants['MAX_DOC_LEN']

        # encoder layers:
        self.enc_pos_emb = PositionalEncoding(input_size, 0.0, self.MAX_DOC_LEN)
        self.mu_linear = nn.Linear(input_size, distr_dim)
        self.logsigma_linear = None
        if mode == 'VAE':
            self.logsigma_linear = nn.Linear(input_size, distr_dim)
        elif mode != 'AE':
            raise ValueError('Unknown mode: must be "VAE" or "AE", not {}'.format(mode))
        enc_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                               nhead=enc_n_heads,
                                               dim_feedforward=enc_feedforward_dim,
                                               activation="gelu")
        self.document_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)
        self.doc_encoder_linear = nn.Linear(input_size * 2, 1, bias=False)

        # decoder layers:
        self.decoder_emb = nn.Embedding(dec_num_embeds, dec_embed_dim)
        self.dec_pos_emb = PositionalEncoding(dec_embed_dim, 0.0, self.MAX_SUMMARY_LEN)
        self.dec_enc_to_mem = nn.Linear(input_size, dec_embed_dim)
        self.dec_latent_to_add_mem = nn.Linear(distr_dim, dec_embed_dim)
        self.decoder = TrDecoder(self.dec_embed_dim, self.dec_depth, self.dec_n_heads, self.dec_feedforward_dim)
        seq_mask = self.decoder.generate_square_subsequent_mask(self.MAX_SUMMARY_LEN)
        self.register_buffer('seq_mask', seq_mask)
        self.decoder_linear1 = nn.Linear(dec_embed_dim, dec_embed_dim)
        self.decoder_linear2 = nn.Linear(dec_embed_dim, dec_num_embeds)

    def encode_vae(self, inputs, pad_mask=None):
        # inputs: (n_docs, n_tokens, input_size)

        # BERT embeddings per document + positional embeddings to document embeddings:
        x = self.enc_pos_emb(inputs).transpose(0, 1)
        x = self.document_encoder(x, src_key_padding_mask=pad_mask).transpose(0, 1)
        if pad_mask is None:
            doc_embeds = torch.mean(x, dim=1)
        else:
            not_pad_mask = torch.logical_not(pad_mask)
            doc_embeds = torch.sum(x * not_pad_mask.unsqueeze(2), dim=1) / torch.sum(not_pad_mask, dim=1).view(-1, 1)

        # document embeddings to set-of-documents embedding:
        if doc_embeds.shape[0] == 1:  # only 1 document in inputs
            set_of_docs_emb = doc_embeds
        else:  # more than 1 document in inputs
            d_sum = torch.sum(doc_embeds, dim=0)
            d_conc = torch.cat((doc_embeds, d_sum.view(1, -1).expand(doc_embeds.shape[0], -1)), -1)
            weights = self.doc_encoder_linear(d_conc)
            weights = F.softmax(weights, dim=0)
            set_of_docs_emb = torch.mm(weights.view(1, -1), doc_embeds)

        # set-of-documents embedding to mu and logvar:
        mu = self.mu_linear(set_of_docs_emb)
        logsigma = self.logsigma_linear(set_of_docs_emb)
        if pad_mask is None:
            return mu, logsigma, (x.reshape((-1, x.shape[2])).unsqueeze(1), None)
        else:
            return mu, logsigma, (x.reshape((-1, x.shape[2])).unsqueeze(1), pad_mask.reshape((1, -1)))

    def encode_ae(self, inputs, pad_mask=None):
        # inputs: (n_docs, n_tokens, input_size)

        # BERT embeddings per document + positional embeddings to document embeddings:
        x = self.enc_pos_emb(inputs).transpose(0, 1)
        x = self.document_encoder(x, src_key_padding_mask=pad_mask).transpose(0, 1)
        if pad_mask is None:
            doc_embeds = torch.mean(x, dim=1)
        else:
            not_pad_mask = torch.logical_not(pad_mask)
            doc_embeds = torch.sum(x * not_pad_mask.unsqueeze(2), dim=1) / torch.sum(not_pad_mask, dim=1).view(-1, 1)

        # document embeddings to set-of-documents embedding:
        if doc_embeds.shape[0] == 1:  # only 1 document in inputs
            set_of_docs_emb = doc_embeds
        else:  # more than 1 document in inputs
            d_sum = torch.sum(doc_embeds, dim=0)
            d_conc = torch.cat((doc_embeds, d_sum.view(1, -1).expand(doc_embeds.shape[0], -1)), -1)
            weights = self.doc_encoder_linear(d_conc)
            weights = F.softmax(weights, dim=0)
            set_of_docs_emb = torch.mm(weights.view(1, -1), doc_embeds)

        # set-of-documents embedding to mu:
        mu = self.mu_linear(set_of_docs_emb)
        if pad_mask is None:
            return mu, (x.reshape((-1, x.shape[2])).unsqueeze(1), None)
        else:
            return mu, (x.reshape((-1, x.shape[2])).unsqueeze(1), pad_mask.reshape((1, -1)))

    def reparam_sample(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, memory, pad_mask=None, target_summary=None, max_summary_len=None):
        # teacher forcing:
        if target_summary is not None:
            if target_summary.shape[1] > self.MAX_SUMMARY_LEN:
                raise ValueError("target_summary's len must be <= MAX_SUMMARY_LEN = {}".format(self.MAX_SUMMARY_LEN))
            embs = self.decoder_emb(target_summary)
            embs = self.dec_pos_emb(embs)
            mem = self.dec_enc_to_mem(memory)
            add_mem = self.dec_latent_to_add_mem(z.view(1, 1, -1))
            embs = embs + add_mem
            decoded = self.decoder(embs, mem=mem, seq_mask=self.seq_mask[:embs.shape[1], :embs.shape[1]],
                                   pad_mask=pad_mask)
            return F.log_softmax(self.decoder_linear2(F.relu(self.decoder_linear1(decoded))), dim=-1)
        # autoregressive one-by-one:
        else:
            if max_summary_len is None:
                max_len = self.MAX_SUMMARY_LEN
            else:
                max_len = min(max_summary_len, self.MAX_SUMMARY_LEN)
            inds = [self.DEC_SOS_IND]
            embs = self.decoder_emb(torch.tensor([inds], dtype=torch.long, device=z.device))
            embs = self.dec_pos_emb(embs)
            mem = self.dec_enc_to_mem(memory)
            add_mem = self.dec_latent_to_add_mem(z.view(1, 1, -1))
            embs = embs + add_mem
            log_probs = None
            for i in range(max_len - 2):
                decoded = self.decoder(embs, mem=mem, seq_mask=self.seq_mask[:embs.shape[1], :embs.shape[1]],
                                       pad_mask=pad_mask)
                log_probs = F.log_softmax(self.decoder_linear2(F.relu(self.decoder_linear1(decoded))), dim=-1)
                next_ind = torch.argmax(log_probs, dim=-1).view(-1)[-1].item()
                # next_ind = torch.multinomial(torch.exp(log_probs[0, -1].view(-1)), 1).item()
                if next_ind == self.DEC_EOS_IND:
                    return log_probs
                inds.append(next_ind)
                embs = self.decoder_emb(torch.tensor([inds], dtype=torch.long, device=z.device))
                embs = self.dec_pos_emb(embs)
                embs = embs + add_mem
            return log_probs

    def forward(self, inputs, pad_mask=None, summary=None, max_summary_len=None, deterministic=False):
        # inputs: (n_docs, n_sentences, input_size)

        if self.mode == 'VAE':
            if not deterministic:
                mu, logsigma, (mem, mem_pad) = self.encode_vae(inputs, pad_mask)
                z = self.reparam_sample(mu, logsigma)
                return [self.decode(z, mem, mem_pad, summary, max_summary_len), mu, logsigma]
            else:
                mu, logsigma, (mem, mem_pad) = self.encode_vae(inputs, pad_mask)
                return [self.decode(mu, mem, mem_pad, summary, max_summary_len), mu, logsigma]
        else:
            z, (mem, mem_pad) = self.encode_ae(inputs, pad_mask)
            return self.decode(z, mem, mem_pad, summary, max_summary_len), z

    # generate text summary
    def summarize(self, inp, dec_vocab_ind2word, pad_mask=None, deterministic=True, max_summary_len=None):
        # deterministic: latent vector z equals mu if True, else z is sampled

        with torch.no_grad():
            m = self.training
            self.eval()
            if deterministic or self.mode == 'AE':
                z, (mem, mem_pad) = self.encode_ae(inp, pad_mask)
            else:
                mu, logsigma, (mem, mem_pad) = self.encode_vae(inp, pad_mask)
                z = self.reparam_sample(mu, logsigma)
            summary_inds = torch.reshape(torch.argmax(self.decode(z, mem, mem_pad, None, max_summary_len), dim=-1),
                                         (-1,))
            summary = ''
            for ind in summary_inds:
                summary += dec_vocab_ind2word[ind.item()] + ' '
            self.train(m)
            return summary[:-1]

    def full_vae_loss(self, pred, target, mu, logsigma, reduction='mean'):
        nll = F.nll_loss(pred.squeeze(0), target.view(-1), reduction=reduction)
        kl = -0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp())
        if reduction == 'mean':
            kl_weight = 1 / target.view(-1).shape[0]
        else:
            kl_weight = 1
        full = nll + kl_weight * kl
        # full = nll + kl
        return full, nll, kl
