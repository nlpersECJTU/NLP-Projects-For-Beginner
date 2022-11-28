# by wucx
# 2022-11-23

import torch
import numpy as np
from utils import make_pad_mask, make_tril_mask


def batch_greedy_decode(model, src, max_len=64, sos=2, eos=3):
    batch_size = src.size(0)
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0
    device = model.device

    src_mask = make_pad_mask(src, src, model.src_pad_idx, model.src_pad_idx)
    src_enc = model.encoder(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(sos).type_as(src.data)

    for i in range(max_len):
        tgt_src_mask = make_pad_mask(tgt, src, model.tgt_pad_idx, model.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, model.tgt_pad_idx, model.tgt_pad_idx) * make_tril_mask(tgt, tgt).to(device)
        out = model.decoder(tgt, 
                            src_enc,
                            tgt_mask, 
                            tgt_src_mask)
        
        # the predicted words 
        out = out[:, -1, :]
        pred_words = torch.argmax(out, dim=-1)
        # pad the predicted words to tgt
        tgt = torch.cat((tgt, pred_words.unsqueeze(1)), dim=-1)
        
        preds = pred_words.cpu().numpy()
        for j in range(batch_size):
            if stop_flag[j] is False:
                if preds[j] == eos:
                    count += 1
                    stop_flag[j] = True
                else:
                    results[j].append(preds[j].item())

        if count == batch_size:
            break

    return results


# Warning: the method 'greedy_decode' is not tested
def greedy_decode(model, src, max_len=64, sos=2, eos=3):
    """ 
        Greedy decode one sentence 
    """
    
    device = model.device
    src_mask = make_pad_mask(src, src, model.src_pad_idx, model.src_pad_idx)
    src_enc = model.encoder(src, src_mask)
    tgt = torch.ones(1, 1).fill_(sos).type_as(src.data)

    for i in range(max_len):
        tgt_src_mask = make_pad_mask(tgt, src, model.tgt_pad_idx, model.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, model.tgt_pad_idx, model.tgt_pad_idx) * make_tril_mask(tgt, tgt).to(device)
        out = model.decoder(tgt, 
                            src_enc,
                            tgt_mask,
                            tgt_src_mask)
        
        # the predicted word
        out = out[:, -1, :]
        pred_word = torch.argmax(out, dim=-1)
        if pred_word == eos:
            break
        # pad the predicted word to tgt 
        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(pred_word)], dim=-1)
    
    return tgt.cpu().numpy()



class BeamHypotheses(object):
    
    def __init__(self, beam_size, max_len, length_penalty=0.7):
        """
        Initialize n-best list of hypotheses
        :param length_penalty, for 'beam search length normalization', please google it
        https://www.cnblogs.com/dangui/p/14691132.html
        """
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list
        """
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        if len(self) < self.beam_size or score > self.worst_score:
            # update beams
            self.beams.append((score, hyp))
            if len(self) > self.beam_size:
                # delete the hyp with the worst score
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    
    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        Is generating translation finished?
        :param best_sum_logprobs, the best score of new generated hypotheses
        """
        if len(self) < self.beam_size:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_len
            cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret



def beam_search(model, src, beam_size=3, length_penalty=0.7, max_len=64, sos=2, eos=3):
    """
    https://zhuanlan.zhihu.com/p/114669778
    """
    batch_size = src.size(0)
    device = model.device
    softmax = torch.nn.Softmax(dim=-1)
    generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty) for _ in range(batch_size)]
    # scores of generated_hyps
    beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
    beam_scores = beam_scores.view(-1)

    # Is translating each source sentence done ?
    done = [False for _ in range(batch_size)]

    # encode src
    src_mask = make_pad_mask(src, src, model.src_pad_idx, model.src_pad_idx)
    src_enc = model.encoder(src, src_mask)
    # repeated as (batch_size * beam_size, length, d_model)
    src_enc = src_enc.unsqueeze(1).repeat(1, beam_size, 1, 1)
    src_enc = src_enc.view(-1, src_enc.size(-2), src_enc.size(-1))
    # to compute mask
    src_ = src.unsqueeze(1).repeat(1, beam_size, 1).view(-1, src.size(-1))

    # init tgt input, batch_size * beam_size
    # generated all hypotheses in parallel
    tgt = torch.Tensor(batch_size * beam_size, 1).fill_(sos).type_as(src.data)
    cur_len = 1
    
    for i in range(max_len):
        tgt_src_mask = make_pad_mask(tgt, src_, model.tgt_pad_idx, model.src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, model.tgt_pad_idx, model.tgt_pad_idx) * make_tril_mask(tgt, tgt).to(device)
        out_logits = model.decoder(tgt, 
                            src_enc,
                            tgt_mask, 
                            tgt_src_mask)
        out_log_scores = torch.log(softmax(out_logits))

        # TODO

        # how to compute the score of each hyp={t1, t2, ..., tn}?
        # p(hyp) = p(t1|src) * p(t2| src, t1) * ... * p(tn|src, t1, t2, ..., t_n-1)
        # score(hyp) = log p(hyp) = log p(t1|src) + log p(t2| src, t1) + ... + log p(tn|src, t1, t2, ..., t_n-1)
        # after log, we can just add (instead of multiply) the score the next token
        scores = out_log_scores[:, -1, :]
        next_scores = scores + beam_scores.unsqueeze(-1)

        # reshape next_scores as (batch_size, beam_size * vocab_size)
        # to retrieve the top-k scores and their corresponding token conveniently
        next_scores = next_scores.view(batch_size, 
                                       beam_size * model.dec_voc_size)
        next_scores, next_tokens = torch.topk(next_scores, 
                                              2 * beam_size,
                                              dim=1, 
                                              largest=True,
                                              sorted=True)
        
        # constuct the next tgt by cat the predicted next_tokens
        # each element in next_batch_beam (score, token_id, beam_id)
        next_batch_beam = []
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                next_batch_beam.extend([(0, model.tgt_pad_idx, 0)] * beam_size)
                continue

            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                            zip(next_tokens[batch_idx], next_scores[batch_idx])):

                beam_id = torch.div(beam_token_id, model.dec_voc_size, rounding_mode='trunc')
                token_id = beam_token_id % model.dec_voc_size
                effective_beam_id = batch_idx * beam_size + beam_id

                if (token_id.item() == eos):
                    # if beam token does not belong to top beam_size tokens, it should not be added
                    if beam_token_rank >= beam_size:
                        continue
                    # add hyp
                    generated_hyps[batch_idx].add(
                        tgt[effective_beam_id].clone(), beam_token_score.item()
                    )
                else:
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                if len(next_sent_beam) == beam_size:
                    break

            # 
            done[batch_idx] = done[batch_idx] or \
                generated_hyps[batch_idx].is_done(next_scores[batch_idx].max().item(), cur_len)
            next_batch_beam.extend(next_sent_beam)
        
        # 
        if all(done):
            break

        # 
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = tgt.new([x[1] for x in next_batch_beam])
        beam_idx = tgt.new([x[2] for x in next_batch_beam])

        # pad the predicted words to prepare the next tgt
        tgt = tgt[beam_idx, :]
        tgt = torch.cat([tgt, beam_tokens.unsqueeze(1)], dim=-1)

        # update length
        cur_len = cur_len + 1

    # beam search is done, then prepare output
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        
        # some sentences may not ended with eos
        for beam_id in range(beam_size):
            effective_beam_id = batch_idx * beam_size + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = tgt[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    
    # output the best one here, as List  
    # we can also return the top-k best ones
    results = [[] for _ in range(batch_size)]
    for i, hyps in enumerate(generated_hyps):
        sorted_hyps = sorted(hyps.beams, key=lambda x: x[0])
        best_hyp = sorted_hyps.pop()[1]
        best_hyp = [x.item() for x in best_hyp]
        results[i] = best_hyp

    return results
    