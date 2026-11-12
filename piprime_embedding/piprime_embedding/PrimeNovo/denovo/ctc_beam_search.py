# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from ctcdecode import CTCBeamDecoder
    _HAS_CTCDECODE = True
except Exception:
    CTCBeamDecoder = None  # type: ignore
    _HAS_CTCDECODE = False

import torch
from torch import TensorType
from .ctc_decoder_base import CTCDecoderBase
from typing import Dict, List
#from fairseq.data.ctc_dictionary import CTCDictionary


class CTCBeamSearchDecoder(CTCDecoderBase):
    """
    CTC Beam Search Decoder
    """

    def __init__(self, decoder, decoder_parameters: Dict) -> None:
        super().__init__(decoder)
        self.attn_decoder = decoder 
        # Use native ctcdecode when available, otherwise fall back to a simple greedy CTC decoder
        if _HAS_CTCDECODE:
            self.decoder = CTCBeamDecoder(
                decoder.get_symbols(),
                model_path=None,
                alpha=0,
                beta=0,
                cutoff_top_n=30,
                cutoff_prob=1,
                beam_width=decoder_parameters["beam"],
                num_processes=4,
                blank_id=decoder.get_blank_idx(),
                log_probs_input=False,
            )  # This is true since our criteria script returns log_prob.
            print("notice here: ", self.decoder._beam_width)
            self._use_fallback = False
        else:
            # Flag to use fallback greedy decoding when ctcdecode is unavailable
            self.decoder = None
            self._use_fallback = True
        
    def decode(self, log_prob: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function for the CTC beam search decoder.
        """
        if not self._use_fallback:
            beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(log_prob)
            top_beam_tokens = beam_results[:, 0, :]  # extract the most probable beam
            top_beam_len = out_lens[:, 0]
            mask = (
                torch.arange(0, top_beam_tokens.size(1))
                .type_as(top_beam_len)
                .repeat(top_beam_len.size(0), 1)
                .lt(top_beam_len.unsqueeze(1))
            )
            top_beam_tokens[~mask] = self.attn_decoder.get_pad_idx()  # mask out nonsense index with pad index.
            return top_beam_tokens, beam_scores[:, 0]

        # Fallback greedy CTC decoding (no external ctcdecode dependency)
        with torch.no_grad():
            # log_prob: (B, T, V)
            best_tokens = torch.argmax(log_prob, dim=2)  # (B, T)
            batch_size, max_len = best_tokens.size(0), best_tokens.size(1)
            pad_idx = self.attn_decoder.get_pad_idx()
            blank_id = self.attn_decoder.get_blank_idx()

            out_tokens = []
            out_scores = []
            for b in range(batch_size):
                seq = best_tokens[b].tolist()
                collapsed = []
                prev = None
                for t in seq:
                    if t == blank_id:
                        prev = t
                        continue
                    if prev == t:
                        # collapse repeats
                        continue
                    collapsed.append(t)
                    prev = t
                if len(collapsed) < max_len:
                    collapsed = collapsed + [pad_idx] * (max_len - len(collapsed))
                else:
                    collapsed = collapsed[:max_len]
                out_tokens.append(collapsed)
                out_scores.append(0.0)

            top_beam_tokens = torch.tensor(out_tokens, device=log_prob.device, dtype=best_tokens.dtype)
            beam_scores = torch.tensor(out_scores, device=log_prob.device, dtype=log_prob.dtype)
            return top_beam_tokens, beam_scores

    def decode_topk(self, log_prob: TensorType, topk: int):
        """
        Return top-k beams' tokens and scores.
        If fallback greedy is used, duplicate top-1 into k beams.
        Returns: (tokens_k, scores_k) where tokens_k is (B, K, T), scores_k is (B, K)
        """
        if topk <= 1:
            toks, scr = self.decode(log_prob)
            return toks.unsqueeze(1), scr.unsqueeze(1)

        if not self._use_fallback:
            beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(log_prob)
            K = min(topk, beam_results.size(1))
            beam_results = beam_results[:, :K, :].clone()
            out_lens = out_lens[:, :K]
            B, K, T = beam_results.shape
            pad_idx = self.attn_decoder.get_pad_idx()
            for b in range(B):
                for k in range(K):
                    L = out_lens[b, k]
                    if L < T:
                        beam_results[b, k, L:] = pad_idx
            return beam_results, beam_scores[:, :K]

        toks, scr = self.decode(log_prob)
        B, T = toks.shape
        toks_k = toks.unsqueeze(1).repeat(1, topk, 1)
        scr_k = scr.unsqueeze(1).repeat(1, topk)
        return toks_k, scr_k
