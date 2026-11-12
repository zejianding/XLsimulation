# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified CTC Beam Search Decoder that returns top-K candidates
支持返回top-K个候选peptide的CTC Beam Search解码器
"""

import torch
from torch import TensorType
from .ctc_decoder_base import CTCDecoderBase
from typing import Dict, List, Tuple
import numpy as np
import sys


class CTCBeamSearchDecoderTopK(CTCDecoderBase):
    """
    CTC Beam Search Decoder - Returns Top-K Candidates
    返回top-K个候选序列的CTC Beam Search解码器
    """

    def __init__(self, decoder, decoder_parameters: Dict) -> None:
        super().__init__(decoder)
        self.attn_decoder = decoder
        self.symbols = decoder.get_symbols()
        self.beam_width = decoder_parameters.get("beam", 100)
        self.top_k = decoder_parameters.get("top_k", 100)  # 返回top-K个候选
        self.blank_id = decoder.get_blank_idx()
        self.cutoff_top_n = 30
        sys.stderr.write(f"Top-K CTC Decoder initialized with beam_width: {self.beam_width}, top_k: {self.top_k}\n")

    def decode(self, log_prob: TensorType, **kwargs) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """
        Decoding function that returns top-K candidates.
        
        Args:
            log_prob: (batch_size, seq_len, vocab_size) log probabilities
            
        Returns:
            all_candidates: (batch_size, top_k, max_len) top-K decoded sequences
            all_scores: (batch_size, top_k) scores for each candidate
        """
        batch_size, seq_len, vocab_size = log_prob.shape
        device = log_prob.device
        
        # Convert to probabilities
        probs = torch.exp(log_prob)
        
        # Process each batch independently
        all_batch_candidates = []
        all_batch_scores = []
        
        for batch_idx in range(batch_size):
            batch_probs = probs[batch_idx]  # (seq_len, vocab_size)
            
            # Perform beam search for this batch
            top_k_results = self._beam_search_top_k(
                batch_probs, 
                self.beam_width,
                self.top_k,
                self.blank_id,
                self.cutoff_top_n
            )
            
            candidates = [tokens for tokens, _ in top_k_results]
            scores = [score for _, score in top_k_results]
            
            all_batch_candidates.append(candidates)
            all_batch_scores.append(scores)
        
        # Convert to tensors and pad
        max_len = max(
            max(len(tokens) for tokens in batch_candidates) 
            for batch_candidates in all_batch_candidates
        )
        
        # Pad sequences
        padded_candidates = []
        for batch_candidates in all_batch_candidates:
            batch_padded = []
            for tokens in batch_candidates:
                padded = tokens + [self.attn_decoder.get_pad_idx()] * (max_len - len(tokens))
                batch_padded.append(padded)
            padded_candidates.append(batch_padded)
        
        # Convert to tensors
        top_k_tokens = torch.tensor(padded_candidates, dtype=torch.long, device=device)
        top_k_scores = torch.tensor(all_batch_scores, dtype=torch.float, device=device)
        
        return top_k_tokens, top_k_scores

    def _beam_search_top_k(self, probs: torch.Tensor, beam_width: int, top_k: int,
                           blank_id: int, cutoff_top_n: int) -> List[Tuple[List[int], float]]:
        """
        Perform beam search and return top-K candidates.
        
        Args:
            probs: (seq_len, vocab_size) probabilities
            beam_width: beam width for search
            top_k: number of candidates to return
            blank_id: index of blank token
            cutoff_top_n: only consider top N tokens at each step
            
        Returns:
            top_k_results: list of (tokens, score) tuples, sorted by score
        """
        seq_len, vocab_size = probs.shape
        
        # Initialize beams: (prefix, last_token, score)
        beams = {(tuple(), blank_id): 0.0}
        
        for t in range(seq_len):
            new_beams = {}
            
            # Get top-k tokens at this timestep
            top_probs, top_indices = torch.topk(probs[t], min(cutoff_top_n, vocab_size))
            
            for (prefix, last_token), score in beams.items():
                for prob, token_idx in zip(top_probs, top_indices):
                    token_idx = token_idx.item()
                    prob = prob.item()
                    
                    if prob < 1e-10:
                        continue
                    
                    new_score = score + np.log(prob + 1e-10)
                    
                    if token_idx == blank_id:
                        # Blank token - don't extend prefix
                        key = (prefix, blank_id)
                        if key not in new_beams or new_beams[key] < new_score:
                            new_beams[key] = new_score
                    else:
                        # Non-blank token
                        if token_idx == last_token:
                            # Same as last token - don't extend prefix
                            key = (prefix, token_idx)
                        else:
                            # Different token - extend prefix
                            key = (prefix + (token_idx,), token_idx)
                        
                        if key not in new_beams or new_beams[key] < new_score:
                            new_beams[key] = new_score
            
            # Prune beams - keep only top beam_width
            if len(new_beams) > beam_width:
                sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                beams = dict(sorted_beams[:beam_width])
            else:
                beams = new_beams
        
        # Get top-K beams
        if not beams:
            return [([], 0.0)]
        
        sorted_beams = sorted(beams.items(), key=lambda x: x[1], reverse=True)
        top_k_beams = sorted_beams[:min(top_k, len(sorted_beams))]
        
        # Extract results
        results = [(list(prefix), score) for (prefix, _), score in top_k_beams]
        
        return results