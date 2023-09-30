from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = ""
        for ind in inds:
            if self.ind2char[ind] == self.EMPTY_TOK:
                continue
            if len(result) == 0:
                result += self.ind2char[ind]
                continue
            if self.ind2char[ind] == result[-1]:
                continue
            result += self.ind2char[ind]
        return result

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        for frame in probs:
            hypos = self.extend_and_merge(frame, hypos)
            hypos = self.truncate(hypos, beam_size)
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
    def extend_and_merge(self, frame, hypos):
        new_hypos = []
        for next_char_index, next_char_proba in enumerate(frame):
            next_char = self.ind2char[next_char_index]
            for hypo in hypos:
                last_char = hypo.text[-1]
                if last_char == next_char or next_char == self.EMPTY_TOK:
                    new_text = hypo.text
                else:
                    new_text = hypo.text + next_char
                new_prob = hypo.prob * next_char_proba
                new_hypos.append(Hypothesis(new_text, new_prob))
        return new_hypos
    
    def truncate(self, hypos, beam_size):
        return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

