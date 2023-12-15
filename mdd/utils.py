import torch
import numpy as np

VOCAB_PATH = "mdd/db/vi_phonemes.txt"


def read_vocab():
    return open(VOCAB_PATH, encoding="utf-8").read().strip().split(" ")


VOCAB = read_vocab()
LEN_VOCAB = len(VOCAB)
PHONE2IDS = dict(zip(VOCAB, range(LEN_VOCAB)))
IDS2PHONE = dict(zip(range(LEN_VOCAB), VOCAB))


def word_tokenizer(seq):
    return seq.split(" $ ")


def phoneme_tokenizer(seq, sep="|"):
    return sep.join(word_tokenizer(seq)).split(" ")


def encode_phone(seq):
    return torch.LongTensor([PHONE2IDS[i] for i in seq])


# 0 is <pad>
def decode_phone(ids, skip_special_tokens=True):
    return [IDS2PHONE[i] for i in ids if not (i == 0 and skip_special_tokens)]


def greedy_decode(argmax):
    shape = argmax.shape
    # if isinstance(argmax, np.ndarray):
    #     shape = argmax.shape
    # elif isinstance(argmax, torch.Tensor):
    #     shape = argmax.size()
    # if len(shape) == 3:
    #     argmax = argmax.argmax(dim=-1)
    # elif len(shape) == 1:
    #     argmax = argmax.unsqueeze(0)

    outputs = []
    for i in argmax:
        out = decode_phone(i.tolist())
        outputs.append(out)

    return outputs
