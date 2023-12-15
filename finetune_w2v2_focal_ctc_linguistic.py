import torch
import numpy
import json
import torchaudio
import evaluate
from torch import nn
import transformers
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
import os
import numpy as np
from tqdm.auto import tqdm
from mdd.utils import phoneme_tokenizer, encode_phone, greedy_decode, VOCAB
from mdd.augmentation import SpeedPerturbation
from mdd.criterion import FocalCTCLoss
from torchaudio.transforms import MelSpectrogram
import wandb
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import CausalLMOutput
import pandas as pd
import torch.nn.functional as F
import argparse
import random
import jiwer
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--focal-alpha', type=float, default=0.99)
parser.add_argument('--focal-gamma', type=float, default=2)

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = 'put_token_here'

login(token=HF_TOKEN)

SAMPLING_RATE = 16000

spec_augment = True

pad_id = 0
ignore_value = -100

_HIDDEN_STATES_START_POSITION = 2

def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # cudnn_deterministic = True
    # cudnn_benchmark = False
    # print("cudnn_deterministic set to False")
    # print("cudnn_benchmark set to True")
    
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(random_seed)
    #     torch.backends.cudnn.deterministic = cudnn_deterministic
    #     torch.backends.cudnn.benchmark = cudnn_benchmark
    return

reproducibility(1211)



class SupervisedDataset(Dataset):
    def __init__(self, data_path, do_augment=False):
        super().__init__()
        self.data = json.load(open(data_path, encoding="utf-8"))
        self.n_fft = 512
        self.hop_len = 128
        self.n_mels = 80
        self.cal_mel = MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.n_mels,
        )
        self.do_augment = do_augment
        self.speed_pertub = SpeedPerturbation(SAMPLING_RATE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.data[idx]["path"])

        if self.do_augment:
            wav = self.speed_pertub(wav)

        ret_dict = dict(input_values=wav[0].numpy())

        if "transcript" in self.data[idx]:
            phoneme = phoneme_tokenizer(self.data[idx]["transcript"], sep=" ")
            ids = encode_phone(phoneme)
            ret_dict["labels"] = ids

        if "canonical" in self.data[idx]:
            canoncial_phoneme = phoneme_tokenizer(self.data[idx]["canonical"], sep=" ")
            canonical_ids = encode_phone(canoncial_phoneme)
            ret_dict["canonical_labels"] = canonical_ids

        if "tonal" in self.data[idx]:
            tonal_ids = torch.LongTensor(self.data[idx]["tonal"])
            ret_dict["tonal_labels"] = tonal_ids

        return ret_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    processor: Wav2Vec2Processor

    def __call__(self, features):
        audio = [i["input_values"] for i in features]

        batch = self.processor(
            audio=audio, padding=True, return_tensors="pt", sampling_rate=SAMPLING_RATE
        )

        if "labels" in features[0]:
            text = [i["labels"] for i in features]
            labels_batch = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
            labels = labels_batch.masked_fill(labels_batch.eq(pad_id), ignore_value)
            batch["labels"] = labels

        if "canonical_labels" in features[0]:
            canon_text = [i["canonical_labels"] for i in features]
            canon_labels_batch = torch.nn.utils.rnn.pad_sequence(
                canon_text, batch_first=True
            )
            # canonical_labels = canon_labels_batch.masked_fill(canon_labels_batch.eq(pad_id), ignore_value)
            batch["canonical_labels"] = canon_labels_batch

        if "tonal_labels" in features[0]:
            tonal = [i["tonal_labels"] for i in features]
            tonal_labels_batch = torch.nn.utils.rnn.pad_sequence(
                tonal, batch_first=True
            )
            tonal_labels = tonal_labels_batch.masked_fill(
                tonal_labels_batch.eq(pad_id), ignore_value
            )
            batch["tonal_labels"] = tonal_labels

        return batch


wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = (
        pred.label_ids if not isinstance(pred.label_ids, tuple) else pred.label_ids[0]
    )

    label_ids[label_ids == ignore_value] = pad_id

    pred_str = greedy_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = greedy_decode(label_ids)

    pred_str = [" ".join(i) for i in pred_str]
    label_str = [" ".join(i) for i in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


processor_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
# model_id = "./wav2vec2-base-finetune-vi_phone-non_freeze"
model_id = processor_id

vocab_size = len(VOCAB)

print("Vocab size:", vocab_size)

processor = Wav2Vec2Processor.from_pretrained(processor_id)
data_collator = DataCollatorForSupervisedDataset(processor=processor)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Wav2VecForLinguisticTonalForCTC(transformers.Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        self.wav2vec2 = transformers.Wav2Vec2Model(config)
        # self.dropout = nn.Dropout(config.final_dropout)

        num_tonals = 7
        # NormLayer = nn.LayerNorm
        NormLayer = RMSNorm

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )

        # linguistic part
        # embedding -> pos enc -> norm -> attention & add -> norm & ffn & add
        self.ling_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=output_hidden_size,
            padding_idx=pad_id,
        )
        self.pos_enc = PositionalEncoding(d_hid=output_hidden_size, n_position=128)
        self.mha = nn.MultiheadAttention(
            embed_dim=output_hidden_size,
            num_heads=output_hidden_size // 64,
            batch_first=True,
        )
        self.attn_norm = NormLayer(output_hidden_size, eps=1e-6)

        self.ffn_layer = nn.Sequential(
            NormLayer(output_hidden_size),
            nn.Linear(output_hidden_size, output_hidden_size * 2),
            SwiGLU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(output_hidden_size, output_hidden_size),
        )

        self.out_norm = NormLayer(output_hidden_size, eps=1e-6)

        self.lm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(output_hidden_size, config.vocab_size)
        )

        self.focal_alpha = args.focal_alpha
        self.focal_gamma = args.focal_gamma

        self.focal_ctc_loss = FocalCTCLoss(self.focal_alpha, self.focal_gamma)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        tonal_labels: Optional[torch.Tensor] = None,
        canonical_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        enc_states = outputs[0]

        ling_states = self.pos_enc(self.ling_emb(canonical_labels))
        ling_states = self.attn_norm(ling_states)

        # acoustic as query
        hidden_states = (
            self.mha(enc_states, ling_states, ling_states, need_weights=False)[0]
            + enc_states
        )

        hidden_states = self.ffn_layer(hidden_states) + hidden_states

        hidden_states = self.out_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = self.focal_ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model_configs = {}

if processor_id == model_id:
    model_configs["ignore_mismatched_sizes"] = True
    model_configs["ctc_loss_reduction"] = "mean"
    model_configs["pad_token_id"] = pad_id
    model_configs["vocab_size"] = vocab_size

if spec_augment:
    model_configs["mask_time_prob"] = 0.025
    model_configs["mask_time_length"] = 10
    model_configs["mask_feature_prob"] = 0.001
    model_configs["mask_feature_length"] = 16

model = Wav2VecForLinguisticTonalForCTC.from_pretrained(model_id, **model_configs)
# model = Wav2VecForLinguisticTonalForCTC.from_pretrained('./fine-w2v2base-bs8-ep200-lr2e-05-non-freeze-lr_cosine-red_aug-tonal_0.1-full-linguistic-rmsnorm')


# print(model)

data_path = "data/splitted_data"

train_dataset = SupervisedDataset(os.path.join(data_path, "train.json"), True)
eval_dataset = SupervisedDataset(os.path.join(data_path, "public_test.json"))
test_dataset = SupervisedDataset(os.path.join(data_path, "private_test.json"))

print("Train:", len(train_dataset))
print("Eval:", len(eval_dataset))
print("Test:", len(test_dataset))

# not freezing at all
# model.freeze_feature_encoder()

# print("Frezzing weights...")
# for p in model.wav2vec2.parameters():
#   p.requires_grad = False

continue_train = False
epochs = 100
accum_grads = 1
train_batchsize = 8
eval_batchsize = 16
save_steps = 100
log_steps = 100
eval_steps = 100
default_lr = 2e-5
lr_divide_factor = 1
label_smoothing = 0.0
warmup_ratio = 0.1
log_result = True
focal_alpha = model.focal_alpha
focal_gamma = model.focal_gamma

timenow = datetime.datetime.now()

# warmup_steps = round(len(train_dataset) / (train_batchsize * accum_grads) / 4 * epochs * 0.1)
# no tonal by default
focal_part = f'a{focal_alpha}_g{focal_gamma}'
run_name = f'w2v2_ablation_focal_ctc_{focal_part}-best_on-ling_head-tp0.025_tl10_fp0.001_fl16'
# run_name = f"fine-w2v2base-bs{train_batchsize}-ep{epochs}-lr{default_lr}-linguistic-rmsnorm-focal_ctc_{add_part}-0.05_10_0.004_40-final"

timenow = datetime.datetime.now()

with open('list_run_till_now.txt', 'a') as f:
    f.write(run_name + ' - ' + str(timenow) + '\n')

# can try layernorm
if log_result:
    os.environ["WANDB_PROJECT"] = "md_d_vlsp_2023"  # name your W&B project

print("Run name:", run_name)

training_args = TrainingArguments(
    output_dir=run_name,
    group_by_length=False,
    per_device_train_batch_size=train_batchsize,
    per_device_eval_batch_size=eval_batchsize,
    eval_accumulation_steps=eval_batchsize,
    gradient_accumulation_steps=accum_grads,
    evaluation_strategy="steps",
    num_train_epochs=epochs,
    gradient_checkpointing=bool(accum_grads > 1),
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.98,
    ddp_find_unused_parameters=False,
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=log_steps,
    learning_rate=default_lr / lr_divide_factor,
    label_smoothing_factor=label_smoothing,
    warmup_ratio=warmup_ratio,
    save_total_limit=2,
    push_to_hub=True,
    torch_compile=False,
    resume_from_checkpoint=continue_train,
    report_to="wandb" if log_result else "none",
    run_name=run_name,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_wer",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total params:", total_params)
print(
    "Trainable params:",
    trainable_params,
    "% trainable:",
    trainable_params / total_params,
)

trainer.train(resume_from_checkpoint=continue_train)
trainer.save_model()
trainer.save_state()
trainer.push_to_hub()

# output = trainer.predict(test_dataset)

# print("Output:", output)

# torch.save(output, "private_test_predict.pt")

# predict = greedy_decode(np.argmax(output.predictions, axis=-1))

# predictions = []
# for datum, pred in zip(test_dataset.data, predict):
#     path = datum["path"]
#     path = path.split("VMD-VLSP23-private-test")[-1]
#     predictions.append({"id": datum["id"], "path": path, "predict": " ".join(pred)})

# df = pd.DataFrame(predictions)
# df.to_csv(f"private_test_submission_final.csv", index=False)

# # os.system("python fix_vi_ftfy.py")
