import torch
import numpy
import json
import torchaudio
import evaluate
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import TrainingArguments, Trainer
from huggingface_hub import login
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from mdd.utils import phoneme_tokenizer, encode_phone, greedy_decode, VOCAB
from mdd.augmentation import SpeedPerturbation
from torchaudio.transforms import MelSpectrogram
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

login(token="<hf_token>")

SAMPLING_RATE = 16000

spec_augment = True

pad_id = 0
ignore_value = -100


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

        if "transcript" in self.data[idx]:
            phoneme = phoneme_tokenizer(self.data[idx]["transcript"], sep=" ")
            ids = encode_phone(phoneme)
            return dict(input_values=wav[0].numpy(), labels=ids)

        return dict(input_values=wav[0].numpy())


@dataclass
class DataCollatorForSupervisedDataset(object):
    processor: Wav2Vec2Processor

    def __call__(self, features):
        have_label = "labels" in features

        audio = [i["input_values"] for i in features]

        if have_label:
            text = [i["labels"] for i in features]

        batch = self.processor(
            audio=audio, padding=True, return_tensors="pt", sampling_rate=SAMPLING_RATE
        )

        if have_label:
            labels_batch = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
            labels = labels_batch.masked_fill(labels_batch.eq(pad_id), ignore_value)
            batch["labels"] = labels

        return batch


wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == ignore_value] = pad_id

    pred_str = greedy_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = greedy_decode(pred.label_ids)

    pred_str = [" ".join(i) for i in pred_str]
    label_str = [" ".join(i) for i in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


processor_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
model_id = "./wav2vec2-base-finetune-vi_phone-non_freeze-spec_aug-500epoch"
# model_id = processor_id

vocab_size = len(VOCAB)

print("Vocab size:", vocab_size)

processor = Wav2Vec2Processor.from_pretrained(processor_id)
data_collator = DataCollatorForSupervisedDataset(processor=processor)

model_configs = {}

if processor_id == model_id:
    model_configs["ignore_mismatched_sizes"] = True
    model_configs["ctc_loss_reduction"] = "mean"
    model_configs["pad_token_id"] = pad_id
    model_configs["vocab_size"] = vocab_size

if spec_augment:
    model_configs["mask_time_prob"] = 0.065
    model_configs["mask_time_length"] = 10
    model_configs["mask_feature_prob"] = 0.012
    model_configs["mask_feature_length"] = 64

# model = Wav2Vec2ForCTC.from_pretrained(model_id, **model_configs)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# data_path = "/data/tuanio/data/share_with_150/data_vlsp_md_d_2023/splitted_data_113"
data_path = "/data/tuanio/projects/md_d_vlsp2023/data"

train_dataset = SupervisedDataset(os.path.join(data_path, "train.json"), True)
eval_dataset = SupervisedDataset(os.path.join(data_path, "public_test.json"))
test_dataset = SupervisedDataset(os.path.join(data_path, "private_test.json"))

print("Train:", len(train_dataset))
print("Eval:", len(eval_dataset))
print("Test:", len(test_dataset))

# not freezing at all
model.freeze_feature_encoder()

# print("Frezzing weights...")
# for p in model.wav2vec2.parameters():
#   p.requires_grad = False

continue_train = False
epochs = 200
accum_grads = 1
train_batchsize = 8
eval_batchsize = 64
save_steps = 20
log_steps = 20
eval_steps = 60
default_lr = 3e-4
lr_divide_factor = 1
label_smoothing = 0.0
warmup_ratio = 0.1
log_result = False

# warmup_steps = round(len(train_dataset) / (train_batchsize * accum_grads) / 4 * epochs * 0.1)

run_name = f"fine-w2v2base-bs8-ep{epochs}-lr{default_lr}-freeze_cnn-lr_cosine-spec_aug"

if log_result:
    os.environ["WANDB_PROJECT"] = "md_d_vlsp_2023"  # name your W&B project

training_args = TrainingArguments(
    output_dir=f"wav2vec2-base-finetune-vi_phone-freeze_cnn-spec_aug-{epochs}epoch",
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
    save_total_limit=3,
    push_to_hub=False,
    torch_compile=False,
    resume_from_checkpoint=continue_train,
    report_to="wandb" if log_result else "none",
    run_name=run_name,
    lr_scheduler_type="cosine",
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

output = trainer.predict(test_dataset)

print("Output:", output)

torch.save(output, "private_test_predict.pt")

predict = greedy_decode(np.argmax(output.predictions, axis=-1))

predictions = []
for full_path, pred in zip(test_dataset.data, predict):
    path = full_path["path"]
    id_ = path.rsplit(os.sep, 1)[-1].split(".")[0]
    path = path.split("VMD-VLSP23-private-test")[-1]
    predictions.append({"id": id_, "path": path, "predict": " ".join(pred)})

df = pd.DataFrame(predictions)
df.to_csv("private_test_submission.csv", index=False)

# trainer.train(resume_from_checkpoint=continue_train)
# trainer.save_state()
# trainer.push_to_hub()
