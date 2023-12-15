import torch
from torch import nn
from torchaudio import models
import numpy as np


def init_sub_block(
    input_dim,
    model_dim,
    kernel_size=3,
    stride=2,
):
    return nn.Sequential(
        nn.Conv1d(input_dim, model_dim, kernel_size, stride, padding=1),
        nn.ReLU(),
    )


def calc_length(
    lengths, padding=1, kernel_size=3, stride=2, ceil_mode=False, repeat_num=1
):
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class Conformer(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=4,
        model_kernel_size=31,
        dropout=0.1,
        vocab_size=123,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        self.model = models.Conformer(
            model_dim, num_heads, model_dim * 4, num_layers, model_kernel_size, dropout
        )
        self.out = nn.Linear(model_dim, vocab_size)

    def forward(self, x, x_len):
        x = self.down(x)
        x_len = calc_length(x_len, repeat_num=2)
        x, x_len = self.model(x.permute(0, 2, 1), x_len)
        o = self.out(x)
        o = nn.functional.log_softmax(o, dim=-1)
        return o, x_len


class ConformerPitch(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=4,
        model_kernel_size=31,
        dropout=0.1,
        vocab_size=123,
    ):
        super().__init__()
        # total_divide = 4
        # assert model_dim % total_divide == 0, "Must be divisible"
        # model_dim_each = model_dim // total_divide
        half_dim = model_dim // 2

        self.down_mel = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, half_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        self.down_pitch = nn.Sequential(
            nn.Conv1d(1, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, half_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        self.model = models.Conformer(
            model_dim, num_heads, model_dim * 4, num_layers, model_kernel_size, dropout
        )
        self.out = nn.Linear(model_dim, vocab_size)

    def forward(self, x, x_len, pitch):
        x = self.down_mel(x)
        px = self.down_pitch(pitch)
        x_len = calc_length(x_len, repeat_num=2)
        x = x.permute(0, 2, 1)
        px = px.permute(0, 2, 1)
        x = torch.cat([x, px], dim=-1)
        # x = self.input_proj(xx)
        x, x_len = self.model(x, x_len)
        o = self.out(x)
        o = nn.functional.log_softmax(o, dim=-1)
        return o, x_len


class ConformerPitchTonal(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=4,
        model_kernel_size=31,
        dropout=0.1,
        vocab_size=123,
        num_tonals=6,
    ):
        super().__init__()

        # total_divide = 4
        # assert model_dim % total_divide == 0, "Must be divisible"
        # model_dim_each = model_dim // total_divide

        half_dim = model_dim // 2

        self.down_mel = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, half_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        self.down_pitch = nn.Sequential(
            nn.Conv1d(1, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, half_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        num_layers_first = 1
        num_layers_second = num_layers - 1

        self.out_tonal = nn.Linear(model_dim, num_tonals)
        self.tonal_proj = nn.Sequential(
            nn.Linear(model_dim + num_tonals, model_dim),
            nn.ReLU(),
        )

        self.model_first = models.Conformer(
            model_dim,
            num_heads,
            model_dim * 4,
            num_layers_first,
            model_kernel_size,
            dropout,
        )

        self.model_second = models.Conformer(
            model_dim,
            num_heads,
            model_dim * 4,
            num_layers_second,
            model_kernel_size,
            dropout,
        )

        self.out = nn.Sequential(
            nn.Linear(model_dim, vocab_size), nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_len, pitch, return_tonals=False):
        x = self.down_mel(x)
        px = self.down_pitch(pitch)
        x_len = calc_length(x_len, repeat_num=2)
        x = x.permute(0, 2, 1)
        px = px.permute(0, 2, 1)
        x = torch.cat([x, px], dim=-1)
        x, x_len = self.model_first(x, x_len)

        tonal_logit = self.out_tonal(x)
        tonal_prob = nn.functional.softmax(tonal_logit, dim=-1)

        x = torch.cat([x, tonal_prob], dim=-1)
        x = self.tonal_proj(x)
        x, x_len = self.model_second(x, x_len)

        o = self.out(x)

        if return_tonals:
            return o, x_len, nn.functional.log_softmax(tonal_logit, dim=-1)

        return o, x_len


class EffConformer(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=[2, 2, 4],
        model_kernel_size=15,
        dropout=0.1,
        vocab_size=123,
    ):
        super().__init__()
        down = []
        model = []
        self.num_down = len(num_layers)
        for i in range(self.num_down):
            down.append(init_sub_block(input_dim, model_dim))
            model.append(
                models.Conformer(
                    model_dim,
                    num_heads,
                    model_dim * 4,
                    num_layers[i],
                    model_kernel_size,
                    dropout,
                )
            )
            input_dim = model_dim
        self.down = nn.ModuleList(down)
        self.model = nn.ModuleList(model)
        self.out = nn.Linear(model_dim, vocab_size)

    def forward(self, x, x_len):
        for i in range(self.num_down):
            x = self.down[i](x)
            x = x.permute(0, 2, 1)
            x, x_len = self.model[i](x, calc_length(x_len))
            if i < self.num_down - 1:
                x = x.permute(0, 2, 1)
        o = self.out(x)
        o = nn.functional.log_softmax(o, dim=-1)
        return o, x_len


class ConvLSTMNet(nn.Module):
    def __init__(
        self, input_dim, model_dim=64, hidden_dim=128, num_layers=2, vocab_size=123
    ):
        super().__init__()
        self.cnn1 = init_sub_block(input_dim, model_dim)
        self.cnn2 = init_sub_block(model_dim, model_dim * 2)
        self.rnn = nn.LSTM(
            model_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, x_len):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x_len = calc_length(x_len, repeat_num=2)
        x = x.permute(0, 2, 1)
        o, (h, c) = self.rnn(x)
        o = self.out(o)
        o = nn.functional.log_softmax(o, dim=-1)
        return o, x_len


class PhoneCNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc = nn.Linear(1024,768)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        # x = self.fc(x)
        x = self.Conv2d(x)
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)

        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        return x


class PhoneRNNStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, bidirectional=True)

    def forward(self, x):
        x = self.bilstm(x)
        x = self.bn(x[0].squeeze(0))
        x = self.drop_out(x)

        return x.unsqueeze(0)


class Phonetic_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = PhoneCNNStack()
        self.RNN = PhoneRNNStack()

    def forward(self, x):
        x = self.CNN(x)
        x = self.RNN(x)
        return x


class AcousticCNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        # print(x.shape)
        x = self.Conv2d(x)
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class AcousticCNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class AcousticRNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=81, hidden_size=384, bidirectional=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class AcousticRNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, bidirectional=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0).squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class Acoustic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN1 = AcousticCNNStack1()
        self.CNN = AcousticCNNStack()
        self.RNN = AcousticRNNStack()
        self.RNN1 = AcousticRNNStack1()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN(x)
        x = self.RNN1(x)
        x = self.RNN(x)
        return x


class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(256, 64)
        self.fc1 = nn.Linear(128, 2304)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True)
        self.fc3 = nn.Linear(128, 2304)

    def forward(self, x):
        x = torch.t(x)  # (len(canonical) x 1)
        x = torch.tensor(x, dtype=torch.int)
        x = self.embedding(x).squeeze(1)
        o, (h_n, c_n) = self.bilstm(x)
        y = self.fc1(o)
        x = self.fc3(o)
        return x, y


class PitchCNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(1, 81)
        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        # print(x.shape)
        x = self.fc(x)
        x = self.Conv2d(x)
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class PitchCNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.Conv2d(x)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)

        return x


class PitchRNNStack1(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=81, hidden_size=384, bidirectional=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class PitchRNNStack(nn.Module):
    def __init__(self):
        super().__init__()

        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, bidirectional=True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0).squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class Pitch_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN1 = PitchCNNStack1()
        self.CNN = PitchCNNStack()
        self.RNN = PitchRNNStack()
        self.RNN1 = PitchRNNStack1()

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN(x)
        x = self.RNN1(x)
        x = self.RNN(x)
        return x


class PitchAcousticPhoneticLinguistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.Acoustic_encoder = Acoustic_encoder()
        self.Phonetic_encoder = Phonetic_encoder()
        self.Linguistic_encoder = Linguistic_encoder()
        self.Pitch_encoder = Pitch_encoder()
        # self.text_to_tensor = text_to_tensor
        # self.tensor_to_text = tensor_to_text
        self.fc1 = nn.Linear(4608, 159, bias=True)
        self.multihead_attn = nn.MultiheadAttention(2304, 16, batch_first=True)

    def forward(self, acoustic, phonetic, linguistic, pitch):
        phonetic = self.Phonetic_encoder(phonetic)  # batch x time x 768
        acoustic = self.Acoustic_encoder(acoustic)  # batch x time x 768
        pitch = self.Pitch_encoder(pitch)
        linguistic = self.Linguistic_encoder(linguistic)  # shape [0]: 2304 x len(canon)
        Hv = linguistic[0]
        Hk = linguistic[1]
        acoustic = acoustic.squeeze(0).squeeze(0)
        acoustic = torch.t(acoustic)
        acoustic = acoustic.unsqueeze(0)
        pitch = pitch.squeeze(0).squeeze(0)
        pitch = torch.t(pitch)
        pitch = pitch.unsqueeze(0)
        Hq = torch.cat((acoustic, phonetic, pitch), 2)
        Hq = Hq
        Hk = Hk.unsqueeze(0)
        Hv = Hv.unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(Hq, Hk, Hv)
        c = attn_output
        before_Linear = torch.cat((c, Hq), 2)
        output = self.fc1(before_Linear)
        return output.squeeze(0)


class ConformerEncoderModule(nn.Module):
    def __init__(
        self,
        input_dim=80,
        model_dim=144,
        num_heads=4,
        num_layers=4,
        model_kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, 3, 2, padding=1),
            nn.ReLU(),
        )
        self.model = models.Conformer(
            model_dim, num_heads, model_dim * 4, num_layers, model_kernel_size, dropout
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_len):
        x = self.down(x)
        x_len = calc_length(x_len, repeat_num=2)
        x, x_len = self.model(x.permute(0, 2, 1), x_len)
        return x, x_len


class TransformerDecoderModule(nn.Module):
    def __init__(
        self,
        model_dim=144,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        layer_norm_eps=1e-5,
        bias=True,
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            model_dim,
            num_heads,
            model_dim * 4,
            dropout,
            layer_norm_eps,
            batch_first=True,
            bias=bias,
        )
        decoder_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps, bias=bias)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=decoder_norm
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, y, memory, y_mask=None, memory_mask=None):
        return self.decoder(y, memory, y_mask, memory_mask)
