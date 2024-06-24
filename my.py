import os
import datasets

name = 'mozilla-foundation/common_voice_16_0'
save_path = os.path.join('dataset', name.replace('/', '_'))
os.makedirs(save_path, exist_ok=True)

try:
    dataset = datasets.load_dataset(name, 'zh-CN', split='train',force_download=True)
    dataset.save_to_disk(save_path)
    print("Dataset loaded and saved successfully.")
except Exception as e:
    print(f"An error occurred: {e}")

import os
from transformers import Wav2Vec2BertForCTC

name = 'lansinuote/Chinese_Speech_to_Text_CTC'
save_path = os.path.join('model', name.replace('/', '_'))
os.makedirs(save_path, exist_ok=True)

Wav2Vec2BertForCTC.from_pretrained(name).save_pretrained(save_path)


import torch
import random

from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('./processor',
                                                 bos_token='[CLS]',
                                                 eos_token='[SEP]',
                                                 unk_token='[UNK]',
                                                 pad_token='[PAD]')


feature_extractor = SeamlessM4TFeatureExtractor(sampling_rate=16000,
                                                padding_value=0.0)


processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

del tokenizer
del feature_extractor

processor


data = processor(text=['测试文字1', '测试测试文字2'],
                 audio=[torch.randn(8000).numpy(),
                        torch.randn(16000).numpy()],
                 sampling_rate=16000,
                 padding=True,
                 return_tensors='pt')


data = processor.tokenizer(['测试文字1', '测试测试文字2'],
                           padding=True,
                           truncation=True,
                           max_length=35 + 2,
                           return_tensors='pt')

data = processor.feature_extractor(
    [torch.randn(8000).numpy(),
     torch.randn(16000).numpy()],
    sampling_rate=16000,
    padding=True,
    truncation=True,
    max_length=900,
    padding_value=0.0,
    return_tensors='pt')

for k, v in data.items():
    print(k, v.shape, v.dtype, v)

from datasets import load_from_disk, Audio, load_dataset

dataset = load_from_disk('dataset/mozilla-foundation/common_voice_16_0')

dataset = dataset.remove_columns([
    'accent', 'age', 'client_id', 'down_votes', 'gender', 'locale', 'segment',
    'up_votes', 'path', 'variant'
])
dataset = dataset.rename_columns({'sentence': 'text'})
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))


def f(data):
    lens_audio = len(data['audio']['array']) / 16000
    lens_text = len(data['text'])
    return 1 <= lens_audio <= 9 and 2 <= lens_text <= 35


dataset = dataset.filter(f)

dataset, dataset[3]

def show(data):
    from IPython.display import Audio, display
    display(Audio(data=data, rate=16000))


show(dataset[3]['audio']['array'])
dataset[3]['text']

def f(data):
    text = [i['text'] for i in data]
    text = processor.tokenizer(text,
                               padding=True,
                               truncation=True,
                               max_length=35 + 2,
                               return_tensors='pt').to('cuda')

    audio = [i['audio']['array'] for i in data]
    audio = processor.feature_extractor(audio,
                                        sampling_rate=16000,
                                        padding=True,
                                        truncation=True,
                                        max_length=900,
                                        padding_value=0.0,
                                        return_tensors='pt').to('cuda')

    return text.input_ids, audio.input_features, audio.attention_mask


loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=4,
                                     collate_fn=f,
                                     drop_last=True,
                                     shuffle=True)

len(loader), next(iter(loader))

class Wav2Vec2BertForCTC(torch.nn.Module):

    def __init__(self):
        super().__init__()

        from transformers import Wav2Vec2BertModel, Wav2Vec2BertConfig
        config = Wav2Vec2BertConfig.from_pretrained(
            'model/lansinuote/Chinese_Speech_to_Text_CTC')

        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.lm_head = torch.nn.Linear(1024, processor.tokenizer.vocab_size)

        from transformers import Wav2Vec2BertForCTC
        parameters = Wav2Vec2BertForCTC.from_pretrained(
            'model/lansinuote/Chinese_Speech_to_Text_CTC')
        self.wav2vec2_bert.load_state_dict(
            parameters.wav2vec2_bert.state_dict())
        del parameters

        self.train()
        self.to('cuda')

    def forward(self, input_features, attention_mask):
        last_hidden_state = self.wav2vec2_bert(
            input_features, attention_mask=attention_mask).last_hidden_state

        last_hidden_state = self.dropout(last_hidden_state)

        return self.lm_head(last_hidden_state)


model = Wav2Vec2BertForCTC()

with torch.no_grad():
    input_features = torch.randn(4, 377, 160).to('cuda')
    attention_mask = torch.ones(4, 377).long().to('cuda')
    print(model(input_features, attention_mask).shape)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id,
                             reduction='mean',
                             zero_infinity=False)

for epoch in range(1):
    for i, (input_ids, input_features, attention_mask) in enumerate(loader):
        logits = model(input_features, attention_mask)

        log_probs = logits.log_softmax(dim=2).transpose(0, 1)
        input_lengths = (attention_mask.sum(1) / 2).ceil().long()
        input_ids_mask = input_ids != processor.tokenizer.pad_token_id

        loss = criterion(log_probs=log_probs,
                         targets=input_ids[input_ids_mask],
                         input_lengths=input_lengths,
                         target_lengths=input_ids_mask.sum(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 500 == 0:
            print(epoch, i, loss.item())
            print(processor.tokenizer.decode(input_ids[0]))
            print(processor.tokenizer.decode(logits[0].argmax(1)))