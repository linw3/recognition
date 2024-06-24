from datasets import load_dataset

name = 'mozilla-foundation/common_voice_16_0'
load_dataset(name, 'zh-CN', split='train').save_to_disk('dataset/' + name)

from transformers import Wav2Vec2BertForCTC

name = 'lansinuote/Chinese_Speech_to_Text_CTC'
Wav2Vec2BertForCTC.from_pretrained(name).save_pretrained('model/' + name)