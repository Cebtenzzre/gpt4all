import sys
import struct
import json
import torch
import numpy as np
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

import gguf


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = Path(sys.argv[1])
fname_out = dir_model / "ggml-model.gguf"

with open(dir_model / "config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

with open(dir_model / "vocab.txt", "r", encoding="utf-8") as f:
    vocab = f.readlines()
# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = dir_model / ("ggml-model-" + ftype_str[ftype] + ".gguf")


ARCH = gguf.MODEL_ARCH.BERT
gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH])

print("gguf: get model metadata")

block_count = hparams["num_hidden_layers"]

gguf_writer.add_name('BERT')
gguf_writer.add_context_length(hparams['max_position_embeddings'])
gguf_writer.add_embedding_length(hparams["hidden_size"])
gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
gguf_writer.add_block_count(hparams["num_hidden_layers"])
gguf_writer.add_head_count(hparams["num_attention_heads"])
gguf_writer.add_head_count_kv(hparams["num_attention_heads"])
gguf_writer.add_file_type(ftype)

# TOKENIZATION

print("gguf: get tokenizer metadata")

tokens: list[bytearray] = []
scores: list[float] = []
toktypes: list[int] = []

tokenizer_json_file = dir_model / 'tokenizer.json'
if not tokenizer_json_file.is_file():
    print(f'Error: Missing {tokenizer_json_file}', file = sys.stderr)
    sys.exit(1)

# wordpiece tokenizer
gguf_writer.add_tokenizer_model("bert")

with open(tokenizer_json_file, "r", encoding="utf-8") as f:
    tokenizer_json = json.load(f)

print("gguf: get wordpiece tokenizer vocab")

# The number of tokens in tokenizer.json can differ from the expected vocab size.
# This causes downstream issues with mismatched tensor sizes when running the inference
vocab_size = hparams["vocab_size"]

tokenizer = AutoTokenizer.from_pretrained(dir_model)
print(tokenizer.encode('I believe the meaning of life is'))

reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}

for i in range(vocab_size):
    if i in reverse_vocab:
        text = reverse_vocab[i]
    else:
        print(f"Key {i} not in tokenizer vocabulary. Padding with an arbitrary token.")
        pad_token = f"[PAD{i}]".encode("utf8")
        text = bytearray(pad_token)

    tokens.append(text)
    scores.append(0.0)                      # dummy
    toktypes.append(gguf.TokenType.NORMAL)  # dummy

gguf_writer.add_token_list(tokens)
gguf_writer.add_token_scores(scores)
gguf_writer.add_token_types(toktypes)

special_vocab = gguf.SpecialVocab(dir_model, load_merges = True)
special_vocab.add_to_gguf(gguf_writer)

# TOKENS

tensor_map = gguf.get_tensor_name_map(ARCH, block_count)

# tensor info
print("gguf: get tensor metadata")

model = AutoModel.from_pretrained(dir_model, low_cpu_mem_usage=True)
print(model)

list_vars = model.state_dict()
for name in list_vars.keys():
    print(name, list_vars[name].shape, list_vars[name].dtype)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    if name in ['embeddings.position_ids', 'pooler.dense.weight', 'pooler.dense.bias']:
        continue
    print("Processing variable: " + name + " with shape: ", data.shape)

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    if ftype == 1 and name[-7:] == ".weight" and n_dims == 2:
        print("  Converting to float16")
        data = data.astype(np.float16)
        l_type = 1
    else:
        l_type = 0

    # map tensor names
    new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
    if new_name is None:
        print("Can not map tensor '" + name + "'")
        sys.exit()

    gguf_writer.add_tensor(new_name, data)


print("gguf: write header")
gguf_writer.write_header_to_file()
print("gguf: write metadata")
gguf_writer.write_kv_data_to_file()
print("gguf: write tensors")
gguf_writer.write_tensors_to_file()

gguf_writer.close()

print(f"gguf: model successfully exported to '{fname_out}'")
print("")
