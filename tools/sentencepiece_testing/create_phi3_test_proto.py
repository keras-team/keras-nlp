import pathlib

import sentencepiece.sentencepiece_model_pb2 as sp_pb2

from tools.sentencepiece_testing.utils import train_sentencepiece

ADDED_TOKENS = [
    "<|endoftext|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
    "<|user|>",
]


def add_added_tokens(filename):
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_hub"
        / "src"
        / "tests"
        / "test_data"
        / filename,
        mode="rb",
    ) as sp_model_file:
        model_proto = sp_pb2.ModelProto()
        model_proto.ParseFromString(sp_model_file.read())
    for token in ADDED_TOKENS:
        new_token = sp_pb2.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0.0
        new_token.type = 4  # user defined symbols.
        model_proto.pieces.append(new_token)
    with open(
        pathlib.Path(__file__).parent.parent.parent
        / "keras_hub"
        / "src"
        / "tests"
        / "test_data"
        / filename,
        mode="wb",
    ) as f:
        f.write(model_proto.SerializeToString())


def main():
    train_sentencepiece(
        ["the fox on the table", "the fox on the earth"],
        "phi3_test_vocab.spm",
        vocab_size=15,
        model_type="bpe",  # BPE
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    add_added_tokens("phi3_test_vocab.spm")


if __name__ == "__main__":
    main()
