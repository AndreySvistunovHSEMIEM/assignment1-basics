import os
import json
import regex as re
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

from .tokenizer_utils import get_freqs_of_words_and_pairs, find_chunk_boundaries, gpt2_bytes_to_unicode, split_special_tokens, PAT


class Tokenizer:
    def __init__(
            self, 
            input_path: str | None = None,
            vocab: dict[int, bytes] | None = None,
            merges: dict[tuple[bytes, bytes]] | None = None,
            special_tokens: list[str] | None = None,
    ) -> None:
        
        self.special_tokens = special_tokens

        if vocab and merges:
            if self.special_tokens:
                last_id = max(vocab) + 1
                existing_values = set(vocab.values())
                for token in self.special_tokens:
                    token = token.encode("utf-8")
                    if token not in existing_values:
                        vocab[last_id] = token
                        last_id += 1

            self.vocabulary = vocab
            self.inverted_vocabulary = {v: k for k, v in self.vocabulary.items()}
            self.merges = merges

        elif input_path:
            assert os.path.exists(input_path), f"File {input_path} doesn't exists!"
            self.input_path = input_path
            special_token = special_tokens[0].encode("utf-8") if self.special_tokens else None
            self.word_freqs, self.pairs_freqs = self._preprocess_file(self.input_path, special_token)
            self.vocabulary, self.last_id = self._prepare_vocabulary()
            self.merges: list[bytes] = []
        
        else:
            raise AttributeError("__init__ gets `input_path` or `vocab` and `merges`!")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
                                                                                                            
        with open(vocab_filepath) as f:                                                                     
            gpt2_vocab = json.load(f)   

        vocab = {                                                                                           
            idx: bytes([gpt2_byte_decoder[ch] for ch in token_str])
            for token_str, idx in gpt2_vocab.items()                                                        
        }                                                                                                   
                                                                                                            
        merges = []                                                                                         
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip()
                if not line or len(line.split(" ")) != 2:                                                   
                    continue
                a, b = line.split(" ")                                                                      
                merges.append((
                    bytes([gpt2_byte_decoder[ch] for ch in a]),                                             
                    bytes([gpt2_byte_decoder[ch] for ch in b]),
                ))                                                                                                
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens) 

    def encode(self, text: str) -> list[int]:
        text_int_ids = []
        splitted_text = split_special_tokens(text, self.special_tokens)
        for sentence in splitted_text:
            if self.special_tokens and sentence in set(self.special_tokens):
                text_int_ids.append(self.inverted_vocabulary[sentence.encode("utf-8")])
                continue
            for match_pattern in re.finditer(PAT, sentence):
                word = [bytes([element]) for element in match_pattern.group().encode("utf-8")]
                new_word = []
                for merge in self.merges:
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and merge == (word[i], word[i + 1]):
                            new_word.append(b"".join(merge))
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                    new_word = []
                for element in word:
                    text_int_ids.append(self.inverted_vocabulary[element])
        return text_int_ids
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = []
        for vocab_id in ids:
            text.append(self.vocabulary[vocab_id])
        return (b"".join(text)).decode("utf-8", "ignore")
    
    def train_bpe(self, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert vocab_size > (256 + len(self.special_tokens)), \
            f"Given vocab size ({vocab_size}) must be greater than {(256 + len(self.special_tokens))}!"

        while len(self.vocabulary) < vocab_size:
            most_frequent_pair = max(self.pairs_freqs.items(), key=lambda kv: (kv[1], kv[0]))[0]
            del self.pairs_freqs[most_frequent_pair]
            self.merges.append(most_frequent_pair)
            self.vocabulary[self.last_id] = b"".join(most_frequent_pair)
            self.last_id += 1

            self._update_words_and_pairs(most_frequent_pair)
        
        return self.vocabulary, self.merges

    def _update_words_and_pairs(self, most_frequent_pair: tuple[bytes]) -> None:
        new_word_freqs = Counter()
        for word, frequency in self.word_freqs.items():
            new_word = []
            i = 0
            
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_frequent_pair:
                    new_word.append(b"".join(most_frequent_pair))
                    if i > 0:
                        self.pairs_freqs[(new_word[-2], word[i])] -= frequency
                        self.pairs_freqs[(new_word[-2], new_word[-1])] += frequency
                    if i + 1 < len(word) - 1:
                        self.pairs_freqs[(word[i + 1], word[i + 2])] -= frequency
                        self.pairs_freqs[(new_word[-1], word[i + 2])] += frequency
                    i += 1
                else:
                    new_word.append(word[i])
                i += 1
            if len(word) > 1 and (word[-2], word[-1]) != most_frequent_pair:
                new_word.append(word[-1])
            new_word_freqs[tuple(new_word)] = frequency
        
        self.word_freqs = new_word_freqs

    def _preprocess_file(
            self, 
            input_path: str, 
            special_tokens: bytes,
    ) -> tuple[Counter[bytes], Counter[tuple[bytes]]]:
        
        num_workers = os.process_cpu_count()
        chunk_size = num_workers * 8
        with open(input_path, "rb") as file:
            boundaries = find_chunk_boundaries(file, chunk_size, special_tokens)

        num_workers = os.process_cpu_count()
        with ProcessPoolExecutor(num_workers) as executor:
            list_of_freqs_words_and_pairs = list(
                executor.map(
                    get_freqs_of_words_and_pairs,
                    [input_path] * (len(boundaries) - 1),
                    boundaries[:-1],
                    boundaries[1:],
                    [special_tokens] * (len(boundaries) - 1),
                )
            )
        words_freqs, pairs_freqs = Counter(), Counter()
        for i, j in list_of_freqs_words_and_pairs:
            words_freqs.update(i)
            pairs_freqs.update(j)


        return words_freqs, pairs_freqs
    
    def _prepare_vocabulary(self) -> tuple[dict[int, bytes], int]:
        vocabulary = {}
        for ind, token in enumerate(self.special_tokens):
            vocabulary[ind] = token.encode("utf-8")

        for byte in range(256):
            vocabulary[byte + len(self.special_tokens)] = bytes([byte])
        
        return vocabulary, 256 + len(self.special_tokens)
