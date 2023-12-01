import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TranslatorModule:
    def __init__(self, device = "cuda:1", model_name="vinai/vinai-translate-vi2en-v2", cache_dir = 'cache'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN", cache_dir = cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir = cache_dir)
        self.device = device
        self.model.to(self.device)

    def translate_vi2en(self, vi_texts: str) -> str:
        input_ids = self.tokenizer(vi_texts, padding=True, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **input_ids,
            decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        en_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return en_texts

