import ctranslate2
import torch
import transformers
from dataclasses import dataclass
import huggingface_hub
from whisperlivekit.translation.mapping_languages import get_nllb_code
from timed_objects import Translation


#In diarization case, we may want to translate just one speaker, or at least start the sentences there

PUNCTUATION_MARKS = {'.', '!', '?', '。', '！', '？'}


@dataclass
class TranslationModel():
    translator: ctranslate2.Translator
    tokenizer: dict

def load_model(src_langs):
    MODEL = 'nllb-200-distilled-600M-ctranslate2'
    MODEL_GUY = 'entai2965'
    huggingface_hub.snapshot_download(MODEL_GUY + '/' + MODEL,local_dir=MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    translator = ctranslate2.Translator(MODEL,device=device)
    tokenizer = dict()
    for src_lang in src_langs:
        tokenizer[src_lang] = transformers.AutoTokenizer.from_pretrained(MODEL, src_lang=src_lang, clean_up_tokenization_spaces=True)
    return TranslationModel(
        translator=translator,
        tokenizer=tokenizer
    )

def translate(input, translation_model, tgt_lang):
    source = translation_model.tokenizer.convert_ids_to_tokens(translation_model.tokenizer.encode(input))
    target_prefix = [tgt_lang]
    results = translation_model.translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]
    return translation_model.tokenizer.decode(translation_model.tokenizer.convert_tokens_to_ids(target))

class OnlineTranslation:
    def __init__(self, translation_model: TranslationModel, input_languages: list, output_languages: list):
        self.buffer = []
        self.validated = []
        self.translation_pending_validation = ''
        self.translation_model = translation_model
        self.input_languages = input_languages
        self.output_languages = output_languages

    def compute_common_prefix(self, results):
        if not self.buffer:
            self.buffer = results
        else:
            for i in range(min(len(self.buffer), len(results))):
                if self.buffer[i] != results[i]:
                    self.commited.extend(self.buffer[:i])
                    self.buffer = results[i:]

    def translate(self, input, input_lang=None, output_lang=None):
        if not input:
            return ""
        if input_lang is None:
            input_lang = self.input_languages[0]
        if output_lang is None:
            output_lang = self.output_languages[0]
        nllb_input_lang = get_nllb_code(input_lang)
        nllb_output_lang = get_nllb_code(output_lang)
            
        source = self.translation_model.tokenizer[input_lang].convert_ids_to_tokens(self.translation_model.tokenizer[input_lang].encode(input))   
        results = self.translation_model.translator.translate_batch([source], target_prefix=[[nllb_output_lang]])
        target = results[0].hypotheses[0][1:]
        results = self.translation_model.tokenizer[input_lang].decode(self.translation_model.tokenizer[input_lang].convert_tokens_to_ids(target))
        return results

    def translate_tokens(self, tokens):
        if tokens:
            text = ' '.join([token.text for token in tokens])
            start = tokens[0].start
            end = tokens[-1].end
            translated_text = self.translate(text)
            translation = Translation(
                text=translated_text,
                start=start,
                end=end,
            )
            return translation
        return None
            
        

    def insert_tokens(self, tokens):
        self.buffer.extend(tokens)
        pass
    
    def process(self):
        i = 0
        while i < len(self.buffer):
            if self.buffer[i].text in PUNCTUATION_MARKS:
                translation_sentence = self.translate_tokens(self.buffer[:i+1])
                self.validated.append(translation_sentence)
                self.buffer = self.buffer[i+1:]
                i = 0
            else:
                i+=1
        translation_remaining = self.translate_tokens(self.buffer)
        return self.validated + [translation_remaining]
                

if __name__ == '__main__':
    output_lang = 'fr'
    input_lang = "en"
    
    shared_model = load_model([input_lang])
    online_translation = OnlineTranslation(shared_model, input_languages=[input_lang], output_languages=[output_lang])
    
    result = online_translation.translate('Hello world')
    print(result)