import ctranslate2
import torch
import transformers
from dataclasses import dataclass
import huggingface_hub
from .mapping_languages import get_nllb_code

@dataclass
class TranslationModel():
    translator: ctranslate2.Translator
    tokenizer: transformers.AutoTokenizer

def load_model(src_lang):
    MODEL = 'nllb-200-distilled-600M-ctranslate2'
    MODEL_GUY = 'entai2965'
    huggingface_hub.snapshot_download(MODEL_GUY + '/' + MODEL,local_dir=MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = ctranslate2.Translator(MODEL,device=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, src_lang=src_lang, clean_up_tokenization_spaces=True)
    return TranslationModel(
        translator=translator,
        tokenizer=tokenizer
    )

def translate(input, translation_model, tgt_lang):
    if not input:
        return ""
    source = translation_model.tokenizer.convert_ids_to_tokens(translation_model.tokenizer.encode(input))
    target_prefix = [tgt_lang]
    results = translation_model.translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]
    return translation_model.tokenizer.decode(translation_model.tokenizer.convert_tokens_to_ids(target))


if __name__ == '__main__':
    tgt_lang = 'fr'
    src_lang = "en"
    nllb_tgt_lang = get_nllb_code(tgt_lang)
    nllb_src_lang = get_nllb_code(src_lang)
    translation_model = load_model(nllb_src_lang)
    result = translate('Hello world', translation_model=translation_model, tgt_lang=nllb_tgt_lang)
    print(result)