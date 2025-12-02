from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, PeftType, prepare_model_for_kbit_training
import torch

def openai_whisper_small():
    # 0.2B parameters
    model_name = "openai/whisper-small"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return processor, model

def openai_whisper_small_custom_tokenizer():
    model_name = "openai/whisper-small"

    tokenizer = WhisperTokenizerFast.from_pretrained("taiwanese_tokenizer")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    
    target_vocab_size = len(tokenizer)
    print(f"Resizing model embeddings from {model.config.vocab_size} to {target_vocab_size}")
    model.resize_token_embeddings(target_vocab_size)
    
    actual_embed_size = model.model.decoder.embed_tokens.num_embeddings
    actual_proj_out_size = model.proj_out.out_features

    model.config.use_cache = False
    model.config.vocab_size = target_vocab_size
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    
    model.generation_config.vocab_size = target_vocab_size
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.task = None
    model.generation_config.language = None
    model.generation_config.is_multilingual = False
    model.config.is_multilingual = False

    print(f"Embedding layer: {actual_embed_size}")
    print(f"Projection out: {actual_proj_out_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Generation vocab size: {model.generation_config.vocab_size}")
    
    return processor, model
    
def openai_whisper_medium():
    # 0.8B parameters
    model_name = "openai/whisper-medium"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        task="transcribe"
    )
    return processor, model

def openai_whisper_large_v3():
    # 2B parameters
    model_name = "openai/whisper-large-v3"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return processor, model

def openai_whisper_large_v3_turbo():
    # 0.8B parameters
    model_name = "openai/whisper-large-v3-turbo"

    processor = AutoProcessor.from_pretrained(
        model_name,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return processor, model

def custom(model_dir):
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        task="transcribe"
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        local_files_only=True
    )
    model.config.use_cache = False
    model.generation_config.task = "transcribe"
    model.generation_config.language = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    return processor, model

# fix TypeError: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'
# https://discuss.huggingface.co/t/unexpected-keywork-argument/91356/3
# https://github.com/huggingface/peft/issues/1988
class WhisperTuner(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
            self,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_ids=None,
            **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(**kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs

def apply_lora(processor, model, r=8, alpha=16, dropout=0.05):

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=[
            "k_proj",
            "q_proj",
            "v_proj",
            "out_proj",
        ]
    )

    model = WhisperTuner(model, lora_config)
    return processor, model

def openai_whisper_large_v3_turbo_lora(lora_r=8, lora_alpha=16, lora_dropout=0.05):
    processor, model = openai_whisper_large_v3_turbo()
    return apply_lora(processor, model, lora_r, lora_alpha, lora_dropout)

def openai_whisper_small_lora(lora_r=8, lora_alpha=16, lora_dropout=0.05):
    processor, model = openai_whisper_small()
    return apply_lora(processor, model, lora_r, lora_alpha, lora_dropout)

def openai_whisper_large_v3_turbo_4bit(lora_r=8, lora_alpha=16, lora_dropout=0.05):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        'openai/whisper-large-v3-turbo',
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        use_cache=False
    )
    
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    processor = AutoProcessor.from_pretrained(
        'openai/whisper-large-v3-turbo',
        task="transcribe"
    )
    return apply_lora(processor, base_model, lora_r, lora_alpha, lora_dropout)
