# train_job.py
import os, json, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from utils.logger import setup_logger
from utils.storage import write_output
from configs.config import DEFAULT_BATCH, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_MAX_LENGTH

logger = setup_logger("train_job")

def _make_text(example):
    instr = example.get("instruction","")
    inp = example.get("input","")
    out = example.get("output","")
    prompt = instr + ("\n\n" + inp if inp else "") + "\n\nResponse:\n"
    return {"text": prompt + out}

def run_training(job_id: str, jsonl_path: str, base_model: str, use_qlora: bool, epochs:int, batch:int, lr:float, max_length:int):
    """
    job_id: id string
    jsonl_path: local path to jsonl training file (train + valid combined -> will be split)
    base_model: huggingface id
    use_qlora: bool
    """
    start = time.time()
    out_dir = os.path.join("outputs", job_id)
    os.makedirs(out_dir, exist_ok=True)
    log_path = write_output(job_id, "train.log", "")  # create empty
    logger.info("Starting training job %s", job_id)

    # Load dataset
    ds = load_dataset("json", data_files={"train": jsonl_path})
    # split small validation
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)
    ds["train"] = ds["train"].map(_make_text, remove_columns=ds["train"].column_names)
    ds["test"] = ds["test"].map(_make_text, remove_columns=ds["test"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Quantization config if qlora
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    # LoRA config (safe defaults)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    # Train
    trainer.train()
    # Save adapter/peft
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save metadata
    meta = {
        "job_id": job_id,
        "base_model": base_model,
        "use_qlora": use_qlora,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "duration_seconds": int(time.time()-start)
    }
    write_output(job_id, "meta.json", json.dumps(meta, indent=2))
    logger.info("Training finished for job %s, outputs at %s", job_id, out_dir)
    return out_dir