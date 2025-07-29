import os
import torch
import argparse
from collections import Counter
from datasets import load_dataset, DatasetDict, ClassLabel, Features, Dataset, Value
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn.functional as F
import logging
import warnings
import json
import shutil
import optuna
import sys
from transformers.trainer_utils import IntervalStrategy
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    TrainerCallback
)

# Run like: 
#               source venv/bin/activate
#               python mainOptimizer.py --do_optuna --n_trials 2 | tee output.log
#               tmux new -s lora_session
#               tmux attach -t lora_session

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Setting save_embedding_layers to True as the embedding layer has been resized during finetuning."
)

# 15 
# LP Labels
LABELS = ["Execution", "Persistence", "Discovery", "Impact", "Defense Evasion", "Harmless", "Other"]

model_llama_3_70b = "meta-llama/Meta-Llama-3-70B"                       
model_llama_3_8b = "meta-llama/Meta-Llama-3-8B"

model = model_llama_3_8b   

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "disabled"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaMA-3 with QLoRA for MITRE ATT&CK tactic classification"
    )
    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--train_file", type=str, default="trainingLP.json")
    parser.add_argument("--val_file", type=str, default="validationLP.json")
    parser.add_argument("--output_dir", type=str, default="OptimizerOutputFinal")
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)            # 16
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--max_length", type=int, default=4096)  
    parser.add_argument("--lora_r", type=int, default=16)               # 16
    parser.add_argument("--lora_alpha", type=int, default=32)           # 32  
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=1)          # 2
    parser.add_argument("--do_optuna", action="store_true", default=True)  # True
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "polynomial"])
    parser.add_argument("--n_trials", type=int, default=50) # 50
    return parser.parse_args()

class WeightedTrainer(Trainer):
    """Trainer that injects class weights into the cross-entropy loss."""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        if return_outputs:
            return loss, outputs
        return loss

def clean_up_trials(output_dir, best_trial_dir):
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and path != best_trial_dir:
            print(f"Deleting {path}")
            shutil.rmtree(path)

class ResultsLoggerCallback(TrainerCallback):
    def __init__(self, filename, hyperparams):
        self.filename = filename
        self.hyperparams = hyperparams

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            lines = []
            lines.append("="*40)
            lines.append(f"Epoch {int(state.epoch) if state.epoch is not None else 'N/A'}")
            lines.append("Hyperparameters:")
            for k, v in self.hyperparams.items():
                lines.append(f"  {k}: {v}")
            lines.append("Metrics:")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")
            lines.append("")  # Blank line

            # Write to file
            with open(self.filename, "a") as f:
                f.write('\n'.join(lines) + '\n')

            # Print to console
            print('\n'.join(lines))
            sys.stdout.flush()

def main():
    global trial_hparams
    trial_results = []
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    accelerator = Accelerator()

    trial_hparams = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "grad_accum_steps": args.grad_accum_steps
    }

    # Make sure hyperparameters are not single-point!
    def optuna_hp_space(trial):
        # trial_hparams["num_epochs"] = trial.suggest_int("num_epochs", 3, 6)  
        # trial_hparams["lora_r"] = trial.suggest_categorical("lora_r", [2, 4]) 
        # trial_hparams["lora_alpha"] = trial.suggest_categorical("lora_alpha", [4, 8, 16])                  # Testing purposes
        # trial_hparams["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)
        # trial_hparams["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [1])  
        # trial_hparams["batch_size"] = trial.suggest_categorical("batch_size", [1, 2]) 

        # Change by weekend

        trial_hparams["num_epochs"] = trial.suggest_int("num_epochs", 3, 12)  
        trial_hparams["lora_r"] = trial.suggest_categorical("lora_r", [2, 4, 8, 16, 32]) 
        trial_hparams["lora_alpha"] = trial.suggest_categorical("lora_alpha", [2, 4, 8, 16, 32, 64])                  # Testing purposes
        trial_hparams["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)
        trial_hparams["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [1])  
        trial_hparams["batch_size"] = trial.suggest_categorical("batch_size", [1]) 

        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 3e-4, log=True),    
            "per_device_train_batch_size": trial_hparams["batch_size"],
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.05),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
        }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    def flatten_sessions_with_ids_and_context(filepath):
        with open(filepath, "r") as f:
            session_data = json.load(f)
        flat = []
        session_ids = []
        for i, session in enumerate(session_data):
            prev_cmds = []
            for j, cmd in enumerate(session):
                # Build context string from all previous commands in this session
                # e.g., "yum ; | wget ... | ls ;"
                context = " | ".join([c["command"] for c in prev_cmds]) if prev_cmds else ""
                flat_cmd = dict(cmd)  # make a copy
                flat_cmd["context"] = context
                flat.append(flat_cmd)
                session_ids.append(i)
                prev_cmds.append(cmd)
        return flat, session_ids, session_data

    # Flatten the session-based files
    train_data, train_session_ids, train_sessions = flatten_sessions_with_ids_and_context(args.train_file)
    val_data, val_session_ids, val_sessions = flatten_sessions_with_ids_and_context(args.val_file)

    for d, sid in zip(train_data, train_session_ids):
        d["session_id"] = sid
    for d, sid in zip(val_data, val_session_ids):
        d["session_id"] = sid

    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    original_features = ds["train"].features

    shared_features = Features({
        **original_features,
        "tactic": ClassLabel(names=LABELS),
        "session_id": Value("int64"),
        "context": Value("string"),
    })

    ds["train"] = ds["train"].cast(shared_features)
    ds["validation"] = ds["validation"].cast(shared_features)

    num_labels = ds["train"].features["tactic"].num_classes

    train_tactic_labels = ds["train"]["tactic"]
    val_tactic_labels = ds["validation"]["tactic"]
    val_session_ids_flat = ds["validation"]["session_id"]

    counts = Counter(train_tactic_labels)
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(num_labels)],
        dtype=torch.float32
    )

    def make_input(example):
        fields = []
        context = example.get("context", "")
        if context.strip():
            fields.append(f"Context: {context}")
        command = example.get("command", "")
        if command:
            fields.append(f"Command: {command}")
        return " | ".join(fields)

    def preprocess_fn(examples):
        texts = [make_input({k: examples[k][i] for k in examples if k != "session_id"}) for i in range(len(examples["command"]))]
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length
            # padding="max_length"
        )
        tokens["labels"] = examples["tactic"]
        tokens["session_id"] = examples["session_id"]
        return tokens

    ds = ds.map(preprocess_fn, batched=True, remove_columns=[col for col in ds["train"].column_names if col not in ("session_id", "context")])

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # For tracking trial numbers
    current_trial_number = [0]

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids

        predicted_labels = [LABELS[int(i)] for i in preds.tolist()[:20]]
        ground_truth_labels = [LABELS[int(i)] for i in labels.tolist()[:20]]

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        # --- SESSION-LEVEL ACCURACY ---
        # val_session_ids_flat must be a list of session ids matching the eval dataset order
        session_ids = val_session_ids_flat
        from collections import defaultdict

        sessions_pred = defaultdict(list)
        sessions_label = defaultdict(list)

        for sid, pred, label in zip(session_ids, preds.tolist(), labels.tolist()):
            sessions_pred[sid].append(pred)
            sessions_label[sid].append(label)

        total_sessions = len(sessions_pred)
        correct_sessions = 0
        for sid in sessions_pred:
            if sessions_pred[sid] == sessions_label[sid]:
                correct_sessions += 1

        session_accuracy = correct_sessions / total_sessions if total_sessions > 0 else 0

        # Track current trial number and increment 
        current_trial_number[0] += 1

        # Store results in custom line format
        result_strs = []

        result_strs.append(f"Trial {current_trial_number[0]}")
        
        # Hyperparams line
        hp_line = "Hyperparameters: " + ", ".join([
            f"{k}={v}" for k,v in trial_hparams.items()
        ])
        result_strs.append(hp_line)

        # Metrics line (now includes session_acc)
        metric_line = "Metrics: " + ", ".join([
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k,v in {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "session_acc": session_accuracy}.items()
        ])
        result_strs.append(metric_line)

        # Predicted labels line
        pred_line = "Predicted: " + ", ".join(predicted_labels)
        result_strs.append(pred_line)
        
        # Ground truth line
        gt_line = "Ground Truth: " + ", ".join(ground_truth_labels)
        result_strs.append(gt_line)

        # Blank separator line
        result_strs.append("")

        # Save as dict for sorting by f1 later
        trial_results.append({
            'trial_number': current_trial_number[0],
            'f1': f1,
            'result_lines': result_strs
        })

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "session_acc": session_accuracy
        }

    def model_init():
        print("="*40 + "\n")
        print("Starting new trial with hyperparameters: ")
        
        for k, v in trial_hparams.items():
                print(f"    {k}: {v}")

        print("="*40 + "\n")
        sys.stdout.flush()

        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir="./hf_cache"
        )
        lora_cfg = LoraConfig(
            r=trial_hparams['lora_r'],
            lora_alpha=trial_hparams['lora_alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
            lora_dropout=trial_hparams['lora_dropout'],
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(base_model, lora_cfg)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
        return model

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=trial_hparams.get("batch_size", args.batch_size),
        per_device_eval_batch_size=trial_hparams.get("batch_size", args.batch_size),
        learning_rate=trial_hparams.get("learning_rate", args.learning_rate),
        num_train_epochs=trial_hparams.get("num_epochs", args.num_epochs),
        eval_strategy="epoch",
        save_strategy="no",         #"epoch"
        logging_strategy="steps",    
        logging_steps=50,
        save_total_limit=0, #1
        load_best_model_at_end=False,   #True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=trial_hparams.get("warmup_ratio", args.warmup_ratio),
        weight_decay=trial_hparams.get("weight_decay", args.weight_decay),
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        seed=args.seed,
        report_to=None,
        logging_dir=None,   #os.path.join(args.output_dir, "logs"),
        run_name=None,  #"qlora-llama3-classification",
        gradient_accumulation_steps=trial_hparams.get("grad_accum_steps", args.grad_accum_steps),
        label_names=["labels"],
        max_grad_norm=2.0,
        ddp_find_unused_parameters=False 
    )

    results_file = os.path.join(args.output_dir, "all_trials_report.txt")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            ResultsLoggerCallback(results_file, trial_hparams)
        ]
    )

    if args.do_optuna:
        import uuid
        study_name=f"llama3_lora_optuna_{uuid.uuid4()}"

        best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=args.n_trials,
        hp_space=optuna_hp_space,
        compute_objective=lambda metrics: metrics["eval_f1"],
        backend="optuna",
        study_name=study_name,
        )

        best_trial_dir = os.path.join(args.output_dir, f"run-{best_trial.trial_id}")
        clean_up_trials(args.output_dir, best_trial_dir)

    else:
        trainer.train()
    
    # After all trials: Write report with best first!
    if len(trial_results) > 0:
        sorted_results = sorted(trial_results,key=lambda x: x['f1'],reverse=True) # best first

        with open(os.path.join(args.output_dir,"all_trials_report.txt"),"w") as f:
            for r in sorted_results:
                for line in r['result_lines']:
                    f.write(line + "\n")

if __name__ == "__main__":
    main()