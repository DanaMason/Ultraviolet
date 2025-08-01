import os
import torch
import argparse
from collections import Counter, defaultdict
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
import wandb
from rouge_score import rouge_scorer
from seqeval.metrics import classification_report, f1_score as seq_f1_score
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    TrainerCallback
)



# Run using commands like: 
#               source venv/bin/activate
#               python mainOptimizerLPToken.py --do_optuna --n_trials 20 | tee output.log
#               tmux new -s lora_session
#               tmux attach -t lora_session



# Warnings and Error Supression

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Setting save_embedding_layers to True as the embedding layer has been resized during finetuning.")



# LogPrecis Labels
LABELS = ["Execution", "Persistence", "Discovery", "Impact", "Defense Evasion", "Harmless", "Other"]

# Model Options

model_llama_3_70b = "meta-llama/Meta-Llama-3-70B"                       
model_llama_3_8b = "meta-llama/Meta-Llama-3-8B"

model = model_llama_3_8b   

# For Memory usage and setting the output to WandB (needs account)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "online"

# Takes in command-line level inputs (so you can send in a new hyperparameter X 
# when running the program from command line (defaults are hardcoded)).

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3 with QLoRA for MITRE ATT&CK tactic classification")

    parser.add_argument("--model_name", type=str, default=model)
    parser.add_argument("--train_file", type=str, default="trainingLP.json")
    parser.add_argument("--val_file", type=str, default="validationLP.json")
    parser.add_argument("--output_dir", type=str, default="OptimizerOutputFinal")
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)       
    parser.add_argument("--learning_rate", type=float, default=4e-05)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--max_length", type=int, default=4096)  
    parser.add_argument("--lora_r", type=int, default=32)            
    parser.add_argument("--lora_alpha", type=int, default=32)           
    parser.add_argument("--lora_dropout", type=float, default=0.007979638733989103) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.002)
    parser.add_argument("--early_stopping_patience", type=int, default=8)  
    parser.add_argument("--grad_accum_steps", type=int, default=1)         
    parser.add_argument("--do_optuna", action="store_true", default=False) 
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "polynomial"])
    parser.add_argument("--n_trials", type=int, default=1) 

    return parser.parse_args()

# Weighted Trainer subclass of HF's Trainer.
# Uses class weights for Cross-Entropy while training 
# for class imbalance.

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),   
            labels.view(-1),                      
            weight=self.class_weights,
            ignore_index=-100
        )
        if return_outputs:
            return loss, outputs

        return loss

# Cleanup function (mostly can be ignored)

def clean_up_trials(output_dir, best_trial_dir):
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)

        if os.path.isdir(path) and path != best_trial_dir:
            print(f"Deleting {path}")
            shutil.rmtree(path)

# Custom Callback with a special evaluate function to report various 
# pieces of information to the operator.

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

            lines.append("")  

            with open(self.filename, "a") as f:
                f.write('\n'.join(lines) + '\n')

            print('\n'.join(lines))
            sys.stdout.flush()

# Main

def main():

    # Setup / Initialization of various things including
    # vars, seed, accelerator, trial hyperparameters, etc.

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

    # Defines hyperparameter search space to determine optimal hyperparameters.
    # (This is where your range for hyperparameters goes).

    #       [X, Y] = X or Y.              X, Y = X through Y.
    
    def optuna_hp_space(trial):
        trial_hparams["num_epochs"] = trial.suggest_int("num_epochs", 12, 12)  
        trial_hparams["lora_r"] = trial.suggest_categorical("lora_r", [32]) 
        trial_hparams["lora_alpha"] = trial.suggest_categorical("lora_alpha", [32])                  # Testing purposes
        trial_hparams["lora_dropout"] = trial.suggest_float("lora_dropout", 0.007979638733989103, 0.007979638733989103)
        trial_hparams["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [1])  
        trial_hparams["batch_size"] = trial.suggest_categorical("batch_size", [1]) 

        # Uncomment following:

        # trial_hparams["num_epochs"] = trial.suggest_int("num_epochs", 3, 12)  
        # trial_hparams["lora_r"] = trial.suggest_categorical("lora_r", [2, 4, 8, 16, 32]) 
        # trial_hparams["lora_alpha"] = trial.suggest_categorical("lora_alpha", [2, 4, 8, 16, 32, 64])                  # Testing purposes
        # trial_hparams["lora_dropout"] = trial.suggest_float("lora_dropout", 0.0, 0.1)
        # trial_hparams["grad_accum_steps"] = trial.suggest_categorical("grad_accum_steps", [1])  
        # trial_hparams["batch_size"] = trial.suggest_categorical("batch_size", [1]) 

        # Uncomment":

        return {
            "learning_rate": trial.suggest_float("learning_rate", 4e-05, 4e-05, log=True),    
            "per_device_train_batch_size": trial_hparams["batch_size"],
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.02, 0.02),
            "weight_decay": trial.suggest_float("weight_decay", 0.002, 0.002),

            # Uncomment:

            # "learning_rate": trial.suggest_float("learning_rate", 1e-6, 3e-4, log=True),    
            # "per_device_train_batch_size": trial_hparams["batch_size"],
            # "warmup_ratio": trial.suggest_float("warmup_ratio", 0.01, 0.05),
            # "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
        }

    # Tokenizer setup and padding.

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Creates dataset to be properly formatted.

    def build_token_dataset(inputs, labels, session_ids):
        return Dataset.from_dict({
            "input_ids": inputs,
            "labels": labels,
            "session_id": session_ids
        })

    # Flattens, tokenizes, and chunks datasets.

    def flatten_and_tokenize_sessions(filepath, tokenizer, max_length=4096, stride=128):
        
        with open(filepath, "r") as f:
            session_data = json.load(f)

        chunked_inputs = []
        chunked_labels = []
        chunked_session_ids = []
        chunked_offsets = []  
        
        for session_id, session in enumerate(session_data):
            tokens = []
            labels = []

            # Tokenization

            for cmd in session:
                cmd_text = cmd["command"]
                tactic = cmd["tactic"]

                cmd_tokens = tokenizer.tokenize(cmd_text)
                tokens.extend(cmd_tokens)
                label_id = LABELS.index(tactic)
                labels.extend([label_id] * len(cmd_tokens))
           
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Chunking

            for i in range(0, len(input_ids), stride):
                
                chunk_input_ids = input_ids[i:i+max_length]
                chunk_labels = labels[i:i+max_length]

                pad_len = max_length - len(chunk_input_ids)
                
                if pad_len > 0:
                    chunk_input_ids += [tokenizer.pad_token_id] * pad_len
                    chunk_labels += [-100] * pad_len  
                
                chunked_inputs.append(chunk_input_ids)
                chunked_labels.append(chunk_labels)
                chunked_session_ids.append(session_id)
                chunked_offsets.append((i, min(i+max_length, len(input_ids))))

        return chunked_inputs, chunked_labels, chunked_session_ids, chunked_offsets

    # Flattens/Tokenizes/Chunks the actual datasets being used.

    train_inputs, train_labels, train_session_ids, _ = flatten_and_tokenize_sessions(args.train_file, tokenizer=tokenizer, max_length=args.max_length, stride=args.max_length//2)
    val_inputs, val_labels, val_session_ids, _ = flatten_and_tokenize_sessions(args.val_file, tokenizer=tokenizer, max_length=args.max_length, stride=args.max_length//2)

    # Setup for datasets

    ds_train = build_token_dataset(train_inputs, train_labels, train_session_ids)
    ds_val   = build_token_dataset(val_inputs, val_labels, val_session_ids)
    
    ds = DatasetDict({"train": ds_train, "validation": ds_val})

    # Computes class weights.

    num_labels = len(LABELS)

    flat_train_labels = [l for sub in train_labels for l in sub if l != -100]
    counts = Counter(flat_train_labels)
    total = sum(counts.values())

    class_weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(num_labels)],
        dtype=torch.float32
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Q in QLoRA: this is the 4-bit quantization.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    current_trial_number = [0]

    # Calculates the various metrics: accuracy, precision, recall,
    # f1, metrics per category, ROUGE-1, Binary Fidelity (session accuracy).

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids    
                          
        if hasattr(p, "inputs") and p.inputs is not None and "session_id" in p.inputs:
            session_ids = p.inputs["session_id"]
        else:
            session_ids = [0] * len(preds)

        pred_label_list = []
        true_label_list = []
        pred_flat = []
        true_flat = []
        session_pred_dict = defaultdict(list)
        session_true_dict = defaultdict(list)

        predicted_labels = [LABELS[int(i)] for i in preds.tolist()[:20]]
        ground_truth_labels = [LABELS[int(i)] for i in labels.tolist()[:20]]

        # Processes each token in every chunk to analyze

        for pred_seq, label_seq, sid in zip(preds, labels, session_ids):
            temp_pred = []
            temp_true = []

            for p_i, l_i in zip(pred_seq, label_seq):

                if l_i == -100:  
                    continue

                temp_pred.append(LABELS[p_i])
                temp_true.append(LABELS[l_i])

                pred_flat.append(p_i)
                true_flat.append(l_i)

            pred_label_list.append(temp_pred)
            true_label_list.append(temp_true)

            session_pred_dict[sid].extend(temp_pred)
            session_true_dict[sid].extend(temp_true)

        # Calculate both averaged and per tactic:

        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average="micro", zero_division=0
        )

        acc = accuracy_score(true_flat, pred_flat)

        precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average=None, 
            labels=list(range(len(LABELS))), zero_division=0
        )

        per_tactic_metrics = {}
        for idx, label_name in enumerate(LABELS):
            per_tactic_metrics[f"precision_{label_name}"] = precision_per[idx]
            per_tactic_metrics[f"recall_{label_name}"] = recall_per[idx]
            per_tactic_metrics[f"f1_{label_name}"] = f1_per[idx]

        # Determines session accuracy (Binary Fidelity).

        total_sessions = len(session_pred_dict)
        correct_sessions = sum(
            session_pred_dict[sid] == session_true_dict[sid]
            for sid in session_pred_dict
        )

        session_accuracy = correct_sessions / total_sessions if total_sessions > 0 else 0

        # ROUGE-1 calculations

        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
        rouge_1_scores = []

        for sid in session_pred_dict:
            pred_seq = ' '.join(session_pred_dict[sid])
            true_seq = ' '.join(session_true_dict[sid])
            score = scorer.score(true_seq, pred_seq)['rouge1'].fmeasure
            rouge_1_scores.append(score)
        
        rouge_1_mean = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0

        seqeval_f1 = seq_f1_score(true_label_list, pred_label_list)

        # Logging and reporting results

        result_strs = []

        result_strs.append(f"Trial {current_trial_number[0]}")
        hp_line = "Hyperparameters: " + ", ".join([
            f"{k}={v}" for k,v in trial_hparams.items()
        ])

        result_strs.append(hp_line)

        result_strs.append("Per-tactic F1:")
        for idx, label_name in enumerate(LABELS):
            result_strs.append(f"  {label_name}: {f1_per[idx]:.4f}")

        metric_line = "Metrics: " + ", ".join([
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k,v in {
                "accuracy": acc,
                "precision": precision_micro,
                "recall": recall_micro,
                "f1": f1_micro,
                "session_acc": session_accuracy,
                "rouge1": rouge_1_mean,
            }.items()
        ])

        result_strs.append(metric_line)

        pred_line = "Predicted: " + ", ".join(predicted_labels)
        result_strs.append(pred_line)
        
        gt_line = "Ground Truth: " + ", ".join(ground_truth_labels)
        result_strs.append(gt_line)
        result_strs.append("")

        trial_results.append({
            'trial_number': current_trial_number[0],
            'f1': f1_micro,
            'result_lines': result_strs
        })

        metrics = {
            "accuracy": acc,
            "precision": precision_micro,
            "recall": recall_micro,
            "f1": f1_micro,
            "session_acc": session_accuracy,
            "rouge1": rouge_1_mean,
            "seqeval_f1": seqeval_f1,
        }

        if wandb.run is not None:
            wandb.log(metrics)

        metrics.update(per_tactic_metrics)
        return metrics

    # Initialize the model.
    # Includes a base model utilizing token classification.
    # Also has the LoRA setup information.
    # Finally, sets the model with all this information.

    def model_init():
        print("="*40 + "\n")
        print("Starting new trial with hyperparameters: ")
        
        for k, v in trial_hparams.items():
                print(f"    {k}: {v}")

        print("="*40 + "\n")
        sys.stdout.flush()

        base_model = AutoModelForTokenClassification.from_pretrained(
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

    # Training Arguments sent to the Trainer for training.

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=trial_hparams.get("batch_size", args.batch_size),
        per_device_eval_batch_size=trial_hparams.get("batch_size", args.batch_size),
        learning_rate=trial_hparams.get("learning_rate", args.learning_rate),
        num_train_epochs=trial_hparams.get("num_epochs", args.num_epochs),
        eval_strategy="epoch",
        save_strategy="epoch",         
        logging_strategy="steps",    
        logging_steps=50,
        save_total_limit=1, 
        load_best_model_at_end=True,   
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=trial_hparams.get("warmup_ratio", args.warmup_ratio),
        weight_decay=trial_hparams.get("weight_decay", args.weight_decay),
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        seed=args.seed,
        report_to="wandb",
        logging_dir=os.path.join(args.output_dir, "logs"),
        run_name="qlora-llama3-classification",
        gradient_accumulation_steps=trial_hparams.get("grad_accum_steps", args.grad_accum_steps),
        label_names=["labels"],
        max_grad_norm=2.0,
        ddp_find_unused_parameters=False
    )

    results_file = os.path.join(args.output_dir, "all_trials_report.txt")

    # Train the model via the WeightedTrainer subclass, class weights, early stopping callback and a callback for reporting.

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

    # Logic for running optuna.
    # Sets it to run for n trials, using the provided search space, finding most optimal evaluation f1 score.
    # Cleans up the directory output as well.
    # If no optuna, just runs with default args passed.

    if args.do_optuna:
        import uuid
        study_name=f"llama3_lora_optuna_{uuid.uuid4()}"

        best_trial = trainer.hyperparameter_search(
        direction="maximize",
        n_trials=args.n_trials,
        hp_space=optuna_hp_space,
        compute_objective=lambda metrics: metrics["f1"],
        backend="optuna",
        study_name=study_name,
        )

        best_trial_dir = os.path.join(args.output_dir, f"run-{best_trial.trial_id}")
        clean_up_trials(args.output_dir, best_trial_dir)


    else:
        trainer.train()

    # Saves best model.

    best_model_path = trainer.state.best_model_checkpoint

    if best_model_path is not None:
        target_path = os.path.join(args.output_dir, "best_model")

        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        shutil.copytree(best_model_path, target_path)
        print(f"Best model saved to: {target_path}")
    
    # Finally, report the best metrics and hyperparameters in a txt folder.

    if len(trial_results) > 0:
        sorted_results = sorted(trial_results,key=lambda x: x['f1'],reverse=True)

        with open(os.path.join(args.output_dir,"all_trials_report.txt"),"w") as f:
            for r in sorted_results:
                
                for line in r['result_lines']:
                    f.write(line + "\n")



if __name__ == "__main__":
    main()