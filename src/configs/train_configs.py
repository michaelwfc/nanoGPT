from transformers import TrainingArguments


def get_training_args(learning_rate=1.0e-5, num_train_epochs=1, max_steps: int = 3,
                      per_device_train_batch_size=1, output_dir: str = None) -> TrainingArguments:
    training_args = TrainingArguments(

        # Learning rate
        learning_rate=learning_rate,

        # Number of training epochs
        num_train_epochs=num_train_epochs,

        # Max steps to train for (each step is a batch of data)
        # Overrides num_train_epochs, if not -1
        max_steps=max_steps,

        # Batch size for training
        per_device_train_batch_size=per_device_train_batch_size,

        # Directory to save model checkpoints
        output_dir=output_dir,

        # Other arguments
        overwrite_output_dir=False,  # Overwrite the content of the output directory
        disable_tqdm=False,  # Disable progress bars
        eval_steps=120,  # Number of update steps between two evaluations
        save_steps=120,  # After # steps model is saved
        warmup_steps=1,  # Number of warmup steps for learning rate scheduler
        per_device_eval_batch_size=1,  # Batch size for evaluation
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        optim="adafactor",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,

        # Parameters for early stopping
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    return training_args
