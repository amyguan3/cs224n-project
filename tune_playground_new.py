import os
from transformers import (WhisperProcessor,
                          WhisperForConditionalGeneration
                          )
from peft import (PeftModel,
                  PeftConfig
                  )
from eval_utils_new import (model_pipeline,
                            evaluate_asr_alt,
                            get_cv_split)
import optuna
import wandb
from alignment_util import (ParamConfig,
                            trainAdapter)
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate


def tune(accents):

    dataset = get_cv_split(accents=accents)

    ######################### hyperparam tuning ############################

    def objective(trial):
        print(f'==================TRIAL {trial.number}==================')

        # Define hyperparameters to optimize
        learning_rate = trial.suggest_uniform('learning_rate', 0.001, 0.1)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16]) # TODO: FIX THIS
        rank = trial.suggest_categorical('rank', [32, 64, 128]) # TODO: FIX THIS
        config = ParamConfig(learning_rate=learning_rate, batch_size=batch_size, rank=rank)
        print(f'Hyperparameters for trial {trial.number}:\nlearning rate: {learning_rate}, batch size: {batch_size}, rank: {rank}')

        wandb.init(project="large_test", config={"learning_rate": learning_rate, "batch_size": batch_size, "rank": rank})

        trainAdapter(config, trial.number)

        # Evaluate the model
        # TODO: FIX THIS ask azure how to get name of model
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")

        peft_config = PeftConfig.from_pretrained(peft_model_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, peft_model_path) # attaches the PEFT module to the Whisper model
        model.config.use_cache = True

        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", task="transcribe")

        pipe = model_pipeline(model, processor, baseline=False)
        eval_result = evaluate_asr_alt(pipe, dataset["train"], True)
        print(f'metrics: {eval_result}')

        # cumulative WER
        wer = 0
        for accent in eval_result:
            wer += eval_result[accent]['wer']
        wer /= len(eval_result)
        print(f'Average WER: {wer}')
            
        # Log metrics to wandb
        wandb.log({"trial": trial.number, "eval_wer": wer})

        return wer

    # Define Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=3)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    importances = optuna.importance.get_param_importances(study)
    print("Importances: {importances}")

    params_sorted = list(importances.keys())
    print("Sorted params: {params_sorted}")

    try:
        fig = plot_parallel_coordinate(study)
        os.system(f"mkdir -p tune_plots")
        fig.savefig('tune_plots/parallel.png')
        fig = plot_optimization_history(study)
        fig.savefig('tune_plots/history.png')
    except:
        print('ERROR GENERATING PLOTS')

    # best model??


def main():
    accents = ["Scottish English", "India and South Asia (India, Pakistan, Sri Lanka)"]
    tune(accents)


if __name__ == "__main__":
    main()