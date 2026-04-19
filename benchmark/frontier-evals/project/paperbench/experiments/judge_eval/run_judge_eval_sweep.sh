#!/bin/bash

export OPENAI_API_KEY="<JUDGE_EVAL_API_KEY>"

COMPLETER_CONFIG="preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config"
EXAMPLE_IDS="pinn/0,rice/0,stay-on-topic-with-classifier-free-guidance/0,all-in-one/0,semantic-self-consistency/0"
OUTPUT_DIR="experiments/judge_eval/judge_eval_results/"

if [ "$OPENAI_API_KEY" = "<JUDGE_EVAL_API_KEY>" ]; then
  echo "Error: Please set a valid OpenAI API key in the script. Replace <JUDGE_EVAL_API_KEY> with the judge eval API key."
  exit 1
fi

for model in o3-mini-2025-01-31 o1-2024-12-17 o1-mini-2024-09-12; do
  echo "Running judge eval for $model-high"
  python paperbench/scripts/run_judge_eval.py \
    judge=simple \
    completer_config="$COMPLETER_CONFIG" \
    completer_config.model=$model \
    completer_config.reasoning_effort=high \
    output_dir=$OUTPUT_DIR \
    example_ids=$EXAMPLE_IDS
  echo "-----------------------------"
done

for model in gpt-4o-mini-2024-07-18 gpt-4o-2024-08-06; do
  echo "Running judge eval for $model"
  python paperbench/scripts/run_judge_eval.py \
    judge=simple \
    completer_config="$COMPLETER_CONFIG" \
    completer_config.model=$model \
    output_dir=$OUTPUT_DIR \
    example_ids=$EXAMPLE_IDS
  echo "-----------------------------"
done

python paperbench/scripts/run_judge_eval.py \
  judge=random \
  output_dir=$OUTPUT_DIR \
  example_ids=$EXAMPLE_IDS

python paperbench/scripts/run_judge_eval.py \
  judge=dummy \
  output_dir=$OUTPUT_DIR \
  example_ids=$EXAMPLE_IDS

# finally, single run of judge-eval on o3-mini-high with --code-only
# to be able to compare token counts with default PaperBench
python paperbench/scripts/run_judge_eval.py \
  judge=simple \
  completer_config="$COMPLETER_CONFIG" \
  completer_config.model=o3-mini-2025-01-31 \
  completer_config.reasoning_effort=high \
  output_dir=experiments/judge_eval/judge_eval_results/code_only \
  example_ids=$EXAMPLE_IDS \
  code_only=true
