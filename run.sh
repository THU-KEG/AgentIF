if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Run the prediction script
python3 code4eval/0.prediction_api.py \
    --input_dir data/ \
    --output_dir results/prediction/${MODEL_NAME} \
    --llm_backbone ${MODEL_NAME} \
    --llm_url ${MODEL_URL} \
    --api_key ${MODEL_API_KEY} \
    --cache .cache/evaluation/${MODEL_NAME}.pkl

# Run the evaluation script
python3 code4eval/1.evaluation_api.py \
    --input_file results/prediction/${MODEL_NAME}/eval.json \
    --output_dir results/scores/${MODEL_NAME}_${EVALUATOR_MODEL_BACKBONE} \
    --llm_backbone ${EVALUATOR_MODEL_BACKBONE} \
    --llm_url ${EVALUATOR_URL} \
    --api_key ${EVALUATOR_API_KEY} \
    --cache .cache/evaluation/${EVALUATOR_MODEL_BACKBONE}.pkl
