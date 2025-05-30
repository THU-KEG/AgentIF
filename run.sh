Model_Name=""
Model_Name_URL=""
Model_Name_API_Key=""

Evaluator_Model_Backbone=""
Evaluator_URL=""
Evaluator_API_Key=""


python code4eval/0.prediction_api.py \
    --input_dir data/ \
    --output_dir results/prediction/${Model_Name} \
    --llm_backbone ${Model_Name} \
    --llm_url ${Model_Name_URL} \
    --api_key ${Model_Name_API_Key} \
    --cache .cache/evaluation/${Model_Name}.pkl 


python code4eval/1.evaluation_api.py \
    --input_file results/prediction/${Model_Name}/eval.json \
    --output_dir results/scores/${Model_Name}_${Evaluator_Model_Backbone} \
    --llm_backbone ${Evaluator_Model_Backbone} \
    --llm_url ${Evaluator_URL} \
    --api_key ${Evaluator_API_Key} \
    --cache .cache/evaluation/${Evaluator_Model_Backbone}.pkl
