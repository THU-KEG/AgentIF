# AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios

We introduce AgentIF, the first benchmark for systematically evaluating LLM instruction following ability in agentic scenarios. AgentIF features three key characteristics: (1) Realistic, constructed from 50 real-world agentic applications. (2) Long, averaging 1,723 words with a maximum of 15,630 words. (3) Complex, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. To construct AgentIF, we collect 707 human-annotated instructions across 50 agentic tasks from industrial application agents and open-source agentic systems. For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation.

## How to evaluation

1. Specify the target model and the evaluator in the `run.sh`  file. We recommend using `gpt-4o-2024-11-20` to reproduce our results.
    
    ```
    Model_Name=""
    Model_Name_URL=""
    Model_Name_API_Key="EMPTY"
    
    Evaluator_Model_Backbone=""
    Evaluator_URL=""
    Evaluator_API_Key=""
    ```
    
2. Then run the script to start the evaluation.
    
    ```
    sh run.sh
    ```