![Logo](images/logo.png)

---

We introduce **AgentIF**, the first benchmark for systematically **evaluating LLM instruction following ability in agentic scenarios**. AgentIF features three key characteristics: (1) **Realistic**, constructed from 50 real-world agentic applications. (2) **Long**, averaging 1,723 words with a maximum of 15,630 words. (3) **Complex**, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. Here is the instruction length distribution in AgentIF, along with the success rates of several representative LLMs across the constraint dimensions we propose:
![Logo](images/fig1.png)
An example instruction of AgentIF:
![Logo](images/example_1.png)

## Leaderboard

#### Metrics
- Constraint Success Rate (CSR) measures the proportion of individual constraints that are correctly satisfied by the model’s response. 
- Instruction Success Rate (ISR) measures the proportion of instructions for which all constraints are satisfied. 

#### Performance Across Constraint Categories

![Logo](images/experiment.png)


## Evaluation
For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation.

### How to evaluation
1. Clone the remote repository to your local environment. The necessary data is already included, so no further actions are needed.
    ```
    git clone https://github.com/THU-KEG/AgentIF.git
    ```

2. Specify the target model and the evaluator in the `run.sh`  file. We recommend using `gpt-4o-2024-11-20` to reproduce our results.
    
    ```
    Model_Name=""
    Model_Name_URL=""
    Model_Name_API_Key="EMPTY"
    
    Evaluator_Model_Backbone=""
    Evaluator_URL=""
    Evaluator_API_Key=""
    ```
    
3. Then run the script to start the evaluation.
    
    ```
    sh run.sh
    ```

## Citation

```
# TODO
```