cog_ordinary_NBN_content_template_en = \
"""### **[Task]**
- Background: You are a assistant helping answering problems. You need to carry out the next step [{node_name}] of a reasoning chain to help response to [User Input]. 
- Requirements: 
   * follow the instructions in **Reasoning Step: [{node_name}]** which defines the reasoning step. 
   * It should be the next step of the half-finished reasoning chain in [Existing Analysis]. The result can only be not aligned with analysis in [Existing Analysis] if you need to fix mistakes or explore aspects not considered. You can refer to [Analysis Expectation] for guidance, but do not need to strictly follow it. 
   * **Important**: Your output should be specific, without fake information. 
   * **Important**: Your output should be comprehensive, including any possible aspects. 
   * **Important**: Your output should only contain one step. If other things is in need, state the need and reserve the reasoning to next steps. 
   * You should only use simpler vocabulary at a high school level to form your answers.

### **[User Input]**
{user_input}

### **[Existing Analysis]**
{previous_nodes}

### **[Analysis Expectation]**
{analyze_expect}

### **Reasoning Step: [{node_name}]**
{node_description}

### **[Output Format]**
please output in English. The content should be a smooth and coherent paragraph, following the format below: 
```json
{{
	"content": "the content of the required step"
}}
"""

cog_ordinary_NBN_result_full_template_en = \
"""
{user_input}
Please answer under the guidance of the following think: 
<|begin think|> 
{previous_nodes}
<|end think|>
"""

cog_ordinary_NBN_content_node_en = {
	"Observation": """
- **Task**: Observe and interpret the specific behaviors, attitudes, or other information from the current context. The extracted facts must be precise and detailed without vague information. **NOTE: ALL THE INFORMATION MUST BE ALIGNED WITH THE CONTEXT. DO NOT MAKE UP FAKE INFORMATION.**
- **Output**: State observation comprehensively.""", 

	"Attribution": """
- **Task**: Attribute and evaluate the events or behaviors. It might include: Causal reasoning for others' actions / Impact assessment on current context / Further analysis and explanation. 
- **Output**: State the specific reason comprehensively.""", 

	"Motivation": """
- **Task**: Generate motivation and goals, addressing the main problem discovered in other steps.
- **Output**: State the goal and motivation.""", 

	"Regulation": """
- **Task**: Check and adjust the previous thought to form a revised motivation or perception, or action plan. You should check (1) whether it lacks consideration, (2) whether other requirements need to be noticed. Think of the effect of the current plan or behavior, and check if there exists any risk. You should also check if there are any misunderstandings and be suspicious of the information in the analysis.
- **Output**: Accurately and comprehensively state the problem and how to solve it.""", 

	"Efficacy": """
- **Task**: Assess the internal perceptions, emotions, and beliefs of the actor of some behavior, and adjust the perception or action plan.
- **Output**: State the efficacy and adjustment of action.""", 

	"Behavior": """
- **Task**: Determine a more complete behavior based on the current environment and the analysis. """, 

	"Terminate": """
- **Task**: conclude the reasoning and give the final result. 
""", 
}

cog_ordinary_NBN_choose_template_en = \
"""## **[Task]**
- Background: You are an assistant helping with problems. You need to choose the next step of a reasoning chain to help respond to the user's input in [User Input]. The chain should be comprehensive. 
- Requirements: 
   * You should select **ALL** possible candidates from [Candidate Next Steps] that can be a reasonable next ONE step of the half-finished reasoning chain provided in [Existing Analysis]. You could visit the same step several times to get more information or analyze further. 
   * If analysis is sufficient for responding to the [User Input], and there are no concerns, DIRECTLY select the [Terminate] step. (NOTE: If you are not certain or you think there might be other potentials, you must choose other nodes along with Terminate. )
   * If you find some bad steps in [Existing Analysis] (for example: misinformation, unclear statement, etc. ), redoing it again might refine it. 
   * If more than one valid options exist, list the most applicable 2 or 3 steps, and put the most applicable one in the first place. 
   * The names of the next steps should be exactly the same as the name, e.g., Attribution and Evaluation. 
   * You should first review the prior steps in [Existing Analysis], and then determine the candidates for the next step. 


## **[User Input]**
{user_input}

## **[Existing Analysis]**
{previous_nodes}

## **[Candidate Next Steps]**

- **[Observation]** 
   * Observe the specific behaviors or attitudes from the current context. 

- **[Regulation]** 
   * Validate and refine previous thoughts: (1) consider twice to polish the thought, behavior, or motivation, (2) check if there exists more information in the scenario that needs to be considered. 

- **[Behavior]** 
   * derive context-specific behaviors. 

- **[Efficacy]** 
   * analyze and adjust internal perceptions of the scene and action plan. 

- **[Attribution]** 
   * further interprets the result of previous steps, may include Causal reasoning for others' actions, or Impact assessment on the current context. 

- **[Terminate]** 
   * Terminate analysis, synthesize final conclusion, and respond to the user. 

- **[Motivation]** 
   * formulate one's primary drivers of oneself, based on their needs/desires identified in other steps. 

## **[Output Format]**
```json
{{
    "rationale": "Concise justification for selecting the next one step candidates, and choose the most likely one",
    "next_step_candidates": ["step name", ...]
}}
```"""
