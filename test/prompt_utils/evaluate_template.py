output_direct_score_template = \
"""**[Task]**

Given the [User Input] and the corresponding multiple answers in [Answers] (which are in random order), please score these answers on a scale of 1-10 (the higher the score, the better) according to the following principles:

(1) Based on the [User Input] and all the answers in [Answers], propose evaluation criteria that can assess the quality of the given answers. Ensure that under these criteria, the first answer listed receives a score of 5.
(2) Explain the scoring principle for each score value sequentially.
(3) Score all the answers in [Answers] based on the established evaluation criteria. You must use the first listed answer as a 5-point reference sample and provide a reason for each score.
(4) Refer to the [Output Format] for the output structure.

Note: You must ensure that the scores for the [Answers] are well-differentiated.

Special Attention: You must ensure that the first answer listed receives a score of 5 under your scoring standard. Use this first answer as a baseline (referred to as "Baseline"). For subsequent answers, a score greater than 5 must mean it is better than the first answer, and a score less than 5 must mean it is worse than the first answer.

**[User Input]**
{user_input}

**[Answers]**
{answers}

**[Output Format]**
Output in JSON format, with every answer giving one score in 1-10. The answer is identified by 'id'. 
```json
{{
    "think": "analyze the user's input, come up with some criterion", 
    "standard": [
        {{
            "score": 10, 
            "standard": "standard of score 10"
        }}, 
        ...
        {{
            "score": 5, 
            "standard": "standard of score 5"
        }}, 
        ... 
        {{
            "score": 1, 
            "standard": "standard of score 1"
        }}, 
    ], 
    "result": [
        {{
            "id": ..., 
            "reason": "compare with the first answer (Baseline), and then judge the quality of this answer", 
            "score": evaluated socre
        }}, 
        ...
        {{
            "id": ..., 
            "reason": "compare with the first answer (Baseline), and then judge the quality of this answer", 
            "score": evaluated score
        }}
    ]  
}}
```
"""

process_direct_score_template = \
"""**[Task]**

Please evaluate the cognitive flow provided in the [Reasoning Flow] based on the three core criteria listed below. You need to score each criterion independently on a scale of 1-10 (the higher the score, the better) and provide a reason for each score.

Evaluation Criteria:
- **Coherence**: Is it logically sound and free of internal contradictions?
- **Interpretability**: Does it clearly explain the social dynamics or core mechanisms involved?
- **Predictability**: Does it offer reasonable insight into the future evolution of the social dynamics?

Please strictly follow the JSON format required in the [Output Format].

**[Reasoning Flow]**
{reasoning_flow}

**[Output Format]**
Please output in JSON format. The JSON structure should include your thought process, the independent scores, and reasons for each criterion.
```json
{{
    "think": "evaluation process",
    "evaluation_result": {{
        "coherence": {{
            "reason": "Explain your reasoning", 
            "score": evaluated_score
        }},
        "interpretability": {{
            "reason": "Explain your reasoning", 
            "score": evaluated_score
        }},
        "predictability": {{
            "reason": "Explain your reasoning", 
            "score": evaluated_score
        }}
    }}
}}
```
"""

output_direct_score_no_ref_template = \
"""**[Task]**

Given a [User Input] and its corresponding [Answer], please provide a comprehensive score between 1 and 10 based on the quality of the answer, where a higher score indicates a better answer. A score of 5 indicates that the answer is basically correct but may be incomplete, unclear, or partially inaccurate.

Scoring Criteria Explanation (for reference; please make a comprehensive judgment):

- 10: Perfect answer. Entirely accurate, informative, well-structured, and appropriately worded. Effectively addresses the user's query, potentially even exceeding expectations.
- 8-9: Excellent answer. Accurate and complete in information, logically clear, fluently expressed, fully satisfying the user's needs.
- 6-7: Good answer. Basically correct and relevant, but may lack depth in certain details or contain minor inaccuracies.
- 5: Passable answer. Generally correct but potentially incomplete, somewhat unclear, or containing individual errors that do not severely impact understanding.
- 3-4: Insufficient answer. Partially relevant but missing key information, containing significant errors, or failing to address the core issue.
- 1-2: Poor answer. Severely off-topic, containing incorrect information, or entirely unhelpful.

When evaluating, you may comprehensively consider the following dimensions (not all are required):
- Accuracy: Whether the answer is factually correct and non-misleading.
- Completeness: Whether it covers the key points of the user's question.
- Relevance: Whether the answer stays closely aligned with the user's question without deviating from the topic.
- Clarity: Whether the expression is clear, easy to understand, and well-organized.
- Practicality: Whether it offers practical help to the user and is actionable (if applicable).

**[User Input]**
{user_input}

**[Answer]**
{answer}

**[Output Format]**
Output in JSON format, with the answer given one integer score in 1-10. 
```json
{{
    "score": evaluated score
}}
```"""