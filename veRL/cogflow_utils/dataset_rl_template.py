rl_rm_instruction_template = \
"""[Task]
Given a user query ([Input]), multiple reference responses ([Reference Responses]), and a candidate response for evaluation ([Candidate Response]).

The reference responses are given in order, and the first reference response is the best one. You should determine whether the candidate response strictly outperforms all reference responses. Thus, 0 means the candidate response is the best one, 1 means the candidate response is worse than at least one reference responses.

[Input]
{user_input}

[Reference Responses]
{reference_responses}

[Candidate Response]
{candidate_response}

[Output]
The rank of the candidate response is: """