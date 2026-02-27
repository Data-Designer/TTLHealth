# !/usr/bin/env python
# -*-coding:utf-8 -*-
# ==============================================================================
 system_prompt_medqa = """You are a medical expert. For the following question, identify the key clinical findings, rule out distractors, and then select the correct option. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {key}. Think step by step before answering.""" # for medqa,medmcqa
#system_prompt_medqa = """"You are a professional medical doctor. Answer the following multiple-choice medical question. Think carefully and step by step. First analyze the question, then analyze each option, then give the final answer. Only use evidence-based medical knowledge. Do not use external information. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {key}."""


train_prompt_medqa = """Question: {question}"""
cot_prompt_medqa =  """
Question: {question}
Answer: ${answer}

Let's think step by step about why this answer was chosen. You should output in exactly the same format as
Thought: [the step-by-step thoughts]
Answer: $[Given Answer]
"""
format_prompt_medqa =  """Answer: ${answer}"""
demonstration_prompt_medqa = """Example {id}: \nQuestion: {question}\n Answer: ${answer}\n"""
test_prompt_medqa_icl = """Examples: \n\n{demonstration} \n\n Your Turn:\n Question: {question}"""
system_prompt_medqa_inverse = """You are a helpful assistant that provides original questions based on the user's answers."""
test_prompt_medqa_icl_inverse = """Examples: \n\n{demonstration}\nSolution: {solution}"""



# ==============================================================================
system_prompt_sum = """Simplify the following medical text for general public understanding.\n\n""" #  \n Let's think step by step. (CoT)
train_prompt_sum = """Medical Text: {question}"""
cot_prompt_sum =  """
Medical Text: {question}
Simplified Text: ${answer}

Let's think step by step about why this answer was chosen. You should output in exactly the same format as
Thought: [the step-by-step thoughts]
Simplified Text: $[Given Answer]
"""
format_prompt_sum =  """Simplified Text: ${answer}"""
demonstration_prompt_sum = """Example {id}: \nMedical Text: {question}\n Simplified Text: ${answer}\n"""
test_prompt_sum_icl = """Below is an example: \n\n{demonstration} \n\n Now please perform the simplification:\n Medical Text: {question}""" # \n Simplified Text:
system_prompt_sum_inverse = """You are a helpful assistant that provides original full content based on the user's summary."""
test_prompt_sum_icl_inverse = """Examples: \n\n{demonstration}\nSolution: {solution}"""

# ==============================================================================

# Answer the following multiple choice question.
system_prompt_diag = """You are a medical expert. For the following question, identify the key clinical findings, rule out distractors, and then select the correct option. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {key}. Think step by step before answering."""
train_prompt_diag = """Question: {question}"""
cot_prompt_diag =  """
Question: {question}
Answer: ${answer}

Let's think step by step about why this answer was chosen. You should output in exactly the same format as
Thought: [the step-by-step thoughts]
Answer: $[Given Answer]
"""
format_prompt_diag =  """Answer: ${answer}"""
demonstration_prompt_diag = """Example {id}: \nQuestion: {question}\n Answer: ${answer}\n"""
test_prompt_diag_icl = """Examples: \n\n{demonstration} \n\n Your Turn:\n Question: {question}"""
system_prompt_diag_inverse = """You are a helpful assistant that provides original questions based on the user's answers."""
test_prompt_diag_icl_inverse = """Examples: \n\n{demonstration}\nSolution: {solution}"""


# ==============================================================================
system_prompt_free = """
You are a medical expert answering open-ended clinical questions. 

【INSTRUCTIONS】
1. Think step by step to arrive at the correct answer
2. Show your clinical reasoning clearly
3. Base your answer on medical knowledge and evidence
4. The last line MUST be exactly: 'Therefore, the final answer is: $\\boxed{ANSWER}$.'
5. ANSWER should be the specific value/concept (NOT A, B, C, D)
"""

train_prompt_free = """Question: {question}"""
cot_prompt_free =  """
Question: {question}
Answer: {answer}
Let's think step by step about why this answer was chosen. You should output in exactly the same format as
Thought: [the step-by-step thoughts]
Answer: [Given Answer]
"""
format_prompt_free=  """Solution: Therefore, the final answer is: $\\boxed{{{answer}}}$."""
demonstration_prompt_free = """Example {id}: \nQuestion: {question}\n Solution: Therefore, the final answer is: $\\boxed{{{answer}}}$.\n"""
test_prompt_free_icl = """Examples: \n\n{demonstration} \n\n Your Turn:\n Question: {question}"""
system_prompt_free_inverse = """You are a helpful assistant that provides original questions based on the user's answers."""
test_prompt_free_icl_inverse = """Examples: \n\n{demonstration}\nSolution: {solution}"""



system_prompt_mcq = """You are a helpful assistant. Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {key}. Think step by step before answering."""
train_prompt_mcq = """Question: {question}"""
cot_prompt_mcq =  """
Question: {question}
Answer: ${answer}

Let's think step by step about why this answer was chosen. You should output in exactly the same format as
Thought: [the step-by-step thoughts]
Answer: $[Given Answer]
"""
format_prompt_mcq =  """Answer: ${answer}"""
demonstration_prompt_mcq = """Example {id}: \nQuestion: {question}\n Answer: ${answer}\n"""
test_prompt_mcq_icl = """Examples: \n\n{demonstration} \n\n Your Turn:\n Question: {question}"""
system_prompt_mcq_inverse = """You are a helpful assistant that provides original questions based on the user's answers."""
test_prompt_mcq_icl_inverse = """Examples: \n\n{demonstration}\nSolution: {solution}"""

# ============================================================================== public prompt
generate_questions = """You are an insightful content analyst. Your task is to deeply mine the provided text and generate multiple new, high-quality Question-Answer (QA) pairs based on its details, context, and implications.

The generated output MUST be a single Python-style list of tuples, strictly following this format:
[("question 1", "answer 1"), ("question 2", "answer 2"), ...]

Each QA pair should:
1. Explore different facets of the input (e.g., facts, logic, underlying assumptions).
2. Be self-contained and factually accurate.
3. Be concise and professional.


Now ,generate the QA pairs based on the following text:
{text}

Do not include any introductory text, explanations, or additional remarks. Output ONLY the list of tuples."""







template_config = {

# ==============================================================================
    'MedQA_MCQ': {'sys': system_prompt_medqa.format(key='ABCD'), 'format': format_prompt_medqa, 'cot': cot_prompt_medqa,
              'train': train_prompt_medqa, 'tes': test_prompt_medqa_icl, 'dem': demonstration_prompt_medqa,
              'inv_sys': system_prompt_medqa_inverse, 'inv_tes': test_prompt_medqa_icl_inverse},

    ########################################public
    'PUBLIC': {'gen_q': generate_questions, }

}

