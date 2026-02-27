
import random
############ 数据集一些特殊的处理，随时可以修改
from string import punctuation

def preprocess_math_collections(dataset, dataset_name):
    if dataset_name in ['AIME2024', 'AIME2025']:
        # 数据提取
        QAs = [{'Q': x, 'A': y}
                    for x, y in zip(dataset['problem'], dataset['answer'])]
    elif dataset_name in ['GSM8K']:
        QAs = [{'Q':x, 'A':y.split('####')[-1].strip()}
                    for x,y in zip(dataset['question'], dataset['answer'])]
    elif dataset_name in ['HMMT2025']:
        QAs = [{'Q': x, 'A': y}
                    for x, y in zip(dataset['problem'], dataset['answer'])]
    elif dataset_name in ['AMC']:
        QAs = [{'Q': x, 'A': y}
                    for x, y in zip(dataset['problem'], dataset['solution'])]
    elif dataset_name in ['MATH500']:
        QAs = [{'Q': x, 'A': y}
                    for x, y in zip(dataset['problem'], dataset['solution'])]
    else:
        raise NotImplementedError(f"Preprocess dataset {dataset_name} not supported yet.")
    return QAs


def preprocess_med_collections(dataset, dataset_name):
    ref_dataset = {'question': [], 'answer': [], 'options': []}
    # 先默认都是一样的
    if dataset_name in ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA','MedQA', 'PubMedQA', 'AfrimedQA', 'MedxpertQA-R', 'MedxpertQA-U']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample[
                                                                                                    'question'] + '?'
            options = ''
            for key, value in raw_sample['options'].items():
                options += f"{key}: {value} \n"
            gold_answer = raw_sample['answer_idx']
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['options'].append(options)
        return ref_dataset
    elif dataset_name in ['GPQA']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            choices = [
                raw_sample['Incorrect Answer 1'],
                raw_sample['Incorrect Answer 2'],
                raw_sample['Incorrect Answer 3'],
                raw_sample['Correct Answer']
            ]
            choices = [choice.strip() for choice in choices]
            random.shuffle(choices)
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=raw_sample["Question"]
            )
            correct_answer_idx = choices.index(raw_sample['Correct Answer'].strip())
            task_template = """{Question}
            A) {A}
            B) {B}
            C) {C}
            D) {D}
            """.strip()

            task = task_template.format(
                Question=choices_dict['Question'],
                A=choices_dict['A'],
                B=choices_dict['B'],
                C=choices_dict['C'],
                D=choices_dict['D'])

            answer = "A" if correct_answer_idx == 0 \
                else "B" if correct_answer_idx == 1 \
                else "C" if correct_answer_idx == 2 \
                else "D"
            ref_dataset['question'].append(task)
            ref_dataset['answer'].append(answer)
            options = '\n'.join([key + value for key, value in zip(choices_dict.keys(), choices_dict.values()) if key != 'Question'])
            ref_dataset['options'].append(options)
        return ref_dataset
    else:
        raise NotImplementedError(f"Preprocess dataset {dataset_name} not supported yet.")





def preprocess_sum_collections(dataset, dataset_name):
    ref_dataset = {'article': [], 'summary': []}
    if dataset_name in ['eLife']: # https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation/tree/main
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['source']# raw_sample['article']
            summary = raw_sample['target']# raw_sample['summary']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['PLOS']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['source']
            summary = raw_sample['target']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['Cochrane']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['source']
            summary = raw_sample['target']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['SumPubmed']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['text']
            summary = raw_sample['shorter_abstract']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['MedQsum']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['query']
            summary = raw_sample['answer']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['ACI-Bench']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['dialogue']
            summary = raw_sample['note']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    elif dataset_name in ['MTS-Diag']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            article = raw_sample['dialogue']
            summary = raw_sample['section_text']
            ref_dataset['article'].append(article)
            ref_dataset['summary'].append(summary)
        return ref_dataset
    else:
        raise NotImplementedError(f"Preprocess dataset {dataset_name} not supported yet.")







def preprocess_diag_collections(dataset, dataset_name):
    ref_dataset = {'question': [], 'answer': [], 'options': [], 'context': [], 'answer_text': []}
    if dataset_name in ['DiagnosisArena']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = "Patient Case: " + raw_sample['Case Information'] + '\n' + "Phsical Exam: " + raw_sample['Physical Examination'] + '\n' + "Diagnostic Tests: " + raw_sample['Diagnostic Tests']  + '\n' + "What is the most likely diagnosis?"
            options = raw_sample['Options']
            anwer_map = ['A', 'B', 'C', 'D']
            gold_answer = anwer_map[raw_sample['Right Option']]
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append("Patient Case: " + raw_sample['Case Information'] + '\n' + "Phsical Exam: " + raw_sample['Physical Examination'] + '\n' + "Diagnostic Tests: " + raw_sample['Diagnostic Tests'])
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(raw_sample['Final Diagnosis'])
            
        return ref_dataset
    elif dataset_name in ['ReDis']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question =  raw_sample['question']
            options = 'A.' + raw_sample['opa'] + '\n' + ', B.' + raw_sample['opb'] + '\n' ', C.' + raw_sample['opc'] + '\n' + ', D.' + raw_sample['opd']
            anwer_map = ['A', 'B', 'C', 'D']
            options_text = [raw_sample['opa'], raw_sample['opb'], raw_sample['opc'], raw_sample['opd']]
            gold_answer = anwer_map[raw_sample['cop']]
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('')
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(options_text[raw_sample['cop']])
        return ref_dataset
    elif dataset_name in ['CupCase']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = raw_sample['clean_case_presentation']
            if random.random() < 0.25:
                options = 'A.' + raw_sample['correct_diagnosis'] + '\n' + ', B.' + raw_sample['distractor1'] + '\n' ', C.' + raw_sample['distractor2'] + '\n' + ', D.' + raw_sample['distractor3']
                gold_answer = 'A'
            elif random.random() < 0.5:
                options = 'A.' + raw_sample['distractor1'] + '\n' + ', B.' + raw_sample['correct_diagnosis'] + '\n' ', C.' + raw_sample['distractor2'] + '\n' + ', D.' + raw_sample['distractor3']
                gold_answer = 'B'
            elif random.random() < 0.75:
                options = 'A.' + raw_sample['distractor1'] + '\n' + ', B.' + raw_sample['distractor2'] + '\n' ', C.' + raw_sample['correct_diagnosis'] + '\n' + ', D.' + raw_sample['distractor3']
                gold_answer = 'C'
            else:
                options = 'A.' + raw_sample['distractor1'] + '\n' + ', B.' + raw_sample['distractor2'] + '\n' ', C.' + raw_sample['distractor3'] + '\n' + ', D.' + raw_sample['correct_diagnosis']
                gold_answer = 'D'
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('')
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(raw_sample['correct_diagnosis'])
        return ref_dataset
    elif dataset_name in ['MediQ']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            # print(raw_sample['facts_old'])

            if raw_sample['atomic_facts'] is not None:
                raw_sample['atomic_facts'] = "\n".join(list(raw_sample['atomic_facts']))
            else:
                raw_sample['atomic_facts'] = " "
            question = 'Patient Case: ' + raw_sample['atomic_facts'] + '\n' + raw_sample['question'] # 有些人没有
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('Patient Case: ' + raw_sample['atomic_facts'])
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(raw_sample['answer'])
        return ref_dataset
    elif dataset_name in ['PubHealth']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            # print(raw_sample['facts_old'])

            question = 'Guideline: ' + raw_sample['retrieved_context_for_judge'] + '\n' + raw_sample['question']
            options = raw_sample['options_formatted']
            gold_answer = raw_sample['answer']
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('Guideline: ' + raw_sample['retrieved_context_for_judge'])
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(raw_sample['options'][raw_sample['answer_index']])
        return ref_dataset
    elif dataset_name in ['QASC']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = 'Facts: ' + raw_sample['combinedfact'] + '\n' + raw_sample['formatted_question']
            options_text, options_ids = raw_sample['choices']['text'], raw_sample['choices']['label']
            options = [idx + '.' + text for idx, text in zip(options_ids, options_text)]
            if raw_sample['answerKey'] == '':
                print("No valid label")
                continue
            gold_answer = raw_sample['answerKey'].strip()
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('Facts: ' + raw_sample['combinedfact'])
            ref_dataset['options'].append(options)
            # print(options_ids, gold_answer)
            ref_dataset['answer_text'].append(options_text[options_ids.index(gold_answer)])
        return ref_dataset

    elif dataset_name in ['ReClor']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = 'Facts: ' + raw_sample['context'] + '\n' + raw_sample['question']
            options_text, options_ids = raw_sample['answers'], ['A', 'B', 'C', 'D']
            options = [idx + '.' + text for idx, text in zip(options_ids, options_text)]
            if raw_sample['label'] == '':
                print("No valid label")
                continue
            gold_answer_id = int(raw_sample['label'])
            gold_answer = options_ids[gold_answer_id]
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('Facts: ' + raw_sample['context'])
            ref_dataset['options'].append(options)
            # print(options_ids, gold_answer)
            ref_dataset['answer_text'].append(options_text[int(raw_sample['label'])])
            if i==0:
                print("XXXXXX", ref_dataset['answer_text'])
        return ref_dataset
    elif dataset_name in ['LOGIQA']:
        for i in range(len(dataset)):
            raw_sample = dataset[i]
            question = 'Context: ' + raw_sample['context'] + '\n' + raw_sample['query']
            options = raw_sample['options']
            options_idx = ['A', 'B', 'C', 'D']
            answer_text = options[raw_sample['correct_option']]
            gold_answer = options_idx[raw_sample['correct_option']]
            options = [f"{idx}. {text}" for idx, text in zip(options_idx, options)]
            options = '\n'.join(options)
            question = f"{question} \n" \
            f"Options: {options} \n"
            ref_dataset['question'].append(question)
            ref_dataset['answer'].append(gold_answer)
            ref_dataset['context'].append('Context: ' + raw_sample['context'])
            ref_dataset['options'].append(options)
            ref_dataset['answer_text'].append(answer_text)
        return ref_dataset

    else:
        raise NotImplementedError(f"Preprocess dataset {dataset_name} not supported yet.")





