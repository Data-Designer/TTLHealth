# !/usr/bin/env python
# -*-coding:utf-8 -*-
import os.path

import torch
from langchain_core.prompts import PromptTemplate
from torch.utils.data import Dataset, DataLoader
from utils import formulate_dem, load_pickle, save_pickle, get_cot_answer, get_completions
from vllm import LLM, SamplingParams
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer



class SimpleTextDataset(Dataset):
    """一个简单的 Dataset，直接返回原始数据字典。"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        """返回数据集的总样本数。"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引 idx 返回原始的样本字典。"""
        return self.data[idx]


def get_loader_fn(data_root, task_name, train_dic, test_dic, template, config, index_root=None, special_op=None):
    sys_tem, train_tem, prompt_tem, demonstration_tem, format_tem, cot_tem = template['sys'], template['train'], \
    template['tes'], template.get('dem', None), template['format'], template['cot']
    if config['GET_COT']:
        train_file_check = data_root + f'/train_dataset_tem_cot_{task_name}.pkl'
        test_file_check = data_root + f'/test_dataset_tem_cot_{task_name}.pkl'
    else:
        train_file_check = data_root + f'/train_dataset_tem_{task_name}.pkl'
        test_file_check = data_root + f'/test_dataset_tem_{task_name}.pkl'
    if os.path.exists(train_file_check):
        print("Already template format!")
        train_dic = load_pickle(train_file_check)
        test_dic = load_pickle(test_file_check)
        test_dic = special_change_template(test_dic, config, template)

        if config['RE_RET']:
            print("Skip re-generating synthetic pairs. Just re-retrieveing.")
            from utils import re_gather_dataset
            _, test_dic = re_gather_dataset(train_dic, test_dic, index_root, special_op, config)
    else:
        print(
            f"First time formatting dataset with template! Note that you may get syn pair for a long time (for test {len(test_dic)} samples).")
        synthetic_llm = LLM(model=config['LLM_PATH'],
                            trust_remote_code=True,
                            gpu_memory_utilization=0.9,
                            tensor_parallel_size=1)
        synthetic_tokenizer = AutoTokenizer.from_pretrained(config['LLM_PATH'])

        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=config['MAX_LEN'],
            min_p=0,
            top_p=0.95,  
            top_k=20,
        )

        for i in range(len(train_dic)):
            prompt = PromptTemplate.from_template(train_tem)
            train_dic[i]['qid'] = i
            train_dic[i]['prompt'] = prompt.format(question=train_dic[i]['Q'],
                                                   )
            train_dic[i]['system'] = sys_tem
            answer = train_dic[i]['A']
            train_dic[i]['answer'] = answer
            train_dic[i]['format_answer'] = format_tem.format(answer=answer)
            if config['GET_COT']:
                train_dic[i]['format_answer'], train_dic[i]['cot'] = get_cot_answer(synthetic_llm, synthetic_tokenizer,
                                                                                    sampling_params, cot_tem,
                                                                                    train_dic[i]['prompt'],
                                                                                    train_dic[i]['format_answer'])

        for i in range(len(test_dic)):
            test_dic[i]['qid'] = i
            retrieval_icl = test_dic[i].get('retrieved_texts', [])
            retrieval_icl = retrieval_icl[:config['TOPK']]
            if config['GET_COT']:
                new_retrieval_icl = []
                for q, a in retrieval_icl:
                    format_answer = format_tem.format(answer=a)
                    cot_answer, _ = get_cot_answer(synthetic_llm, sampling_params, cot_tem, q, format_answer)
                    new_retrieval_icl.append((q, cot_answer))
                retrieval_icl = new_retrieval_icl
                test_dic[i]['retrieved_texts'] = new_retrieval_icl

            demonstrations = formulate_dem(retrieval_icl,
                                           demonstration_tem)
            prompt = PromptTemplate.from_template(prompt_tem)
            test_dic[i]['prompt'] = prompt.format(question=test_dic[i]['Q'],
                                                  demonstration=demonstrations)
            test_dic[i]['system'] = sys_tem
            answer = test_dic[i]['A']
            test_dic[i][
                'answer'] = answer
            test_dic[i]['format_answer'] = format_tem.format(answer=answer)

            sys_prompt = test_dic[i]['system']
            prompt = PromptTemplate.from_template(train_tem)
            user_prompt = prompt.format(question=test_dic[i]['Q'],
                                        )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            generated_text = get_completions(messages, synthetic_llm, synthetic_tokenizer, sampling_params)

            if config['GET_COT']:
                generated_text, _ = get_cot_answer(synthetic_llm, synthetic_tokenizer, sampling_params, cot_tem,
                                                   user_prompt, generated_text)

            test_dic[i]['syn_pair'] = (test_dic[i]['Q'], generated_text)
            if i < 2:
                print("First synthetic pair example:", test_dic[i]['syn_pair'])
                print("Ground Truth answer:", test_dic[i]['A'])

        del synthetic_llm
        torch.cuda.empty_cache()  # 防止显存泄漏

        save_pickle(train_dic, train_file_check)
        save_pickle(test_dic, test_file_check)
        print("First time formatting and saved!")

    if (config['DATASET'] in ['AIME2024', 'AIME2025'] + ['GPQA'] + ['GSM8K'] + ['AMC', 'HMMT2025', 'MATH500'] +
            ['MMLU', 'MMLU-Pro', 'MedBullets', 'MedExQA', 'MedMCQA', 'MedQA', 'PubMedQA', 'AfrimedQA', 'MedxpertQA-R',
             'MedxpertQA-U']
            + ['SumPubmed', 'eLife', 'Cochrane', 'PLOS', 'ACI-Bench', 'MTS-Diag', 'MedQsum'] + ['DiagnosisArena',
                                                                                                'ReDis', 'CupCase',
                                                                                                'MediQ',
                                                                                                'PubHealth'] + [
                'LOGIQA', 'QASC', 'ReClor']):
        train_dataset = SimpleTextDataset(train_dic)
        test_dataset = SimpleTextDataset(test_dic)
        
        return train_dataset, test_dataset
    else:
        raise NotImplementedError(f"Dataloader {config['DATASET']} is not supported.")


def get_demonstrations(dataset_name):
    # 参照medicalagents board 的
    if dataset_name == 'eLife':
        demonstrations = [(
            "In temperate climates, winter deaths exceed summer ones. However, there is limited information on the timing and the relative magnitudes of maximum and minimum mortality, by local climate, age group, sex and medical cause of death. We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions. Death rates in men and women \u2265 45 years peaked in December to February and were lowest in June to August, driven by cardiorespiratory diseases and injuries. In these ages, percent difference in death rates between peak and minimum months did not vary across climate regions, nor changed from 1980 to 2016. Under five years, seasonality of all-cause mortality largely disappeared after the 1990s. In adolescents and young adults, especially in males, death rates peaked in June/July and were lowest in December/January, driven by injury deaths.",
            "In the USA, more deaths happen in the winter than the summer. But when deaths occur varies greatly by sex, age, cause of death, and possibly region. Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment. Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed, or whether existing ones are effective. Scrutinizing seasonal patterns in death over time can also help scientists determine whether large-scale weather or climate changes are affecting the seasonality of death. Now, Parks et al. show that there are age and sex differences in which times of year most deaths occur. Parks et al. analyzed data on US deaths between 1980 and 2016. While overall deaths in a year were highest in winter and lowest in summer, a greater number of young men died during summer \u2013 mainly due to injuries \u2013 than during winter. Seasonal differences in deaths among young children have largely disappeared and seasonal differences in the deaths of older children and young adults have become smaller. Deaths among women and men aged 45 or older peaked between December and February \u2013 largely caused by respiratory and heart diseases, or injuries. Deaths in this older age group were lowest during the summer months. Death patterns in older people changed little over time. No regional differences were found in seasonal death patterns, despite large climate variation across the USA. The analysis by Parks et al. suggests public health and medical interventions have been successful in reducing seasonal deaths among many groups. But more needs to be done to address seasonal differences in deaths among older adults. For example, by boosting flu vaccination rates, providing warnings about severe weather and better insulation for homes. Using technology like hands-free communication devices or home visits to help keep vulnerable elderly people connected during the winter months may also help."
        ),
            (
                "Whether complement dysregulation directly contributes to the pathogenesis of peripheral nervous system diseases, including sensory neuropathies, is unclear. We addressed this important question in a mouse model of ocular HSV-1 infection, where sensory nerve damage is a common clinical problem. Through genetic and pharmacologic targeting, we uncovered a central role for C3 in sensory nerve damage at the morphological and functional levels. Interestingly, CD4 T cells were central in facilitating this complement-mediated damage. This same C3/CD4 T cell axis triggered corneal sensory nerve damage in a mouse model of ocular graft-versus-host disease (GVHD). However, this was not the case in a T-dependent allergic eye disease (AED) model, suggesting that this inflammatory neuroimmune pathology is specific to certain disease etiologies. Collectively, these findings uncover a central role for complement in CD4 T cell-dependent corneal nerve damage in multiple disease settings and indicate the possibility for complement-targeted therapeutics to mitigate sensory neuropathies.",
                "Most people have likely experienced the discomfort of an eyelash falling onto the surface of their eye. Or that gritty sensation when dust blows into the eye and irritates the surface. These sensations are warnings from sensory nerves in the cornea, the transparent tissue that covers the iris and pupil. Corneal nerves help regulate blinking, and control production of the tear fluid that protects and lubricates the eye. But if the cornea suffers damage or infection, it can become inflamed. Long-lasting inflammation can damage the corneal nerves, leading to pain and vision loss. If scientists can identify how this happens, they may ultimately be able to prevent it. To this end, Royer et al. have used mice to study three causes of hard-to-treat corneal inflammation. The first is infection with herpes simplex virus (HSV-1), which also causes cold sores. The second is eye allergy, where the immune system overreacts to substances like pollen or pet dander. And the third is graft-versus-host disease (GVHD), an immune disorder that can affect people who receive a bone marrow transplant. Royer et al. showed that HSV-1 infection and GVHD \u2013 but not allergies \u2013 made the mouse cornea less sensitive to touch. Consistent with this, microscopy revealed damage to corneal nerves in the mice with HSV-1 infection and those with GVHD. Further experiments showed that immune cells called CD4 T cells and a protein called complement C3 were contributing to this nerve damage. Treating the mice with an experimental drug derived from cobra venom protected the cornea from the harmful effects of inflammation. It did so by blocking activation of complement C3 at the eye surface. Identifying factors such as complement C3 that are responsible for corneal nerve damage is an important first step in helping patients with inflammatory eye diseases. Many drugs that target the complement pathway are currently under development. Some of these drugs could potentially be adapted for delivery as eye drops. But first, experiments must test whether complement also contributes to corneal nerve damage in humans. If it does, work can then begin on testing these drugs for safety and efficacy in patients."
            )
        ]
    elif dataset_name == 'PLOS':
        demonstrations = [(
            "The guidance cue UNC-6/Netrin regulates both attractive and repulsive axon guidance. Our previous work showed that in C. elegans, the attractive UNC-6/Netrin receptor UNC-40/DCC stimulates growth cone protrusion, and that the repulsive receptor, an UNC-5: UNC-40 heterodimer, inhibits growth cone protrusion. We have also shown that inhibition of growth cone protrusion downstream of the UNC-5: UNC-40 repulsive receptor involves Rac GTPases, the Rac GTP exchange factor UNC-73/Trio, and the cytoskeletal regulator UNC-33/CRMP, which mediates Semaphorin-induced growth cone collapse in other systems. The multidomain flavoprotein monooxygenase (FMO) MICAL (Molecule Interacting with CasL) also mediates growth cone collapse in response to Semaphorin by directly oxidizing F-actin, resulting in depolymerization. The C. elegans genome does not encode a multidomain MICAL-like molecule, but does encode five flavin monooxygenases (FMO-1, -2, -3, -4, and 5) and another molecule, EHBP-1, similar to the non-FMO portion of MICAL. Here we show that FMO-1, FMO-4, FMO-5, and EHBP-1 may play a role in UNC-6/Netrin directed repulsive guidance mediated through UNC-40 and UNC-5 receptors. Mutations in fmo-1, fmo-4, fmo-5, and ehbp-1 showed VD/DD axon guidance and branching defects, and variably enhanced unc-40 and unc-5 VD/DD axon guidance defects. Developing growth cones in vivo of fmo-1, fmo-4, fmo-5, and ehbp-1 mutants displayed excessive filopodial protrusion, and transgenic expression of FMO-5 inhibited growth cone protrusion. Mutations suppressed growth cone inhibition caused by activated UNC-40 and UNC-5 signaling, and activated Rac GTPase CED-10 and MIG-2, suggesting that these molecules are required downstream of UNC-6/Netrin receptors and Rac GTPases. From these studies we conclude that FMO-1, FMO-4, FMO-5, and EHBP-1 represent new players downstream of UNC-6/Netrin receptors and Rac GTPases that inhibit growth cone filopodial protrusion in repulsive axon guidance.",
            "Mechanisms that guide axons to their targets in the developing nervous system have been elucidated, but how these pathways affect behavior of the growth cone of the axon during outgrowth remains poorly understood. We previously showed that the guidance cue UNC-6/Netrin and its receptors UNC-40/DCC and UNC-5 inhibit lamellipodial and filopodial growth cone protrusion to mediate repulsion from UNC-6/Netrin in C. elegans. Here we report a new mechanism downstream of UNC-6/Netrin involving flavin monooxygenase redox enzymes (FMOs). We show that FMOs are normally required for axon guidance and to inhibit growth cone protrusion. Furthermore, we show that they are required for the anti-protrusive effects of activated UNC-40 and UNC-5 receptors, and that they can partially compensate for loss of molecules in the pathway, indicating that they act downstream of UNC-6/Netrin signaling. Based on the function of the FMO-containing MICAL molecules in Drosophila and vertebrates, we speculate that the FMOs might directly oxidize actin, leading to filament disassembly and collapse, and/or lead to the phosphorylation of UNC-33/CRMP, which we show also genetically interacts with the FMOs downstream of UNC-6/Netrin. In conclusion, this is the first evidence that FMOs might act downstream of UNC-6/Netrin signaling in growth cone protrusion and axon repulsion."
        ),
            (
                "Spontaneous canine head and neck squamous cell carcinoma (HNSCC) represents an excellent model of human HNSCC but is greatly understudied. To better understand and utilize this valuable resource, we performed a pilot study that represents its first genome-wide characterization by investigating 12 canine HNSCC cases, of which 9 are oral, via high density array comparative genomic hybridization and RNA-seq. The analyses reveal that these canine cancers recapitulate many molecular features of human HNSCC. These include analogous genomic copy number abnormality landscapes and sequence mutation patterns, recurrent alteration of known HNSCC genes and pathways (e. g., cell cycle, PI3K/AKT signaling), and comparably extensive heterogeneity. Amplification or overexpression of protein kinase genes, matrix metalloproteinase genes, and epithelial-mesenchymal transition genes TWIST1 and SNAI1 are also prominent in these canine tumors. This pilot study, along with a rapidly growing body of literature on canine cancer, reemphasizes the potential value of spontaneous canine cancers in HNSCC basic and translational research.",
                "Head and neck squamous cell carcinoma (HNSCC) represents the sixth leading cancer by incidence in humans; thus, developing effective therapeutic interventions is important. Although great advance has been made in our understanding of the biology of HNSCC over the past several decades, translating the research findings into clinical success has been frustratingly slow, and anticancer drug development remains a lengthy and expensive process. A significant challenge is that drug effects in current preclinical cancer models often do not predict clinical results, and there lacks translational models that can bridge the gap between preclinical research and human clinical trials. Here we report a pilot study that represents the first genome-wide characterization of spontaneously occurring HNSCCs in pet dogs. The study reveals a strong dog-human molecular homology at various levels, indicating the likelihood that spontaneous canine HNSCC molecularly represents its human counterpart. If conclusions of this pilot study are validated with a large sample size and more efforts are put into building better resource and infrastructure for canine cancer research, spontaneous canine HNSCCs could effectively serve as a much-needed translational model that bridges the gap between preclinical research and human trials."
            )]
    elif dataset_name == 'Cochrane':
        demonstrations = [
            (
            "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers.",
            "We found two small studies that presented data for 49 participants with arterial leg ulcers (search conducted January 2019). The studies also included participants with other kinds of ulcers, and it is not clear what proportion of participants were diabetic. Neither study described the methods fully, both presented limited results for the arterial ulcer participants, and one study did not provide information on the number of participants with an arterial ulcer in the control group. The follow-up periods (six and eight weeks) were too short to measure healing. Therefore, the data that were available were incomplete and cannot be generalised to the greater population of people who suffer from arterial leg ulcers. One study randomised participants to either 2% ketanserin ointment in polyethylene glycol (PEG) or PEG alone, administered twice a day over eight weeks. This study reported increased wound healing in the ketanserin group, when compared with the control group. It should be noted that ketanserin is not licensed for use in humans in all countries. The second study randomised participants to either topically-applied growth factors isolated from the participant's own blood (concentrated growth factors (CGF)), or standard dressing; both applied weekly for six weeks. This study reported that 66.6% of CGF-treated diabetic arterial ulcers showed more than a 50% decrease in ulcer size, compared to 6.7% of non-healing ulcers treated with standard dressing. Only one study mentioned side effects, and reported that no participant experienced side effects during follow-up (six weeks). Neither of the two studies reported time to ulcer healing, patient satisfaction or quality of life measures. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers. We downgraded the overall certainty of the available evidence to 'very low' and 'low', because the studies reported their methods poorly, there were only two studies and few participants with arterial disease, and because the studies were short and reported few results. This made it impossible to determine whether there was any real difference in the number of ulcers healed between the groups."
            ),
            (
                "We identified one RCT that involved 40 participants, and addressed the timing of surgery for people with recently symptomatic carotid artery stenosis. It compared very early surgery with surgery performed after 14 days of the last symptomatic event. The overall quality of the evidence was very low, due to the small number of participants from only one trial, and missing outcome data. We found no statistically significant difference between the effects of very early or delayed surgery in reducing the combined risk of stroke and death within 30 days of surgery (risk ratio (RR) 3.32; confidence interval (CI) 0.38 to 29.23; very low-quality evidence), or the combined risk of perioperative death and stroke (RR 0.47; CI 0.14 to 1.58; very low-quality evidence). To date, no results are available to confirm the optimal timing for surgery. There is currently no high-quality evidence available to support either very early or delayed cerebral revascularization after a recent ischemic stroke. Hence, further randomized trials to identify which patients should undergo very urgent revascularization are needed. Future studies should stratify participants by age group, sex, grade of ischemia, and degree of stenosis. Currently, there is one ongoing RCT that is examining the timing of cerebral revascularization.",
                "The searches are up-to-date to 26 January 2016. We found only one randomized trial that assessed the effect of the timing of surgery. It included a total of 40 participants, ranging in age from 47 to 84 years. From the limited evidence available, we cannot tell if the timing of surgery is an important factor in determining the outcome for individuals with recent symptoms from carotid artery narrowing. There is not enough evidence on the best time for surgical treatment for people with recent symptoms from carotid artery narrowing. The overall quality of the evidence was very low, due to the small number of participants from only one trial and missing outcome data. Further studies with a larger number of patients are needed."
            ),
        ]
    elif dataset_name in ['XXXX']:
        pass
    else:
        print("No ref demonstrations")
    return demonstrations


def special_change_template(test_dic, config, template):
    sys_tem, train_tem, prompt_tem, demonstration_tem, format_tem, cot_tem = template['sys'], template['train'], \
    template['tes'], template.get('dem', None), template['format'], template['cot']

    for i in range(len(test_dic)):
        
        # retrieval_icl = get_demonstrations(config['DATASET'])
        retrieval_icl = test_dic[i].get('retrieved_texts', [])

        test_dic[i]['retrieved_texts'] = retrieval_icl[:config['TOPK']]
        demonstrations = formulate_dem(test_dic[i]['retrieved_texts'],
                                       demonstration_tem)
        prompt = PromptTemplate.from_template(prompt_tem)
        test_dic[i]['prompt'] = prompt.format(question=test_dic[i]['Q'],
                                              demonstration=demonstrations)
        test_dic[i]['system'] = sys_tem
    return test_dic


def special_loader(data_root, train_dataset, test_dataset, set_config, template):
    sys_tem, train_tem, prompt_tem, demonstration_tem, format_tem, cot_tem = template['sys'], template['train'], \
    template['tes'], template.get('dem', None), template['format'], template['cot']
    train_file_check = data_root + f'/train_dataset_tem_noicl_{set_config["TASK"]}.pkl'
    test_file_check = data_root + f'/test_dataset_tem_noicl_{set_config["TASK"]}.pkl'

    if os.path.exists(train_file_check):
        if set_config['MODEL'] in ['ours']:
            print("Already origin template format!")
            train_dataset = train_dataset
            test_dataset = test_dataset
        else:
            print("Already special template format! XXX")
            train_dataset = load_pickle(train_file_check)
            test_dataset = load_pickle(test_file_check)
        return train_dataset, test_dataset
