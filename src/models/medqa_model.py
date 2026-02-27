# !/usr/bin/env python
# -*-coding:utf-8 -*-

import re
from .base_model import BaseModel
from typing import Optional, List, Tuple

class MedQAModel_MCQ(BaseModel):
    def reward_correct(self, item, answer):
        """Check if the answer is correct for MedQA"""

        def extract_letter_answer(text):
            """Extract letter choice from text using multiple regex patterns."""
            letter_choices = r"(?P<letter>[ABCDEFGHIJ])"

            patterns = [
                (rf"(?i:answer)\s*:\s*{letter_choices}", 100),
                (rf"(?i:the answer is)\s*{letter_choices}", 110),
                (rf"(?i:final answer)\s*(?:is)?\s*:?\s*{letter_choices}", 120),
                (rf"(?i:therefore)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 130),
                (rf"(?i:so)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 140),
                (rf"^\s*{letter_choices}\s*[\.\)\,\:]", 200),
                (rf"\n\s*{letter_choices}\s*[\.\)\,\:]", 210),
                (rf"\({letter_choices}\)", 220),
                (rf"{letter_choices}\s*$", 250),
                (rf"{letter_choices}\s*[\.\,]\s*$", 240),
                (rf"(?i:answer)\s*.*?{letter_choices}", 300),
                (rf"{letter_choices}", 400),
            ]
            # patterns = [
            #     # 新增：匹配 Answer: $C 格式，设置最高优先级10
            #     (rf"(?i:answer)\s*:\s*\$\s*{letter_choices}", 10),
            #     (rf"(?i:answer)\s*:\s*{letter_choices}", 100),
            #     (rf"(?i:the answer is)\s*{letter_choices}", 110),
            #     (rf"(?i:final answer)\s*(?:is)?\s*:?\s*{letter_choices}", 120),
            #     (rf"(?i:therefore)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 130),
            #     (rf"(?i:so)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 140),
            #     (rf"^\s*{letter_choices}\s*[\.\)\,\:]", 200),
            #     (rf"\n\s*{letter_choices}\s*[\.\)\,\:]", 210),
            #     (rf"\({letter_choices}\)", 220),
            #     (rf"{letter_choices}\s*$", 250),
            #     (rf"{letter_choices}\s*[\.\,]\s*$", 240),
            #     (rf"(?i:answer)\s*.*?{letter_choices}", 300),
            #     (rf"{letter_choices}", 400),
            # ]

            best_match = None
            best_priority = float('inf')

            for pattern, priority in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    match = matches[-1]
                    if priority < best_priority:
                        best_match = match
                        best_priority = priority

            if best_match:
                return best_match.group('letter').upper()

            return None

        predicted_letter = extract_letter_answer(answer)
        correct_answer = item.get("A", "").strip().upper()

        if predicted_letter is None:
            return -1.0

        result_score = 1.0 if predicted_letter == correct_answer else -1.0
        return result_score

    def reward_format(self, item, answer):
        # GRPO给予format的奖励，即答案是否符合A,B,C,D,E等选项格式
        def has_letter_format(text):
            letter_choices = r"[ABCDEFGHIJ]"
            patterns = [
                rf"(?i:answer)\s*:\s*{letter_choices}",
                rf"(?i:the answer is)\s*{letter_choices}",
                rf"(?i:final answer)\s*(?:is)?\s*:?\s*{letter_choices}",
                rf"(?i:therefore)\s*,?\s*(?:the answer is)?\s*{letter_choices}",
                rf"(?i:so)\s*,?\s*(?:the answer is)?\s*{letter_choices}",
                rf"^\s*{letter_choices}\s*[\.\)\,\:]",
                rf"\n\s*{letter_choices}\s*[\.\)\,\:]",
                rf"\({letter_choices}\)",
                rf"{letter_choices}\s*$",
                rf"{letter_choices}\s*[\.\,]\s*$",
                rf"(?i:answer)\s*.*?{letter_choices}",
                rf"{letter_choices}",
            ]

            for pattern in patterns:
                if re.search(pattern, text):
                    return True
            return False
        return 1.0 if has_letter_format(answer) else -1.0


    def reward_other_metrics(self, item, answer):
        # 其他奖励指标占位符
        return {'NULL': 1.0}
