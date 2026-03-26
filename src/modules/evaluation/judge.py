from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for RAG answers.\n"
    "Evaluate generated answers against reference answers and cited context.\n"
    "Return ONLY valid JSON.\n"
)


@dataclass(slots=True)
class JudgeScore:
    accuracy: int
    completeness: int
    relevance: int
    groundedness: int
    feedback: str


class EvaluationJudgeService:
    def __init__(self, *, llm: LLM) -> None:
        self._llm = llm
        self._config = GenerationConfig(temperature=0.0, max_tokens=350)

    async def evaluate(
        self,
        *,
        question: str,
        generated_answer: str,
        reference_answer: str,
        citations: list[dict],
    ) -> JudgeScore:
        judge_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Question:\n"
                    f"{question}\n\n"
                    "Generated Answer:\n"
                    f"{generated_answer}\n\n"
                    "Reference Answer:\n"
                    f"{reference_answer}\n\n"
                    "Citations:\n"
                    f"{json.dumps(citations, ensure_ascii=False)}\n\n"
                    "Score on four dimensions from 1 to 5:\n"
                    "1. Accuracy\n"
                    "2. Completeness\n"
                    "3. Relevance\n"
                    "4. Groundedness (supported by citations)\n\n"
                    "Rules:\n"
                    "- If generated answer is factually wrong, accuracy must be 1.\n"
                    "- 5 means perfect.\n"
                    "- Return JSON only with fields: "
                    "accuracy, completeness, relevance, groundedness, feedback."
                ),
            ),
        ]
        response = await self._llm.generate(
            messages=judge_messages,
            config=self._config,
            tools=None,
        )
        return self._parse_score(response.content)

    @staticmethod
    def _parse_score(content: str) -> JudgeScore:
        payload_text = content.strip()
        if not payload_text:
            raise ValueError("Judge returned empty response.")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", payload_text, re.DOTALL)
            if not match:
                raise ValueError("Judge response is not valid JSON.")
            payload = json.loads(match.group(0))

        accuracy = EvaluationJudgeService._to_score(payload.get("accuracy"), "accuracy")
        completeness = EvaluationJudgeService._to_score(
            payload.get("completeness"), "completeness"
        )
        relevance = EvaluationJudgeService._to_score(payload.get("relevance"), "relevance")
        groundedness = EvaluationJudgeService._to_score(
            payload.get("groundedness"), "groundedness"
        )
        feedback = str(payload.get("feedback", "")).strip()
        if not feedback:
            feedback = "No feedback provided."

        return JudgeScore(
            accuracy=accuracy,
            completeness=completeness,
            relevance=relevance,
            groundedness=groundedness,
            feedback=feedback,
        )

    @staticmethod
    def _to_score(value: object, field_name: str) -> int:
        try:
            score = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Judge field '{field_name}' must be an integer.") from exc
        if score < 1 or score > 5:
            raise ValueError(f"Judge field '{field_name}' must be between 1 and 5.")
        return score

