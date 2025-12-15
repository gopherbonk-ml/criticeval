from .registry import register
from mathruler.grader import extract_boxed_content


@register(name="nothing")
def nothing_extractor(response: str) -> dict:
    return {}


@register(name="boxed_answer_extractor")
def boxed_answer_extractor(response) -> dict:
    return {
        "answer": extract_boxed_content(response)
    }