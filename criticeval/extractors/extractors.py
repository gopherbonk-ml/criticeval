from .registry import register
from mathruler.grader import extract_boxed_content

@register(name="base_answer_extractor")
def base_answer_extractor(response):
    return response


@register(name="boxed_answer_extractor")
def boxed_answer_extractor(response):
    return extract_boxed_content(response)