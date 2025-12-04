from .registry import register


@register(name="base_answer_extractor")
def base_answer_extractor(response):
    return response


@register(name="boxed_answer_extractor")
def boxed_answer_extractor(response):
    return response