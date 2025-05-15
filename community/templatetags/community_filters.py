from django import template

register = template.Library()

@register.filter
def replace(value, arg):
    """
    문자열에서 특정 패턴을 대체하는 필터
    사용 예: {{ string|replace:"old,new" }}
    """
    if ',' not in arg:
        raise template.TemplateSyntaxError("'replace' filter requires an argument in the format 'old,new'")
    old, new = arg.split(',', 1)
    return value.replace(old, new)