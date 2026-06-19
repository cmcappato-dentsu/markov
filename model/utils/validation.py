import re
from model.preprocessing import canon

def compile_exclusion_predicate(exclude_list, exclude_regex):
    ex_list = {canon(x) for x in (exclude_list or [])}
    ex_re = re.compile(exclude_regex, flags=re.I) if exclude_regex else None

    def _excluded(ch: str) -> bool:
        c = canon(ch)
        if c in ex_list:
            return True
        if ex_re and ex_re.search(c):
            return True
        return False
    return _excluded