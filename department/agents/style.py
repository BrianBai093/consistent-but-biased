"""Injection of the manipulated variable into the two roles that carry it.

The style lives in three small files rather than in six full role prompts, so
that everything except the prefix is byte-identical across arms. An experiment
whose arms differ in incidental wording cannot attribute its findings to the
style.
"""

from __future__ import annotations

from schemas import DirectorStyle

STYLE_PROMPT_FILENAMES = {
    DirectorStyle.DEMOCRATIC: "style_democratic.md",
    DirectorStyle.AUTOCRATIC: "style_autocratic.md",
    DirectorStyle.AVOIDANT: "style_avoidant.md",
    DirectorStyle.NEUTRAL: "style_neutral.md",
}

#: The deputy faces the style; it does not hold it.
#:
#: The director's rules of conduct are written to the person who rules ("before
#: you rule, name two juniors"), and handing them unchanged to the deputy made
#: the deputy behave like a director -- in the first real run it announced that
#: it had canvassed two junior colleagues and invented one of them. The design
#: document gives the deputy its own switches for each style, which is what
#: these files carry.
DEPUTY_STYLE_PROMPT_FILENAMES = {
    DirectorStyle.DEMOCRATIC: "deputy_democratic.md",
    DirectorStyle.AUTOCRATIC: "deputy_autocratic.md",
    DirectorStyle.AVOIDANT: "deputy_avoidant.md",
    DirectorStyle.NEUTRAL: "deputy_neutral.md",
}


def variant_filename(filename: str, variant: str) -> str:
    """``style_avoidant.md`` for wording ``a``; ``style_avoidant_b.md`` for ``b``.

    The ``b`` files state the same rules in other words. A finding that holds
    under one wording only is a finding about the wording.
    """

    if variant == "a":
        return filename
    stem, _, suffix = filename.rpartition(".")
    return f"{stem}_{variant}.{suffix}"


class StyledAgent:
    """Mixin for a role whose behaviour is governed by the department's style."""

    #: Which set of prefixes this role reads. One style, two vantage points.
    style_prompt_filenames = STYLE_PROMPT_FILENAMES

    def __init__(self, *args, style_variant: str = "a", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.style_variant = style_variant
        self.style_prompts = {
            style: self.load_department_prompt(variant_filename(filename, style_variant))
            for style, filename in self.style_prompt_filenames.items()
        }

    def styled_system_prompt(self, style: DirectorStyle) -> str:
        return f"{self.style_prompts[style]}\n\n{self.system_prompt}"
