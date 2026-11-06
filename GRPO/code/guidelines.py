"""Annotation guideline text, examples removed.

Only the explicit *rules* live here. The worked examples from the guideline
document are deliberately excluded to prevent bloat in the steered KV cache.
"""

GENERAL_RULES = """General rules:
1. Always include the article words before the entities - 'the', 'a', 'an'. All markable concepts should be derived from noun phrases.
2. Include special characters.
3. Only the Title and the Methods section are to be annotated."""


ELEMENTS = {
    "Population": {
        "letter": "P",
        "name": "Participant",
        "definition": (
            "The Participant component refers to the descriptions of participants involved in a "
            "medical study, including their recruitment process and the characteristics or "
            "requirements they needed to meet for inclusion, such as age, gender, sample size, "
            "medical diagnoses or conditions, and treatment locations."
        ),
        "rules": """What to include:
1. Mark the longest noun phrases about how the participants were recruited in the medical study and what characteristics or requirements they needed to meet to be included in the study.
2. When subgroup information is inevitably included in the complete noun phrase, include it.

What to exclude:
1. Phrases about grouping should not be considered as Participant.
2. When a location is mentioned together with participants, include them. Otherwise, do not include them.
3. Do not annotate inclusion or exclusion criteria for the participants when the phrases "inclusion criteria" or "exclusion criteria" are explicitly mentioned.
4. Do not include phrases that only mention a condition without participants being mentioned.
5. Exclude prepositional words before the participants.""",
    },

    "Intervention": {
        "letter": "I",
        "name": "Intervention",
        "definition": (
            "The Intervention component represents the proposed treatment or approach being "
            "administered to the participants. Interventions are not limited to medical "
            "treatments; they may also be non-medical approaches such as educational courses "
            "or therapies."
        ),
        "rules": """What to include:
1. Include all the modifiers before the Intervention element when dosage, frequency, and the form of the intervention appear as modifiers.
2. Include an Intervention entity when it is inside a prepositional phrase.
3. If there are multiple distinct entities mentioned within the same context, split them into separate annotations.
4. Annotate continuous entities separately.
5. Include all the general and specific terms of the intervention.
6. Annotate the intervention every time it appears, whether or not it has already been annotated.
7. When one drug is co-administered with another, annotate them separately.

What to exclude:
1. Do not include general terms like 'control group', 'intervention group', or 'a treated group'.
2. When an intervention is mentioned as a group, annotate only the intervention itself; do not include the word 'group'.
3. Exclude dosage, frequency, length, and the form of the intervention when they are located after the Intervention element.
4. Do not include any prepositional phrase following the Intervention element.""",
    },

    "Comparator": {
        "letter": "C",
        "name": "Control",
        "definition": (
            "The Control component refers to the comparison or control treatment used in the "
            "study. The control group serves as a baseline against which the effectiveness of "
            "the intervention is evaluated. It may receive a placebo, a sham procedure, or no "
            "treatment at all."
        ),
        "rules": """What to include:
1. All the treatments or procedures that are given to all the patients in the control group should be annotated as Control.
2. Control entities follow exactly the same rules as Intervention entities. The only difference is that Control is semantically the comparator treatment.
3. Control and Intervention may sometimes be the same treatment. In these cases, annotate them according to semantics: the one given to the study group is Intervention, and the one given to the control group is Control.

What to exclude:
1. Do not include general terms like 'control group' or 'a treated group'.
2. When a control is mentioned as a group, annotate only the treatment itself; do not include the word 'group'.
3. Exclude dosage, frequency, length, and the form of the treatment when they are located after the Control element.
4. Do not include any prepositional phrase following the Control element.""",
    },

    "Outcome": {
        "letter": "O",
        "name": "Outcome",
        "definition": (
            "The Outcome component refers to the measurements or observations used to assess the "
            "effectiveness of the treatment in individuals participating in a trial. Outcomes are "
            "often described by the specific score, scale, measurement tool, or clinical test used "
            "to evaluate the desired result."
        ),
        "rules": """What to include:
1. If a clinical test is the main part of the study and is being used as an intervention, then it has to be annotated as Intervention, not Outcome.
2. Include the outcomes separately when their abbreviations and expansions are both mentioned.
3. Include an outcome that is illustrated both at a summative level and at a detailed level.
4. Include up to one prepositional phrase.
5. If an affirmative sentence is provided, the outcome should be what suggests the result of the study.
6. Include words indicating statistical measurement such as "log", "mean", and "proportion" before the outcome.
7. Include the words "a composite of" when two outcomes are combined.

What to exclude:
1. Do not include how the outcome was measured, unless it is a modifier.
2. Do not include the time span or time point at which the outcome was measured, unless it is a modifier or the measurement depends on time.
3. Do not include how often the outcome was measured, unless it is a modifier.
4. Do not include who received the outcome measure.
5. Do not include information about how an outcome is compared.
6. Do not include the word that indicates the difference, such as "change" or "difference".
7. Do not include statistical methods.
8. Do not annotate outcomes in titles.""",
    },
}

ELEMENT_ORDER = ["Population", "Intervention", "Comparator", "Outcome"]