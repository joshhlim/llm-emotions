"""Statements that should travel with every phase-01 number."""

PROBE_CAVEAT = (
    "Probe accuracy on this keyword-heavy contrastive set is a pipeline check, "
    "not a finding. The sentences name the emotions they are about, so a linear "
    "classifier can succeed without the direction generalising to keyword-free scenes."
)

STEERING_NOTE = (
    "Qualitative samples from one prompt. Fluency can collapse at high strength. "
    "This is not a performance evaluation."
)

PHASE02_CAVEAT = (
    "Cleaned directions have had the average emotion-versus-neutral component "
    "removed, and whatever of that axis remained after the subtraction. A transfer "
    "score means the contrast still ranks keyword-free sentences above their neutrals. "
    "It does not mean the model feels the emotion. Scenario gaps are scenario minus "
    "matched control, after both scores are centred on neutral prompts. The paraphrase "
    "interval is uncertainty across wordings. The random-direction p-value compares "
    "the gap with axes of the same length. Those p-values are not corrected for the "
    "grid. Read the six pre-specified cells first. Reply tone is not measured."
)
