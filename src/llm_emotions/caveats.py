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

PHASE03_CAVEAT = (
    "Phase 03 uses one wording per phase-02 situation, not the six paraphrases, "
    "so an effect can belong to that sentence. Pressure is the impossible situation "
    "with the task left solvable. Each task is held fixed and compared with its own "
    "neutral wording, which is the task alone. Accuracy, length, refusal, and marker "
    "counts describe the reply. Projections use the cleaned keyword-free directions "
    "from phase 02 at the layer that phase chose. Changing the prompt is not an "
    "intervention on the residual stream. A link between the anger projection and "
    "accuracy or tone, after the framing label is accounted for, is correlational. "
    "The stricter association also removes the item mean, so task difficulty is not "
    "the whole of that correlation. Steering is phase 04 and is not run here. "
    "Bootstrap intervals are across items and are not corrected for the number of "
    "framings or outcomes."
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
