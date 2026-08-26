SYSTEM_MESSAGE = "\n".join(
    (
        "You are a professional NLP data parser.",
        "Follow the provided Task and Output Schema to generate the Output JSON.",
        "Do not generate any introduction or conclusion.",
    )
)

# Explicit domain alias for imports that combine templates from several tasks.
FINETUNING_SYSTEM_MESSAGE = SYSTEM_MESSAGE
