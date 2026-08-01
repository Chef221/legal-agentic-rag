"""Raw organizer field names isolated from the reusable core."""

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
CONTEXT_ID_FIELD = "id"
CONTEXT_TITLE_FIELD = "name"
CONTEXT_URL_FIELD = "link"
CONTEXT_PASSAGE_FIELD = "passage"

QUESTION_FIELDS = frozenset({QUESTION_FIELD, ANSWER_FIELD})
CONTEXT_FIELDS = frozenset(
    {
        CONTEXT_ID_FIELD,
        CONTEXT_TITLE_FIELD,
        CONTEXT_URL_FIELD,
        CONTEXT_PASSAGE_FIELD,
    }
)
