"""Mounted Gradio interface for local baseline inspection."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from legal_agentic_rag.exceptions import LegalAgenticRAGError
from legal_agentic_rag.schemas import LegalQuestionRequest
from legal_agentic_rag.serving.query_service import ServingService


def mount_gradio_ui(
    app: FastAPI,
    *,
    service_provider: Callable[[], ServingService],
    path: str,
    title: str,
) -> FastAPI:
    """Mount a local diagnostic UI without creating another runtime."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "Gradio is required when serving.ui_enabled is true"
        ) from error

    def answer_question(
        question: str,
    ) -> tuple[str, list[dict[str, object]], list[dict[str, object]], str, str]:
        try:
            result = service_provider().answer_result(
                LegalQuestionRequest(question=question)
            )
        except (LegalAgenticRAGError, ValueError):
            return (
                "Không thể xử lý câu hỏi. Vui lòng kiểm tra nội dung nhập.",
                [],
                [],
                "request_failed",
                "",
            )
        citations = [
            citation.model_dump(mode="json")
            for citation in result.response.citations
        ]
        evidence = [
            {
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "document_number": item.document_number,
                "article_number": item.article_number,
                "effect_status": item.effect_status,
                "text": item.text,
            }
            for item in result.state.selected_evidence
        ]
        return (
            result.response.answer,
            citations,
            evidence,
            "\n".join(result.response.warnings),
            result.response.trace_id,
        )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            "# Vietnamese Legal Agentic RAG\n"
            "Câu trả lời chỉ mang tính hỗ trợ tra cứu và phải được đối chiếu "
            "với văn bản pháp luật chính thức."
        )
        question = gr.Textbox(
            label="Câu hỏi pháp luật",
            lines=4,
            placeholder="Nhập câu hỏi bằng tiếng Việt...",
        )
        submit = gr.Button("Tra cứu", variant="primary")
        answer = gr.Textbox(label="Câu trả lời", lines=10)
        citations = gr.JSON(label="Trích dẫn")
        evidence = gr.JSON(label="Evidence đã chọn")
        warnings = gr.Textbox(label="Cảnh báo", lines=3)
        trace_id = gr.Textbox(label="Trace ID", interactive=False)
        outputs = [answer, citations, evidence, warnings, trace_id]
        submit.click(
            answer_question,
            inputs=[question],
            outputs=outputs,
            api_name=False,
        )
        question.submit(
            answer_question,
            inputs=[question],
            outputs=outputs,
            api_name=False,
        )
    return gr.mount_gradio_app(
        app,
        demo,
        path=path,
        footer_links=[],
    )
