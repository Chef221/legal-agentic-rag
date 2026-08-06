"""Tests for the audited UIT DSC 2026 passage-cleaning policy."""

from legal_agentic_rag.competition import UitDsc2026PassageCleaner

_TVPL_NOTICE = (
    ".........Bạn phải đăng nhập hoặc đăng ký Thành Viên TVPL Pro để sử dụng "
    "được đầy đủ các tiện ích gia tăng liên quan đến nội dung TCVN.Mọi chi "
    "tiết xin liên hệ: ĐT: (028) 3930 3279 DĐ: 0906 22 99 66"
)


def test_cleaner_removes_only_audited_markup_and_boilerplate() -> None:
    """Source noise disappears while legal text and unknown brackets survive."""
    passage = (
        "  Điều 1. Không áp dụng <Tên người>.\r\n\r\n"
        "<p>Khoản 2. Mức phạt 10.000.000 đồng.</p>"
        '<img alt="Công thức A = B">'
        "<script>tracking()</script>"
        f"{_TVPL_NOTICE}  Điều 2. Có hiệu lực.  "
    )

    result = UitDsc2026PassageCleaner().clean(passage)

    assert result.text == (
        "Điều 1. Không áp dụng <Tên người>.\n\n"
        "Khoản 2. Mức phạt 10.000.000 đồng.\n"
        "Công thức A = B\nĐiều 2. Có hiệu lực."
    )
    assert "tracking" not in result.text
    assert "TVPL Pro" not in result.text
    assert result.html_markup_removed is True
    assert result.boilerplate_occurrence_count == 1
    assert result.unicode_normalized is True
    assert result.newline_normalized is True
    assert result.modified is True


def test_cleaner_does_not_apply_broad_noise_or_html_guesses() -> None:
    """Similar prose and legal angle-bracket placeholders remain untouched."""
    passage = (
        "Điều 3. TVPL Pro là chuỗi kiểm thử, không phải notice đầy đủ.\n"
        "<Cơ quan có thẩm quyền> phải thực hiện."
    )

    result = UitDsc2026PassageCleaner().clean(passage)

    assert result.text == passage
    assert result.html_markup_removed is False
    assert result.boilerplate_occurrence_count == 0
    assert result.modified is False


def test_cleaner_is_deterministic_and_keeps_blank_context_empty() -> None:
    """Repeated calls have identical output and blank source remains explicit."""
    cleaner = UitDsc2026PassageCleaner()

    assert cleaner.clean("  \r\n ").text == ""
    assert cleaner.clean("Điều 4. Phải thực hiện.") == cleaner.clean(
        "Điều 4. Phải thực hiện."
    )
    assert cleaner.policy_identity() == cleaner.policy_identity()
