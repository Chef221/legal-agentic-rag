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


def test_cleaner_removes_only_balanced_audited_naked_script() -> None:
    """Audited page JavaScript and adjacent UI labels do not enter legal text."""
    passage = """Điều 1. Nội dung hợp lệ.
Bản án liên quan
$(document).ready(function () {
    if (true) {
        guiders.show('Guide');
    }
});
var breadcrumbs = "<ul>Điều hướng</ul>";
Hỏi đáp pháp luật
<huongdan data-id="1">Hướng dẫn giao diện</huongdan>
Điều 2. Nội dung tiếp theo."""

    result = UitDsc2026PassageCleaner().clean(passage)

    assert "$(document)" not in result.text
    assert "guiders." not in result.text
    assert "breadcrumbs" not in result.text
    assert "Bản án liên quan" not in result.text
    assert "Hỏi đáp pháp luật" not in result.text
    assert "<huongdan" not in result.text
    assert "Điều 1. Nội dung hợp lệ." in result.text
    assert "Điều 2. Nội dung tiếp theo." in result.text


def test_cleaner_preserves_unbalanced_script_candidate() -> None:
    """An incomplete script marker cannot make the cleaner drop following text."""
    passage = "$(document).ready(function () {\nĐiều 3. Phải được giữ."

    result = UitDsc2026PassageCleaner().clean(passage)

    assert result.text == passage


def test_cleaner_removes_audited_naked_style_and_popup_functions() -> None:
    """Audited TVPL style/function blocks are removed without touching law text."""
    passage = """Điều 1. Nội dung trước.
.huong-dan-dieu-khoan
{
    color: white;
}
var _scrollTopNoiDungTA = 0;
function ScrollNoiDungTA() {
    if (_scrollTopNoiDungTA > 0) {
        $('html,body').animate({ scrollTop: _scrollTopNoiDungTA }, 1);
    }
}
function ShowPopupVBTraiNgiem() {
    guiders.createGuider({ closeOnEscape: true });
}
Điều 2. Nội dung sau."""

    result = UitDsc2026PassageCleaner().clean(passage)

    assert result.text == "Điều 1. Nội dung trước.\nĐiều 2. Nội dung sau."
