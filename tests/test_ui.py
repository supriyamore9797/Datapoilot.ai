"""Tests for UI helpers and footer attribution."""

from __future__ import annotations
from unittest.mock import patch
import html

from src import config, ui


def test_ui_has_show_footer():
    assert hasattr(ui, "show_footer")
    assert callable(ui.show_footer)


def test_footer_text_contains_exact_authorship():
    text = config.get_app_footer_text()
    assert "DataPilot AI" in text
    assert "Developed by Supriya S More" in text
    assert "2026" in text


def test_show_footer_renders_expected_markup():
    with patch("streamlit.markdown") as mock_markdown:
        ui.show_footer()
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        rendered_html = args[0]
        assert 'class="dp-footer"' in rendered_html
        assert "DataPilot AI" in rendered_html
        assert "Developed by Supriya S More" in rendered_html
        assert kwargs.get("unsafe_allow_html") is True

