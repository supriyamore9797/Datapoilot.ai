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


def test_home_page_registration_and_fallback():
    dummy_page = object()
    ui.set_home_page(dummy_page)
    assert ui.get_home_page() is dummy_page

    ui.set_home_page(None)
    fallback = ui.get_home_page()
    assert fallback is not None


def test_require_data_handles_empty_dataset_with_home_link():
    from streamlit.runtime.scriptrunner_utils.exceptions import StopException

    # Reset any state
    ui.set_home_page("dummy_home_page")
    with patch("src.state.has_data", return_value=False), \
         patch("src.state.init_state"), \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.page_link") as mock_page_link, \
         patch("src.ui.show_footer") as mock_footer, \
         patch("streamlit.stop", side_effect=StopException) as mock_stop:

        try:
            ui.require_data()
        except StopException:
            pass

        mock_info.assert_called_once()
        mock_page_link.assert_called_once_with("dummy_home_page", label="Go to Home", icon="🏠")
        mock_footer.assert_called_once()
        mock_stop.assert_called_once()


