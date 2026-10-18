"""D03 unit tests: ICBHI filename parsing — valid cases and every rejection path."""

import pytest

from respiralab.data.filenames import FilenameParseError, parse_icbhi_filename


def test_parses_real_example():
    parsed = parse_icbhi_filename("101_1b1_Al_sc_Meditron")
    assert parsed.patient_id == "101"
    assert parsed.recording_index == "1b1"
    assert parsed.chest_site == "Al"
    assert parsed.mode == "sc"
    assert parsed.device == "Meditron"
    assert parsed.record_id == "101_1b1_Al_sc_Meditron"


@pytest.mark.parametrize(
    "stem",
    [
        "103_2b2_Ar_mc_LittC2SE",
        "104_1b1_Ll_sc_Litt3200",
        "226_1p1_Tc_sc_AKGC417L",
    ],
)
def test_parses_all_documented_vocab_values(stem: str):
    parsed = parse_icbhi_filename(stem)
    assert parsed.record_id == stem


def test_rejects_wrong_part_count():
    with pytest.raises(FilenameParseError):
        parse_icbhi_filename("101_1b1_Al_sc")  # missing device


def test_rejects_unknown_chest_site():
    with pytest.raises(FilenameParseError):
        parse_icbhi_filename("101_1b1_ZZ_sc_Meditron")


def test_rejects_unknown_device():
    with pytest.raises(FilenameParseError):
        parse_icbhi_filename("101_1b1_Al_sc_UnknownMic")


def test_rejects_unknown_mode():
    with pytest.raises(FilenameParseError):
        parse_icbhi_filename("101_1b1_Al_xx_Meditron")
