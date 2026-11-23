# SPDX-License-Identifier: Apache-2.0
import pytest

from ssr_abtest.render.image_tokens import ImageTokenError, image_input_tokens


def test_patch_examples_from_the_vision_guide():
    # gpt-6-astra, detail high: 65,535px max side, 2,500-patch budget, 1.2x.
    assert image_input_tokens("gpt-6-astra", 1024, 1024, "high") == 1229
    assert image_input_tokens("gpt-6-astra", 2048, 2048, "high") == 3000
    assert image_input_tokens("gpt-6-astra", 4096, 512, "high") == 2458


def test_gpt_5_4_mini_uses_the_same_multiplier_and_auto_follows_high():
    # 1280x800 fold viewport: ceil(1280/32)*ceil(800/32) = 40*25 = 1000 patches.
    assert image_input_tokens("gpt-5.4-mini", 1280, 800, "high") == 1200
    assert image_input_tokens("gpt-5.4-mini-2026-03-17", 1280, 800, "auto") == 1200
    # low allows 6,144 patches, so a 2048 square is not shrunk and costs more than high.
    low = image_input_tokens("gpt-5.4-mini", 2048, 2048, "low")
    high = image_input_tokens("gpt-5.4-mini", 2048, 2048, "high")
    assert high == 3000
    assert low > high


def test_tile_model_low_is_the_base_and_high_scales_to_768():
    assert image_input_tokens("gpt-4o", 1024, 1024, "low") == 85
    # Short side scales to 768. 768/512 tiles are 2 by 2. 85 + 4*170.
    assert image_input_tokens("gpt-4o", 1024, 1024, "high") == 765
    with pytest.raises(ImageTokenError):
        image_input_tokens("gpt-4o", 100, 100, "original")


def test_unknown_model_and_oversized_original_are_errors():
    with pytest.raises(ImageTokenError, match="no published"):
        image_input_tokens("local-model", 100, 100, "high")
    with pytest.raises(ImageTokenError, match="30,000|30000"):
        image_input_tokens("gpt-6-astra", 8000, 8000, "original")
