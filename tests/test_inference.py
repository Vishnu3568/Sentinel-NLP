import pytest

from src.inference import predict_spam


def test_spam_message_is_classified_as_spam():
    result = predict_spam(
        "Congratulations! You have won a free prize. Call 08001234567 now."
    )

    assert result["label"] == "spam"
    assert 0.0 <= result["spam_probability"] <= 1.0


def test_normal_message_is_classified_as_ham():
    result = predict_spam(
        "Hey, are we still meeting today at 6?"
    )

    assert result["label"] == "ham"
    assert 0.0 <= result["spam_probability"] <= 1.0


def test_prediction_contains_threshold():
    result = predict_spam("Hello, how are you?")

    assert result["threshold"] == 0.30


def test_non_string_input_raises_type_error():
    with pytest.raises(TypeError):
        predict_spam(123)


def test_empty_message_raises_value_error():
    with pytest.raises(ValueError):
        predict_spam("   ")