from inspection_system.app.training_labels import default_final_class, resolve_learning_class


def test_default_final_class_maps_feedback_to_final_class() -> None:
    assert default_final_class("approve") == "good"
    assert default_final_class("reject") == "reject"
    assert default_final_class("review") is None


def test_resolve_learning_class_prefers_explicit_final_class() -> None:
    assert resolve_learning_class({"feedback": "approve", "final_class": "reject"}) == "reject"


def test_resolve_learning_class_falls_back_to_feedback() -> None:
    assert resolve_learning_class({"feedback": "approve"}) == "good"
    assert resolve_learning_class({"feedback": "reject"}) == "reject"
    assert resolve_learning_class({"feedback": "review"}) is None