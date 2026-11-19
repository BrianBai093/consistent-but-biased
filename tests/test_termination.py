from termination import specialist_fingerprint, update_progress
from tests.factories import spine_output


def test_fingerprint_ignores_public_wording_confidence_and_retrieval_request() -> None:
    original = spine_output()
    restated = original.model_copy(
        update={
            "public_message": "Different public wording.",
            "confidence": 0.2,
            "need_retrieval": True,
            "retrieval_query": "A query",
        }
    )
    assert specialist_fingerprint(original) == specialist_fingerprint(restated)


def test_progress_streak_increments_only_for_repeated_semantic_output() -> None:
    output = spine_output()
    seen, streak = update_progress(
        seen_fingerprints=[],
        no_progress_streak=0,
        output=output,
    )
    assert len(seen) == 1
    assert streak == 0
    seen_again, streak_again = update_progress(
        seen_fingerprints=seen,
        no_progress_streak=streak,
        output=output,
    )
    assert seen_again == seen
    assert streak_again == 1
