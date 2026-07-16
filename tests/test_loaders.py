from loaders import build_question


def test_build_question_default_id():
    problems = build_question("What is 2 + 2?")
    assert problems == [{
        "id": "question",
        "question": "What is 2 + 2?",
        "answer": "",
        "dataset": "custom",
    }]


def test_build_question_custom_id():
    problems = build_question("Prove there are infinitely many primes", problem_id="primes")
    assert len(problems) == 1
    assert problems[0]["id"] == "primes"
    assert problems[0]["answer"] == ""
    assert problems[0]["dataset"] == "custom"
