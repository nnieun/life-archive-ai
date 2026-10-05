"""Evaluation checks must detect missing evidence and unsafe answers."""

from types import SimpleNamespace

from evaluation.qa_topk import grade_result, paired_order, summarize


def result(ids, *, answer='fact', passed=True, failure=None, retrieved=None):
    return SimpleNamespace(final_answer=answer, error=None,
        citations=[SimpleNamespace(memory_id=mid) for mid in ids],
        retrieved_memory_ids=retrieved if retrieved is not None else ids,
        validation_result=SimpleNamespace(passed=passed, failure_code=failure))


def test_verified_partial_answer_is_not_complete():
    case = {'kind': 'multi', 'relevant_memory_ids': ['a','b','c','d'], 'required_terms': [['fact']]}
    grade = grade_result(case, result(['a','b','c']))
    assert grade['retrieval_recall'] == .75
    assert grade['citation_recall'] == .75
    assert not grade['acceptance_passed']
    assert grade_result(case, result(['a','b','c','d']))['acceptance_passed']


def test_wrong_citations_or_failed_verifier_cannot_pass():
    case = {'kind': 'single', 'relevant_memory_ids': ['a'], 'required_terms': [['fact']]}
    assert not grade_result(case, result(['b'], retrieved=['a']))['acceptance_passed']
    assert not grade_result(case, result(['a'], passed=False))['acceptance_passed']
    assert not grade_result(case, result(['a'], answer='something else'))['acceptance_passed']


def test_schema_failure_is_not_a_correct_refusal():
    case = {'kind': 'unanswerable', 'relevant_memory_ids': []}
    assert grade_result(case, result([], passed=False, failure='insufficient_evidence'))['acceptance_passed']
    assert not grade_result(case, result([], passed=False, failure='output_schema_invalid'))['acceptance_passed']
    assert not grade_result(case, result(['a'], answer='invented birthday'))['acceptance_passed']


def test_paired_order_reverses_and_summary_keeps_types_separate():
    assert paired_order(0,0) == (5,3)
    assert paired_order(1,0) == (3,5)
    assert paired_order(0,1) == (3,5)
    rows = [{'kind':'single','top_k':3,'seconds':2,'acceptance_passed':True,'retrieval_recall':1},
            {'kind':'single','top_k':3,'seconds':4,'acceptance_passed':False,'retrieval_recall':0},
            {'kind':'multi','top_k':5,'seconds':9,'acceptance_passed':True,'retrieval_recall':1}]
    summary = summarize(rows)
    single = next(row for row in summary if row['kind']=='single')
    assert single['median_seconds'] == 3
    assert single['accepted'] == 1
    assert single['mean_retrieval_recall'] == .5
