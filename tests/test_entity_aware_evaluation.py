from types import SimpleNamespace

from evaluation.entity_aware_qa import score, summarize


def result(retrieved, cited, *, passed=True, failure=None):
    return SimpleNamespace(retrieved_memory_ids=retrieved,
        citations=[SimpleNamespace(memory_id=item) for item in cited],
        validation_result=SimpleNamespace(passed=passed, failure_code=failure))


def test_overview_requires_all_expected_and_no_irrelevant_citations():
    case = {'kind':'overview','expected_ids':['a','b'],'forbidden_ids':['c']}
    assert score(case, result(['a','b'],['a','b']))['acceptance_passed']
    assert not score(case, result(['a','b'],['a']))['acceptance_passed']
    assert not score(case, result(['a','b','c'],['a','b','c']))['acceptance_passed']


def test_unanswerable_requires_explicit_clean_refusal():
    case = {'kind':'unanswerable','expected_ids':[],'forbidden_ids':['a']}
    assert score(case, result([],[],passed=False,failure='insufficient_evidence'))['acceptance_passed']
    assert not score(case, result(['a'],['a']))['acceptance_passed']


def test_summary_compares_variants():
    rows = [{'variant':'baseline','seconds':4,'acceptance_passed':False,'retrieval_recall':.5},
            {'variant':'entity_aware','seconds':5,'acceptance_passed':True,'retrieval_recall':1}]
    output = summarize(rows)
    assert output[0]['accepted'] == 0 and output[1]['accepted'] == 1
