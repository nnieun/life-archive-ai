"""Explicit, bounded acceptance checks for synthetic QA experiments."""

from statistics import median


def grade_result(case: dict, result) -> dict:
    expected = set(case['relevant_memory_ids'])
    retrieved = set(result.retrieved_memory_ids)
    cited = {citation.memory_id for citation in result.citations}
    groups = case.get('required_terms', [])
    matched = sum(any(term in result.final_answer for term in group) for group in groups)
    refusal = result.validation_result.failure_code == 'insufficient_evidence' and not cited
    if case['kind'] == 'unanswerable':
        correct = refusal and result.error is None
    else:
        correct = (result.validation_result.passed and result.error is None and
                   expected <= cited and cited <= expected and matched == len(groups))
    return {
        'retrieval_recall': len(expected & retrieved)/len(expected) if expected else None,
        'citation_recall': len(expected & cited)/len(expected) if expected else None,
        'citation_precision': len(expected & cited)/len(cited) if cited else None,
        'required_fact_coverage': matched/len(groups) if groups else None,
        'correct_refusal': refusal if case['kind'] == 'unanswerable' else None,
        'acceptance_passed': correct,
    }


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for kind in sorted({row['kind'] for row in rows}):
        for top_k in (5, 3):
            group = [row for row in rows if row['kind'] == kind and row['top_k'] == top_k]
            if not group:
                continue
            recalls = [row['retrieval_recall'] for row in group if row['retrieval_recall'] is not None]
            summary.append({'kind': kind, 'top_k': top_k, 'runs': len(group),
                'median_seconds': round(median(row['seconds'] for row in group), 3),
                'accepted': sum(row['acceptance_passed'] for row in group),
                'mean_retrieval_recall': round(sum(recalls)/len(recalls), 3) if recalls else None})
    return summary


def paired_order(repeat: int, case_index: int) -> tuple[int, int]:
    return (5, 3) if (repeat + case_index) % 2 == 0 else (3, 5)
