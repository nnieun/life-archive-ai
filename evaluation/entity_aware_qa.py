"""Scoring for entity-aware retrieval without using an LLM as the judge."""

from statistics import median


def score(case: dict, result) -> dict:
    expected = set(case['expected_ids'])
    forbidden = set(case['forbidden_ids'])
    retrieved = set(result.retrieved_memory_ids)
    cited = {citation.memory_id for citation in result.citations}
    refusal = result.validation_result.failure_code == 'insufficient_evidence' and not cited
    if case['kind'] == 'unanswerable':
        accepted = refusal
    else:
        accepted = (result.validation_result.passed and expected <= cited and not (forbidden & cited))
    return {
        'retrieval_recall': len(expected & retrieved) / len(expected) if expected else None,
        'citation_recall': len(expected & cited) / len(expected) if expected else None,
        'irrelevant_citations': sorted(forbidden & cited),
        'correct_refusal': refusal if case['kind'] == 'unanswerable' else None,
        'acceptance_passed': accepted,
    }


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for variant in ('baseline', 'entity_aware'):
        group = [row for row in rows if row['variant'] == variant]
        output.append({'variant': variant, 'runs': len(group),
            'median_seconds': round(median(row['seconds'] for row in group), 3),
            'accepted': sum(row['acceptance_passed'] for row in group),
            'mean_retrieval_recall': round(sum(row['retrieval_recall'] for row in group
                if row['retrieval_recall'] is not None) / sum(row['retrieval_recall'] is not None for row in group), 3)})
    return output
