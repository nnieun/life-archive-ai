# Retrieval Evaluation Summary

## Scope

- Dataset: `synthetic-life-memories-v1`
- Synthetic memories: 12
- Evaluation queries: 8
- Chunk candidates: 256, 512, 1024 characters, event-aware
- Search candidates: dense similarity, MMR, BM25, hybrid RRF
- Top-K values: 3, 5, 10
- No OpenAI API or personal transcript was used.

## Aggregate Results

| Chunk | Search | K | Recall@K | Citation | Unsupported | Retrieval ms | E2E ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 1024 | bm25 | 3 | 0.500 | 0.250 | 0.750 | 0.5323 | 0.5326 |
| 1024 | bm25 | 5 | 0.750 | 0.250 | 0.750 | 0.5323 | 0.5325 |
| 1024 | bm25 | 10 | 1.000 | 0.250 | 0.750 | 0.5323 | 0.5325 |
| 1024 | dense | 3 | 0.375 | 0.250 | 0.750 | 0.2755 | 0.2758 |
| 1024 | dense | 5 | 0.625 | 0.250 | 0.750 | 0.2755 | 0.2757 |
| 1024 | dense | 10 | 1.000 | 0.250 | 0.750 | 0.2755 | 0.2757 |
| 1024 | hybrid | 3 | 0.375 | 0.250 | 0.750 | 0.8012 | 0.8015 |
| 1024 | hybrid | 5 | 0.625 | 0.250 | 0.750 | 0.8012 | 0.8014 |
| 1024 | hybrid | 10 | 1.000 | 0.250 | 0.750 | 0.8012 | 0.8014 |
| 1024 | mmr | 3 | 0.375 | 0.250 | 0.750 | 0.3575 | 0.3578 |
| 1024 | mmr | 5 | 0.625 | 0.250 | 0.750 | 0.3575 | 0.3578 |
| 1024 | mmr | 10 | 1.000 | 0.250 | 0.750 | 0.3575 | 0.3579 |
| 256 | bm25 | 3 | 0.750 | 0.625 | 0.375 | 0.5853 | 0.5856 |
| 256 | bm25 | 5 | 1.000 | 0.625 | 0.375 | 0.5853 | 0.5856 |
| 256 | bm25 | 10 | 1.000 | 0.625 | 0.375 | 0.5853 | 0.5855 |
| 256 | dense | 3 | 1.000 | 0.750 | 0.250 | 0.3554 | 0.3560 |
| 256 | dense | 5 | 1.000 | 0.750 | 0.250 | 0.3554 | 0.3557 |
| 256 | dense | 10 | 1.000 | 0.750 | 0.250 | 0.3554 | 0.3556 |
| 256 | hybrid | 3 | 0.750 | 0.625 | 0.375 | 0.9109 | 0.9113 |
| 256 | hybrid | 5 | 0.875 | 0.625 | 0.375 | 0.9109 | 0.9111 |
| 256 | hybrid | 10 | 1.000 | 0.625 | 0.375 | 0.9109 | 0.9111 |
| 256 | mmr | 3 | 1.000 | 0.750 | 0.250 | 1.8420 | 1.8424 |
| 256 | mmr | 5 | 1.000 | 0.750 | 0.250 | 1.8420 | 1.8423 |
| 256 | mmr | 10 | 1.000 | 0.750 | 0.250 | 1.8420 | 1.8422 |
| 512 | bm25 | 3 | 0.875 | 0.375 | 0.625 | 0.5573 | 0.5576 |
| 512 | bm25 | 5 | 1.000 | 0.375 | 0.625 | 0.5573 | 0.5575 |
| 512 | bm25 | 10 | 1.000 | 0.375 | 0.625 | 0.5573 | 0.5575 |
| 512 | dense | 3 | 1.000 | 0.500 | 0.500 | 0.3070 | 0.3073 |
| 512 | dense | 5 | 1.000 | 0.500 | 0.500 | 0.3070 | 0.3073 |
| 512 | dense | 10 | 1.000 | 0.500 | 0.500 | 0.3070 | 0.3072 |
| 512 | hybrid | 3 | 1.000 | 0.500 | 0.500 | 0.8614 | 0.8618 |
| 512 | hybrid | 5 | 1.000 | 0.500 | 0.500 | 0.8614 | 0.8616 |
| 512 | hybrid | 10 | 1.000 | 0.500 | 0.500 | 0.8614 | 0.8616 |
| 512 | mmr | 3 | 1.000 | 0.500 | 0.500 | 0.8157 | 0.8161 |
| 512 | mmr | 5 | 1.000 | 0.500 | 0.500 | 0.8157 | 0.8159 |
| 512 | mmr | 10 | 1.000 | 0.500 | 0.500 | 0.8157 | 0.8159 |
| event_aware | bm25 | 3 | 0.875 | 0.750 | 0.250 | 0.5535 | 0.5539 |
| event_aware | bm25 | 5 | 1.000 | 0.750 | 0.250 | 0.5535 | 0.5538 |
| event_aware | bm25 | 10 | 1.000 | 0.750 | 0.250 | 0.5535 | 0.5538 |
| event_aware | dense | 3 | 1.000 | 0.750 | 0.250 | 0.3426 | 0.3429 |
| event_aware | dense | 5 | 1.000 | 0.750 | 0.250 | 0.3426 | 0.3428 |
| event_aware | dense | 10 | 1.000 | 0.750 | 0.250 | 0.3426 | 0.3428 |
| event_aware | hybrid | 3 | 1.000 | 0.750 | 0.250 | 0.8808 | 0.8811 |
| event_aware | hybrid | 5 | 1.000 | 0.750 | 0.250 | 0.8808 | 0.8810 |
| event_aware | hybrid | 10 | 1.000 | 0.750 | 0.250 | 0.8808 | 0.8810 |
| event_aware | mmr | 3 | 0.875 | 0.750 | 0.250 | 4.7538 | 4.7545 |
| event_aware | mmr | 5 | 0.875 | 0.750 | 0.250 | 4.7538 | 4.7540 |
| event_aware | mmr | 10 | 1.000 | 0.750 | 0.250 | 4.7538 | 4.7540 |

## Best Observed Configuration

- Chunk: `event_aware`
- Search: `dense`
- Top-K: `3`
- Recall@K: `1.000`
- Citation correctness: `0.750`
- Unsupported answer rate: `0.250`

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

- Python: `3.13.14`
- Rankings and quality metrics are deterministic for this dataset.
- Latency is measured locally and can vary by machine and background load.
- Failed query rows remain in both CSV files with zero quality scores.

## Limitations

- Dense similarity uses deterministic lexical-semantic aliases instead of an external embedding API.
- Generation evaluation uses a deterministic grounded-answer simulator, not an LLM.
- Results compare MVP settings on a small synthetic corpus and are not a claim about production accuracy.
