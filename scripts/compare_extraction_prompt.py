"""Compare memory-extraction prompts on the same synthetic text (before vs after)."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_core.messages import HumanMessage, SystemMessage
from backend.app.core.config import get_settings
from backend.app.prompts import extraction
from backend.app.services.memory_extraction import build_openai_memory_model

TEXT = ("영수 오빠랑 김치볶음밥을 먹었다 \n오빠가 해줬다 ㅋㅋㅋㅋ 왜냐면 가위바위보에 져서..\n"
        "근데 다음에는 꼭 져야 겠다고 생각했다 너무 맛없어\n영수 오빠는 음식을 못하나보다")
KEYWORDS = ('맛없', '못')
RUNS = 3


def main() -> None:
    settings = get_settings()
    model = build_openai_memory_model(settings.chat_model, base_url=settings.chat_base_url)
    variants = {'before': extraction.MEMORY_EXTRACTION_SYSTEM_PROMPT_BEFORE,
                'after': extraction.MEMORY_EXTRACTION_SYSTEM_PROMPT}
    rows = []
    for name, prompt in variants.items():
        for run in range(RUNS):
            out = model.invoke([SystemMessage(content=prompt), HumanMessage(
                content=extraction.build_memory_extraction_input(
                    transcript_id='tr_demo', segment_id='seg_demo', segment_content=TEXT))])
            parsed = out['parsed'] if isinstance(out, dict) else out
            summaries = [m.summary for m in parsed.memories] if parsed else []
            joined = ' '.join(summaries + [m.emotion or '' for m in parsed.memories] if parsed else [])
            rows.append({'variant': name, 'run': run + 1, 'summaries': summaries,
                         'kept_opinion': any(k in joined for k in KEYWORDS)})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    for name in variants:
        group = [r for r in rows if r['variant'] == name]
        print(name, sum(r['kept_opinion'] for r in group), '/', len(group))


if __name__ == '__main__':
    main()
