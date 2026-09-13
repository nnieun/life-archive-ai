"""User-facing memory-gap reconstruction and confirmation page."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import ApiClientError, MemoryGapCandidateData
from frontend.ui import get_api_client, show_backend_error

GAP_LABELS = {
    "MISSING_LOCATION": "정확한 장소가 비어 있어요",
    "MISSING_PERSON": "함께 있던 사람이 비어 있어요",
    "MISSING_DATE": "언제였는지가 비어 있어요",
    "UNCERTAIN_EVENT": "사건 내용에 확인이 필요해요",
    "CONFLICTING_FACT": "서로 다른 기록을 확인해야 해요",
    "WEAK_PROVENANCE": "뒷받침할 기록이 더 필요해요",
}
STATUS_LABELS = {
    "OPEN": "찾아볼 수 있음",
    "SEARCHING": "관련 기록을 찾는 중",
    "CANDIDATE_FOUND": "후보 확인 필요",
    "WAITING_USER": "단서 입력 필요",
    "RESOLVED": "확인 완료",
    "DISMISSED": "건너뜀",
}
RELATION_LABELS = {
    "SUPPORTS": "직접 뒷받침하는 기록",
    "RELATED": "관련 있는 기록",
    "CONFLICTS": "기존 내용과 다른 기록",
    "UNKNOWN": "관계를 더 확인해야 함",
}


def _candidate_label(candidate: MemoryGapCandidateData) -> str:
    return f"{candidate.value} · 근거 일치도 {candidate.deterministic_score:.0%}"


st.title("기억 빈칸")
st.caption(
    "기록에서 확인이 필요한 부분을 찾고, 저장된 원문과 기억만으로 후보를 제안합니다. "
    "직접 확인하기 전에는 기억을 바꾸지 않습니다."
)

notice = st.session_state.pop("memory_gap_notice", None)
if notice:
    st.success(notice)
question_notice = st.session_state.pop("memory_gap_question", None)
if question_notice:
    st.info(question_notice)

try:
    api_client = get_api_client()
    gap_views = api_client.list_memory_gaps()
except ApiClientError as exception:
    show_backend_error("기억 빈칸 조회", exception)
else:
    if not gap_views:
        st.success("지금 확인할 기억 빈칸이 없습니다.")
    else:
        st.metric("확인이 필요한 기억 빈칸", len(gap_views))

    for view in gap_views:
        gap = view.gap
        title = GAP_LABELS.get(gap.gap_type, "확인이 필요한 기억 빈칸")
        status = STATUS_LABELS.get(gap.status, "확인 필요")
        with st.expander(f"{title} · {status}", expanded=True):
            st.write(gap.clue_text)
            context = []
            if gap.period_start:
                context.append(f"시기: {gap.period_start}")
            if gap.location:
                context.append(f"기억된 장소: {gap.location}")
            if gap.people:
                context.append(f"등장인물: {', '.join(gap.people)}")
            if context:
                st.caption(" · ".join(context))
            if gap.user_clues:
                st.markdown("**내가 추가한 단서**")
                for clue in gap.user_clues:
                    st.write(f"- {clue}")

            proposed = [
                candidate
                for candidate in view.candidates
                if candidate.status == "PROPOSED"
            ]
            if proposed:
                candidate_by_id = {
                    candidate.candidate_id: candidate for candidate in proposed
                }
                selected_id = st.radio(
                    "가능한 답",
                    options=list(candidate_by_id),
                    format_func=lambda candidate_id: _candidate_label(
                        candidate_by_id[candidate_id]
                    ),
                    key=f"gap-candidate-{gap.gap_id}",
                )
                selected = candidate_by_id[selected_id]
                st.write(selected.explanation)
                st.caption(
                    f"근거 관계: {RELATION_LABELS.get(selected.llm_relation, '확인 필요')}"
                    f" · 연결된 내부 근거 {len(selected.supporting_source_ids)}개"
                )
                for source in selected.external_sources:
                    url = source.get("url")
                    title = source.get("title") or source.get("source_domain") or "외부 출처"
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        st.link_button(f"외부 출처: {title}", url)
                confirmed = st.checkbox(
                    "이 후보를 내 기억에 반영하는 데 동의합니다.",
                    key=f"gap-confirm-{gap.gap_id}",
                )
                if st.button(
                    "선택한 후보로 빈칸 채우기",
                    type="primary",
                    key=f"gap-resolve-{gap.gap_id}",
                    disabled=not confirmed,
                ):
                    try:
                        api_client.resolve_memory_gap(
                            gap.gap_id,
                            selected.candidate_id,
                            user_confirmed=True,
                        )
                    except ApiClientError as exception:
                        show_backend_error("기억 빈칸 반영", exception)
                    else:
                        st.session_state["memory_gap_notice"] = (
                            "확인한 후보를 새 기억 수정본으로 반영했습니다."
                        )
                        st.rerun()
            else:
                st.info("아직 확인할 후보가 없습니다.")

            search_col, web_col, skip_col = st.columns(3)
            if search_col.button(
                "저장된 기록에서 후보 찾기",
                key=f"gap-search-{gap.gap_id}",
            ):
                with st.spinner("내 기억과 업로드한 원문에서 관련 기록을 찾는 중입니다."):
                    try:
                        result = api_client.reconstruct_memory_gap(gap.gap_id)
                    except ApiClientError as exception:
                        show_backend_error("기억 빈칸 후보 찾기", exception)
                    else:
                        st.session_state["memory_gap_notice"] = result.message
                        if result.user_question:
                            st.session_state["memory_gap_question"] = (
                                result.user_question
                            )
                        st.rerun()

            web_consent = web_col.checkbox(
                "인터넷 검색에 사용",
                key=f"gap-web-consent-{gap.gap_id}",
                help="기억의 단서를 공개 웹 검색어로 사용합니다.",
            )
            if web_col.button(
                "인터넷에서 단서 찾기",
                key=f"gap-web-search-{gap.gap_id}",
                disabled=not web_consent,
            ):
                with st.spinner("공개 웹에서 기억의 단서를 찾는 중입니다..."):
                    try:
                        result = api_client.reconstruct_memory_gap(
                            gap.gap_id, web_search_consent=True
                        )
                    except ApiClientError as exception:
                        show_backend_error("인터넷 단서 검색", exception)
                    else:
                        st.session_state["memory_gap_notice"] = result.message
                        st.rerun()

            skip_confirmed = skip_col.checkbox(
                "이 빈칸은 건너뛰기",
                key=f"gap-skip-confirm-{gap.gap_id}",
            )
            if skip_col.button(
                "목록에서 닫기",
                key=f"gap-dismiss-{gap.gap_id}",
                disabled=not skip_confirmed,
            ):
                try:
                    api_client.dismiss_memory_gap(gap.gap_id)
                except ApiClientError as exception:
                    show_backend_error("기억 빈칸 닫기", exception)
                else:
                    st.session_state["memory_gap_notice"] = (
                        "선택한 기억 빈칸을 건너뛰었습니다."
                    )
                    st.rerun()

            clue = st.text_input(
                "추가로 기억나는 단서",
                placeholder="예: 극장 옆에 큰 백화점이 있었어요.",
                key=f"gap-clue-{gap.gap_id}",
            )
            if st.button(
                "단서 저장",
                key=f"gap-clue-save-{gap.gap_id}",
                disabled=not clue.strip(),
            ):
                try:
                    api_client.add_memory_gap_clue(gap.gap_id, clue.strip())
                except ApiClientError as exception:
                    show_backend_error("기억 단서 저장", exception)
                else:
                    st.session_state["memory_gap_notice"] = (
                        "단서를 저장했습니다. 다시 후보를 찾아보세요."
                    )
                    st.rerun()
