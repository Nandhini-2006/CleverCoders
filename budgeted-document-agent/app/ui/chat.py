import streamlit as st
import pandas as pd

from app.config import MAX_TOOL_CALLS
from app.tools.document_store import save_document
from app.agent.agent import run_pipeline


def render_chat():
    st.title("📄 Budgeted Document-Answering Agent")
    st.caption("Deterministic 6-Call Tool Budget • Nemotron Query Planning • Grounded Answer Synthesis")
    st.divider()

    # -------------------------------------------------------------
    # 1. PDF Upload & Document Store
    # -------------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        # Check if new file uploaded
        if (
            "active_doc_id" not in st.session_state
            or st.session_state.get("uploaded_filename") != uploaded_file.name
        ):
            doc_id, _ = save_document(uploaded_file)
            st.session_state["active_doc_id"] = doc_id
            st.session_state["uploaded_filename"] = uploaded_file.name
            # Reset state on new file
            st.session_state.pop("agent_state", None)

        st.success("PDF uploaded successfully!")
        st.write(f"**Document:** `{st.session_state['uploaded_filename']}`")
        st.write(f"**Document ID:** `{st.session_state['active_doc_id']}`")

        doc_id = st.session_state["active_doc_id"]
        st.divider()

        # -------------------------------------------------------------
        # 2. Question Input & Agent Trigger
        # -------------------------------------------------------------
        st.subheader("💬 Ask a Question")

        with st.form("question_form"):
            question = st.text_input(
                "Enter your question about the document:",
                placeholder="e.g. When was the term Artificial Intelligence adopted, and at which meeting?",
                key="user_question"
            )
            submitted = st.form_submit_button("🚀 Run Agent", type="primary")

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Executing End-to-End Pipeline (Phases 3–7)..."):
                    try:
                        state = run_pipeline(
                            doc_id=doc_id,
                            question=question.strip(),
                            max_budget=MAX_TOOL_CALLS,
                        )
                        st.session_state["agent_state"] = state
                    except Exception as e:
                        st.error(f"Agent execution encountered an error: {e}")

        # -------------------------------------------------------------
        # 3. Normal Interface: Final Answer & Retrieved Evidence
        # -------------------------------------------------------------
        if "agent_state" in st.session_state:
            state = st.session_state["agent_state"]
            final_ans = state.get("final_answer", {})
            status = final_ans.get("status", state.get("status", "unknown"))
            answer_text = final_ans.get("answer", "No answer generated.")
            sources = final_ans.get("sources", [])
            groundedness = final_ans.get("groundedness", 0.0)

            st.subheader("💡 Final Answer")

            if status == "unresolved_contradiction":
                st.warning(f"**Contradiction Detected:**\n\n{answer_text}")
            elif status == "insufficient_information":
                st.info(f"**Answer:** {answer_text}")
            elif status == "validation_failed":
                st.error(f"**Validation Failed:** {answer_text}")
            else:
                st.success(f"**Answer:** {answer_text}")

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.write(f"**Status:** `{status}`")
            col_s2.write(f"**Groundedness:** `{groundedness:.2f}`")
            sources_str = ", ".join(f"Page {s}" for s in sources) if sources else "None"
            col_s3.write(f"**Citations:** `{sources_str}`")

            st.divider()

            # Retrieved Pages Viewer
            retrieved_pages = state.get("retrieved_pages", [])
            st.subheader("📑 Retrieved Evidence")
            if not retrieved_pages:
                st.info("No pages were retrieved within the budget or matching the query keywords.")
            else:
                p_nums = [p.get("page_number", p.get("page")) for p in retrieved_pages]
                st.write(f"**Pages Retrieved ({len(p_nums)}):** {', '.join(f'Page {num}' for num in p_nums)}")

                for page in retrieved_pages:
                    page_num = page.get("page_number", page.get("page"))
                    page_text = page.get("text", "").strip() or "*[No text found on this page]*"
                    with st.expander(f"📄 Page {page_num}", expanded=False):
                        st.markdown(page_text)

            st.divider()

            # -------------------------------------------------------------
            # 4. Developer / Debug View (Developer Trace)
            # -------------------------------------------------------------
            with st.expander("🛠️ Developer Trace", expanded=True):
                # Pipeline Flow Diagram
                st.markdown(
                    """
```
Question ──▶ Phase 3 Query Plan ──▶ Phase 4 Candidate Pages ──▶ Tool Calls
         ──▶ Retrieved Evidence ──▶ Phase 5 Evidence ──▶ Phase 6 Budget
         ──▶ Phase 7 Validation ──▶ Final Answer
```
                    """
                )
                st.markdown("### 📊 Budget & Entropy Metrics")

                used_calls = state.get("tool_calls_used", 0)
                remaining_budget = state.get("remaining_budget", max(0, MAX_TOOL_CALLS - used_calls))

                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                b_col1.metric("Tool Calls Used", f"{used_calls} / {MAX_TOOL_CALLS}")
                b_col2.metric("Remaining Budget", f"{remaining_budget}")
                b_col3.metric("Entropy", f"{state.get('entropy', 0.0):.4f}")
                b_col4.metric("Confidence", f"{state.get('confidence', 0.0):.2f}")

                st.markdown("---")

                # Phase 3: Query Plan
                st.markdown("#### 🧠 Phase 3: Nemotron Query Plan")
                plan = state.get("query_plan", {})
                if plan.get("error"):
                    st.error(f"⚠️ Planner Error: {plan.get('error')}")
                st.write(f"**Question Type:** `{plan.get('question_type', 'N/A')}`")
                keywords_list = plan.get("keywords", [])
                st.write(
                    f"**Validated Keywords (Max 3):** "
                    + (", ".join(f"`{k}`" for k in keywords_list) if keywords_list else "None")
                )

                st.markdown("---")

                # Phase 4: Candidate Pages
                st.markdown("#### 🔍 Phase 4: Candidate Pages & Probability Distribution")
                candidates = state.get("candidate_pages", [])
                if candidates:
                    table_rows = []
                    retrieved_set = {p.get("page_number", p.get("page")) for p in retrieved_pages}
                    for c in candidates:
                        page_num = c.get("page_number")
                        matched_kws = ", ".join(c.get("matched_keywords", []))
                        k_val = c.get("K(p,q)", c.get("k_score", len(c.get("matched_keywords", []))))
                        h_val = c.get("H(p,q)", c.get("h_score", 0))
                        c_val = c.get("C(p,q)", c.get("c_score", len(c.get("matched_keywords", []))))
                        r_val = c.get("R(p,q)", c.get("score", 0))
                        p_val = c.get("probability", c.get("P(page|query)", 0.0))

                        table_rows.append({
                            "Page Number": page_num,
                            "Matched Keywords": matched_kws,
                            "K(p,q)": k_val,
                            "H(p,q)": h_val,
                            "C(p,q)": c_val,
                            "Relevance R(p,q)": r_val,
                            "Probability P(p|q)": f"{p_val:.4f}",
                            "Retrieved": "✅ Yes" if page_num in retrieved_set else "❌ No",
                        })

                    df = pd.DataFrame(table_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.write("*No candidate pages found.*")

                st.markdown("---")

                # Phase 6: Tool Execution Trace
                st.markdown("#### ⚙️ Phase 6: Tool Execution Trace (Information Gain)")
                tool_trace = state.get("tool_trace", [])
                if tool_trace:
                    trace_rows = []
                    for t in tool_trace:
                        trace_rows.append({
                            "Call #": t.get("call_number"),
                            "Tool": t.get("tool"),
                            "Input": str(t.get("input", {})),
                            "Result": str(t.get("result_summary", "")),
                            "Entropy (Before → After)": f"{t.get('entropy_before', 0.0):.4f} → {t.get('entropy_after', 0.0):.4f}",
                            "Info Gain (ΔH)": f"{t.get('information_gain', 0.0):.4f}",
                            "Budget (Used / Left)": f"{t.get('calls_used')} / {t.get('remaining_budget')}",
                            "Status": t.get("status"),
                        })
                    df_trace = pd.DataFrame(trace_rows)
                    st.dataframe(df_trace, use_container_width=True, hide_index=True)
                else:
                    st.write("*No tool calls executed.*")

                st.markdown("---")

                # Phase 5: Evidence Management & Contradictions
                st.markdown("#### 🧩 Phase 5: Evidence & Contradiction State")
                evidence_data = state.get("evidence", {})
                st.write(f"**Evidence Status:** `{evidence_data.get('status', 'N/A')}`")
                st.write(f"**Confidence Score:** `{evidence_data.get('confidence', 0.0):.2f}`")

                claims = evidence_data.get("claims", [])
                if claims:
                    st.markdown("**Extracted Claims:**")
                    for cl in claims:
                        st.markdown(f"- **Claim:** {cl.get('claim')} *(Pages: {cl.get('supporting_pages', [])})*")

                contras = evidence_data.get("contradictions", [])
                if contras:
                    st.markdown("**Contradictions Detected:**")
                    for c_item in contras:
                        st.markdown(
                            f"- Type: `{c_item.get('type')}`, Status: `{c_item.get('status')}`, "
                            f"Winning Page: `{c_item.get('winning_page')}`"
                        )
                else:
                    st.write("*No contradictions detected.*")

                st.markdown("---")

                # Phase 7: Validation
                st.markdown("#### 🛡️ Phase 7: Answer Validation")
                st.write(f"**Validation Status:** `{status}`")
                st.write(f"**Groundedness Score:** `{groundedness:.2f}` (Threshold: 0.80)")
                st.write(f"**Cited Sources:** `{sources}`")