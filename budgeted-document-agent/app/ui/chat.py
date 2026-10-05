import streamlit as st
import pandas as pd

from app.config import MAX_TOOL_CALLS
from app.tools.document_store import save_document
from app.agent.planner import plan_query
from app.agent.retriever import retrieve_pages


def render_chat():
    st.title("📄 Budgeted Document-Answering Agent")
    st.write("Upload a PDF and ask questions about its contents.")
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
            # Reset query/retrieval state on new file
            st.session_state.pop("query_plan", None)
            st.session_state.pop("retrieval_result", None)

        st.success("PDF uploaded successfully!")
        st.write(f"**Document:** {st.session_state['uploaded_filename']}")
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
                placeholder="e.g. What authentication methods are supported?",
                key="user_question"
            )
            submitted = st.form_submit_button("🚀 Run Agent", type="primary")

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Executing Query Planner (Phase 3) & Retrieval (Phase 4)..."):
                    # Phase 3: Query Understanding & Keyword Planning
                    plan = plan_query(question.strip())
                    # Phase 4: Page Selection & Retrieval
                    result = retrieve_pages(doc_id=doc_id, query_plan=plan, max_budget=MAX_TOOL_CALLS)

                    st.session_state["query_plan"] = plan
                    st.session_state["retrieval_result"] = result

        # -------------------------------------------------------------
        # 3. Normal Interface: Retrieved Evidence
        # -------------------------------------------------------------
        if "retrieval_result" in st.session_state and "query_plan" in st.session_state:
            result = st.session_state["retrieval_result"]
            plan = st.session_state["query_plan"]
            retrieved_pages = result.get("retrieved_pages", [])

            st.subheader("📑 Retrieved Evidence")

            if not retrieved_pages:
                st.info("No pages were retrieved within the budget or matching the query keywords.")
            else:
                retrieved_page_numbers = [p["page_number"] for p in retrieved_pages]
                st.write(
                    f"**Pages Retrieved:** {', '.join(f'Page {num}' for num in retrieved_page_numbers)}"
                )

                for page in retrieved_pages:
                    page_num = page["page_number"]
                    page_text = page.get("text", "").strip() or "*[No text found on this page]*"
                    with st.expander(f"📄 Page {page_num}", expanded=True):
                        st.markdown(page_text)

            st.divider()

            # -------------------------------------------------------------
            # 4. Developer / Debug View (Developer Trace)
            # -------------------------------------------------------------
            with st.expander("🛠️ Developer Trace", expanded=True):
                st.markdown("### 📊 Execution & Scoring Metrics")

                # Budget Display
                used_calls = result.get("tool_calls_used", 0)
                remaining_budget = max(0, MAX_TOOL_CALLS - used_calls)

                b_col1, b_col2 = st.columns(2)
                b_col1.metric("Tool Calls Used", f"{used_calls} / {MAX_TOOL_CALLS}")
                b_col2.metric("Remaining Budget", f"{remaining_budget}")

                st.markdown("---")

                # Phase 3: Query Plan
                st.markdown("#### 🧠 Phase 3: Query Plan")
                st.write(f"**Question Type:** `{plan.get('question_type', 'N/A')}`")
                keywords_list = plan.get("keywords", [])
                st.write(f"**Keywords:** {', '.join(f'`{k}`' for k in keywords_list) if keywords_list else 'None'}")

                st.markdown("---")

                # Phase 4: Candidate Pages & Entropy
                st.markdown("#### 🔍 Phase 4: Candidate Pages & Entropy")
                entropy_val = result.get("entropy", 0.0)
                st.write(f"**Entropy:** `{entropy_val}`")

                candidates = result.get("candidate_pages", [])
                if candidates:
                    st.markdown("**Candidate Pages:**")

                    # Build tabular presentation with required metrics
                    table_rows = []
                    for c in candidates:
                        page_num = c.get("page_number")
                        matched_kws = ", ".join(c.get("matched_keywords", []))
                        k_val = c.get("K(p,q)", c.get("k_score", len(c.get("matched_keywords", []))))
                        h_val = c.get("H(p,q)", c.get("h_score", 0))
                        c_val = c.get("C(p,q)", c.get("c_score", len(c.get("matched_keywords", []))))
                        r_val = c.get("R(p,q)", c.get("relevance", c.get("score", 0)))
                        p_val = c.get("P(page|query)", c.get("probability", 0.0))

                        table_rows.append({
                            "Page Number": page_num,
                            "Matched Keywords": matched_kws,
                            "K(p,q)": k_val,
                            "H(p,q)": h_val,
                            "C(p,q)": c_val,
                            "Relevance R(p,q)": r_val,
                            "Probability P(page|query)": f"{p_val:.4f}",
                            "Retrieved": "✅ Yes" if page_num in [p["page_number"] for p in retrieved_pages] else "❌ No",
                        })

                    df = pd.DataFrame(table_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.write("*No candidate pages found.*")