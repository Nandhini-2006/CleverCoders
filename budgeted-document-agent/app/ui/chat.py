import streamlit as st

from app.tools.document_store import save_document


def render_chat():

    st.title("📄 Budgeted Document-Answering Agent")

    st.write(
        "Upload a PDF and ask questions about its contents."
    )

    st.divider()

    # PDF uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        # Store only once
        if (
            "active_doc_id" not in st.session_state
            or st.session_state.get("uploaded_filename") != uploaded_file.name
        ):

            # Save PDF
            doc_id, file_path = save_document(uploaded_file)

            # Store temporary session information
            st.session_state["active_doc_id"] = doc_id
            st.session_state["uploaded_filename"] = uploaded_file.name

        st.success("PDF uploaded successfully!")

        st.write(
            f"**Document:** {st.session_state['uploaded_filename']}"
        )

        st.write(
            f"**Document ID:** `{st.session_state['active_doc_id']}`"
        )

        st.info(
            "Document uploaded. Document tools will be connected in Phase 2."
        )