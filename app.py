from datetime import datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import streamlit as st

from exam_topic_predictor.config import MappingConfig
from exam_topic_predictor.pipeline import run_pipeline
from exam_topic_predictor.reporting import write_reports

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def main() -> None:
    st.set_page_config(page_title="Exam Topic Predictor", layout="wide")
    st.title("Exam Topic Predictor")
    st.caption("Upload syllabus and previous-year papers, then get repeated and likely upcoming topics.")

    with st.sidebar:
        st.subheader("Model Settings")
        top_n = st.slider("Likely topics (Top N)", min_value=3, max_value=20, value=10, step=1)
        min_similarity = st.slider("Minimum topic similarity", min_value=0.0, max_value=0.9, value=0.18, step=0.01)
        top_k_topics = st.slider("Max topics per question", min_value=1, max_value=5, value=3, step=1)
        min_question_chars = st.slider("Minimum question length", min_value=10, max_value=120, value=24, step=2)

    syllabus_pdf = st.file_uploader("Syllabus PDF", type=["pdf"], accept_multiple_files=False)
    paper_pdfs = st.file_uploader("Previous-Year Paper PDFs", type=["pdf"], accept_multiple_files=True)
    st.info("Keep year in paper file names, for example `dbms_2023.pdf`.")

    run_clicked = st.button("Run Analysis", type="primary", disabled=not syllabus_pdf or not paper_pdfs)
    if not run_clicked:
        return

    invalid_names = [paper.name for paper in paper_pdfs if not YEAR_PATTERN.search(paper.name)]
    if invalid_names:
        st.error(
            "Year not found in file name for: "
            + ", ".join(invalid_names)
            + ". Please include year like `subject_2023.pdf`."
        )
        return

    mapping_config = MappingConfig(
        min_similarity=min_similarity,
        top_k_topics=top_k_topics,
        min_question_characters=min_question_chars,
    )

    with st.spinner("Running analysis..."):
        try:
            with TemporaryDirectory(prefix="exam_topic_predictor_") as temp_dir:
                temp_path = Path(temp_dir)
                syllabus_path = temp_path / syllabus_pdf.name
                syllabus_path.write_bytes(syllabus_pdf.getbuffer())

                paper_paths: list[Path] = []
                for paper in paper_pdfs:
                    paper_path = temp_path / paper.name
                    paper_path.write_bytes(paper.getbuffer())
                    paper_paths.append(paper_path)

                result = run_pipeline(
                    syllabus_pdf=syllabus_path,
                    paper_pdfs=paper_paths,
                    mapping_config=mapping_config,
                    top_n=top_n,
                )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data/outputs") / f"run_{run_id}"
    reports = write_reports(result, output_dir=output_dir)

    st.success("Analysis completed.")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Syllabus Topics", len(result.syllabus_topics))
    metric_col2.metric("Questions Mapped", result.question_count)
    metric_col3.metric("Papers Processed", result.paper_count)

    st.subheader("Likely Upcoming Topics")
    likely_rows = [
        {
            "Topic": item.topic,
            "Score": item.score,
            "Frequency": item.frequency,
            "Coverage (years)": item.year_coverage,
            "Years": ", ".join(map(str, item.years)),
        }
        for item in result.likely_topics
    ]
    st.dataframe(likely_rows, use_container_width=True, hide_index=True)

    st.subheader("Repeated Topics")
    repeated_rows = [
        {
            "Topic": item.topic,
            "Frequency": item.frequency,
            "Coverage (years)": item.year_coverage,
            "Years": ", ".join(map(str, item.years)),
        }
        for item in result.repeated_topics
    ]
    st.dataframe(repeated_rows, use_container_width=True, hide_index=True)

    with st.expander("Question to Topic Mapping"):
        mapping_rows: list[dict[str, str | int | float]] = []
        for mapping in result.question_mappings:
            if not mapping.matches:
                mapping_rows.append(
                    {
                        "Year": mapping.year,
                        "Question ID": mapping.question.question_id,
                        "Question": mapping.question.text,
                        "Topic": "",
                        "Similarity": 0.0,
                    }
                )
                continue
            for match in mapping.matches:
                mapping_rows.append(
                    {
                        "Year": mapping.year,
                        "Question ID": mapping.question.question_id,
                        "Question": mapping.question.text,
                        "Topic": match.topic,
                        "Similarity": match.similarity,
                    }
                )
        st.dataframe(mapping_rows, use_container_width=True, hide_index=True)

    st.subheader("Download Reports")
    st.download_button(
        label="Download analysis_summary.json",
        data=reports["summary_json"].read_bytes(),
        file_name="analysis_summary.json",
        mime="application/json",
    )
    st.download_button(
        label="Download question_topic_mapping.csv",
        data=reports["mapping_csv"].read_bytes(),
        file_name="question_topic_mapping.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download topic_predictions.csv",
        data=reports["prediction_csv"].read_bytes(),
        file_name="topic_predictions.csv",
        mime="text/csv",
    )
    st.caption(f"Reports also saved at `{output_dir}`.")


if __name__ == "__main__":
    main()
