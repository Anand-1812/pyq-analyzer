from datetime import datetime
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from exam_topic_predictor.config import ForecastConfig, MappingConfig
from exam_topic_predictor.pipeline import run_pipeline
from exam_topic_predictor.reporting import write_reports

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def main() -> None:
    st.set_page_config(page_title="Exam Topic Predictor", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 211, 238, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 30%),
                linear-gradient(180deg, #020617 0%, #0f172a 48%, #020617 100%);
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        .hero-shell {
            padding: 2.5rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(3,7,18,0.92));
            border: 1px solid rgba(34, 211, 238, 0.16);
            box-shadow: 0 24px 80px rgba(2, 6, 23, 0.45);
            backdrop-filter: blur(12px);
        }
        .hero-badge {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: #164e63;
            color: #ecfeff;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 1rem 0 0.65rem 0;
            font-size: 3.4rem;
            line-height: 1.02;
            font-weight: 800;
            color: #f8fafc;
        }
        .hero-copy {
            margin: 0;
            max-width: 720px;
            font-size: 1.08rem;
            line-height: 1.7;
            color: #cbd5e1;
        }
        .hero-credit {
            margin-top: 1.1rem;
            font-size: 0.96rem;
            font-weight: 700;
            color: #67e8f9;
        }
        .glass-card {
            height: 100%;
            padding: 1.25rem;
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(71, 85, 105, 0.4);
            box-shadow: 0 16px 50px rgba(2, 6, 23, 0.3);
        }
        .glass-card h3 {
            margin: 0 0 0.6rem 0;
            color: #f8fafc;
            font-size: 1.08rem;
        }
        .glass-card p {
            margin: 0;
            color: #cbd5e1;
            line-height: 1.65;
            font-size: 0.97rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "show_analyzer" not in st.session_state:
        st.session_state.show_analyzer = False

    if not st.session_state.show_analyzer:
        _render_homepage()
        return

    _render_analyzer()


def _render_homepage() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-badge">Exam Intelligence Platform</div>
            <h1 class="hero-title">Predict high-value exam topics before you start revising.</h1>
            <p class="hero-copy">
                Upload a syllabus PDF and previous-year papers to uncover repeated topics, forecast likely
                upcoming areas, inspect topic patterns, and download polished reports in one place.
            </p>
            <div class="hero-credit">Made by Biplob, Dev, Anand</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        <div class="glass-card">
            <h3>Semantic Topic Mapping</h3>
            <p>Questions are matched against syllabus topics using sentence embeddings instead of simple keyword overlap.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="glass-card">
            <h3>Pattern-Aware Forecasting</h3>
            <p>Frequency, year coverage, recency, and recurrence gaps are combined to highlight likely important topics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="glass-card">
            <h3>Ready-to-Download Reports</h3>
            <p>Generate structured JSON and CSV outputs for quick revision planning, analysis sharing, and record keeping.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    action_col1, action_col2 = st.columns([1.2, 1])
    with action_col1:
        if st.button("Open Analyzer", type="primary", use_container_width=True):
            st.session_state.show_analyzer = True
            st.rerun()
    with action_col2:
        st.info("Best results come from one syllabus PDF and paper files named with years like `dbms_2023.pdf`.")


def _render_analyzer() -> None:
    top_bar_col1, top_bar_col2 = st.columns([5, 1.3])
    with top_bar_col1:
        st.title("Exam Topic Predictor")
        st.caption("Upload syllabus and previous-year papers, then get repeated and likely upcoming topics.")
    with top_bar_col2:
        st.write("")
        if st.button("Back Home", use_container_width=True):
            st.session_state.show_analyzer = False
            st.rerun()

    top_n = 10

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

    mapping_config = MappingConfig()
    forecast_config = ForecastConfig()
    manual_topics = None

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
                    forecast_config=forecast_config,
                    top_n=top_n,
                    manual_topics=manual_topics,
                )
        except ValueError as exc:
            st.error(str(exc))
            return
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

    year_options = ["All"] + [str(year) for year in sorted(result.papers_by_year)]
    topic_options = ["All"] + sorted({item.topic for item in result.likely_topics})
    filter_col1, filter_col2 = st.columns(2)
    selected_year = filter_col1.selectbox("Filter by year", options=year_options)
    selected_topic = filter_col2.selectbox("Filter by topic", options=topic_options)

    st.subheader("Likely Upcoming Topics")
    likely_rows = [
        {
            "Topic": item.topic,
            "Score": f"{item.score:.2f}",
            "Confidence": item.confidence,
            "Frequency": item.frequency,
            "Coverage (years)": item.year_coverage,
            "Last appeared": item.last_appeared_year,
            "Pattern": item.pattern,
            "Years": ", ".join(map(str, item.years)),
        }
        for item in result.likely_topics
    ]
    st.dataframe(
        likely_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.TextColumn(help="Prediction confidence bucket"),
            "Pattern": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader("Repeated Topics")
    repeated_rows = [
        {
            "Topic": item.topic,
            "Frequency": item.frequency,
            "Coverage (years)": item.year_coverage,
            "Last appeared": item.last_appeared_year,
            "Pattern": item.pattern,
            "Years": ", ".join(map(str, item.years)),
        }
        for item in result.repeated_topics
    ]
    st.dataframe(repeated_rows, use_container_width=True, hide_index=True)

    st.subheader("Charts")
    st.caption("Frequency overview and year-wise heatmap for syllabus topics.")
    st.plotly_chart(_build_frequency_bar_chart(result.likely_topics), use_container_width=True)
    st.plotly_chart(
        _build_topic_year_heatmap(
            result.question_mappings,
            selected_year=selected_year,
            selected_topic=selected_topic,
        ),
        use_container_width=True,
    )

    with st.expander("Question to Topic Mapping"):
        mapping_rows: list[dict[str, str | int | float]] = []
        for mapping in result.question_mappings:
            if selected_year != "All" and str(mapping.year) != selected_year:
                continue
            if not mapping.matches:
                mapping_rows.append(
                    {
                        "Year": mapping.year,
                        "Paper": mapping.paper_name,
                        "Question ID": mapping.question.question_id,
                        "Question": mapping.question.text,
                        "Topic": "",
                        "Similarity": 0.0,
                    }
                )
                continue
            for match in mapping.matches:
                if selected_topic != "All" and match.topic != selected_topic:
                    continue
                mapping_rows.append(
                    {
                        "Year": mapping.year,
                        "Paper": mapping.paper_name,
                        "Question ID": mapping.question.question_id,
                        "Question": mapping.question.text,
                        "Topic": match.topic,
                        "Similarity": match.similarity,
                    }
                )
        st.dataframe(mapping_rows, use_container_width=True, hide_index=True)

    st.subheader("Top Predicted Questions")
    predicted_questions = [
        {
            "Topic": pattern.topic,
            "Pattern ID": pattern.pattern_id,
            "Occurrences": pattern.question_count,
            "Similarity to Topic": f"{pattern.similarity_to_topic:.2f}",
            "Years": ", ".join(map(str, pattern.years)),
            "Representative Question": pattern.representative_question,
        }
        for pattern in result.question_patterns
        if (selected_topic == "All" or pattern.topic == selected_topic)
        and (selected_year == "All" or any(str(year) == selected_year for year in pattern.years))
    ]
    st.dataframe(
        predicted_questions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Representative Question": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader("Study Recommendations")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    _render_recommendation_card(rec_col1, "HIGH PRIORITY", result.recommendations["high_priority"], "#ffedd5")
    _render_recommendation_card(rec_col2, "MEDIUM PRIORITY", result.recommendations["medium_priority"], "#fef3c7")
    _render_recommendation_card(rec_col3, "LOW PRIORITY", result.recommendations["low_priority"], "#dbeafe")

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
    st.download_button(
        label="Download question_patterns.csv",
        data=reports["pattern_csv"].read_bytes(),
        file_name="question_patterns.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download study_recommendations.json",
        data=reports["recommendation_json"].read_bytes(),
        file_name="study_recommendations.json",
        mime="application/json",
    )
    st.caption(f"Reports also saved at `{output_dir}`.")


def _render_recommendation_card(column, title: str, topics: list, background: str) -> None:
    if topics:
        body = "<br>".join(
            f"<strong>{item.topic}</strong><br>score={item.score:.2f} | {item.pattern}"
            for item in topics
        )
    else:
        body = "No topics in this bucket"

    column.markdown(
        (
            f"<div style='background:{background};padding:1rem;border-radius:0.75rem;"
            "height:260px;color:#111827;display:flex;flex-direction:column;'>"
            f"<div style='font-weight:700;font-size:1rem;margin-bottom:0.75rem;'>{title}</div>"
            f"<div style='line-height:1.6;font-size:0.95rem;overflow:auto;'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_frequency_bar_chart(topics) -> go.Figure:
    ordered_topics = sorted(topics, key=lambda item: item.frequency, reverse=True)
    color_map = {"HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#3b82f6"}
    figure = go.Figure(
        go.Bar(
            x=[item.frequency for item in ordered_topics],
            y=[item.topic for item in ordered_topics],
            orientation="h",
            marker_color=[color_map.get(item.confidence, "#6b7280") for item in ordered_topics],
            customdata=[[item.confidence, item.pattern] for item in ordered_topics],
            hovertemplate=(
                "<b>%{y}</b><br>Frequency=%{x}<br>Confidence=%{customdata[0]}"
                "<br>Pattern=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Topic Frequency",
        xaxis_title="Frequency",
        yaxis_title="Topic",
        yaxis={"categoryorder": "total ascending"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font={"size": 13, "color": "#f9fafb"},
        xaxis={"gridcolor": "#374151", "zerolinecolor": "#374151"},
        yaxis_showgrid=False,
    )
    return figure


def _build_topic_year_heatmap(question_mappings, selected_year: str, selected_topic: str) -> go.Figure:
    topic_year_counts: dict[str, dict[int, int]] = {}
    years: set[int] = set()

    for mapping in question_mappings:
        if selected_year != "All" and str(mapping.year) != selected_year:
            continue
        if not mapping.matches:
            continue
        for match in mapping.matches[:1]:
            if selected_topic != "All" and match.topic != selected_topic:
                continue
            topic_year_counts.setdefault(match.topic, {})
            topic_year_counts[match.topic][mapping.year] = topic_year_counts[match.topic].get(mapping.year, 0) + 1
            years.add(mapping.year)

    top_topics = sorted(
        ((topic, sum(year_counts.values())) for topic, year_counts in topic_year_counts.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    ordered_topics = [topic for topic, _ in top_topics]
    ordered_years = sorted(years)
    z_values = [
        [topic_year_counts.get(topic, {}).get(year, 0) for year in ordered_years]
        for topic in ordered_topics
    ] or [[0]]

    figure = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=ordered_years or ["No data"],
            y=ordered_topics or ["No topics"],
            colorscale="YlOrRd",
            colorbar={"title": "Frequency"},
            hovertemplate="Topic=%{y}<br>Year=%{x}<br>Frequency=%{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Topic-Year Heatmap",
        xaxis_title="Year",
        yaxis_title="Topic",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font={"size": 13, "color": "#f9fafb"},
        xaxis={"gridcolor": "#374151"},
        yaxis={"gridcolor": "#374151"},
    )
    return figure
if __name__ == "__main__":
    main()
