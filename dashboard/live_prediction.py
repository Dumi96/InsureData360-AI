
"""Live claim anomaly screening UI for InsureData360-AI."""

import html
import json
import os
from datetime import datetime
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_BASE_URL = os.getenv(
    "INSUREDATA360_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
API_URL = f"{API_BASE_URL}/anomalies"

# These labels must match the categories supported by the trained model.
PRODUCT_OPTIONS = [
    "Motor",
    "Property",
    "Marine",
    "Fire",
    "Engineering",
    "Travel",
    "Health",
]

CUSTOMER_TYPE_OPTIONS = [
    "Individual",
    "Corporate",
]


def _add_prediction_styles() -> None:
    """Apply accessible styling to the live prediction interface."""

    st.markdown(
        """
        <style>
        .prediction-intro {
            padding: 22px;
            border: 1px solid #B8D4F5;
            border-radius: 16px;
            background: linear-gradient(120deg, #EAF3FF, #FFFFFF);
            margin: 8px 0 20px;
        }
        .prediction-intro h3 {
            color: #102A43 !important;
            margin: 0 0 8px;
            font-size: 21px;
        }
        .prediction-intro p {
            color: #334155 !important;
            line-height: 1.65;
            margin: 0;
            font-size: 14px;
        }
        .prediction-scan-track {
            height: 5px;
            width: 100%;
            overflow: hidden;
            border-radius: 99px;
            background: #DBEAFE;
            margin-top: 16px;
        }
        .prediction-scan-track span {
            display: block;
            height: 100%;
            width: 32%;
            border-radius: 99px;
            background: linear-gradient(90deg, #2563EB, #06B6D4);
            animation: scanSweep 3.4s ease-in-out infinite;
        }
        .prediction-live-chip {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            color: #1D4ED8 !important;
            background: #DBEAFE;
            border: 1px solid #BFDBFE;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .6px;
            margin-bottom: 10px;
        }
        .prediction-live-chip span {
            width: 7px; height: 7px; border-radius: 50%;
            background: #2563EB; animation: liveBlink 1.4s infinite;
        }
        .prediction-step {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 14px;
            min-height: 105px;
            margin-bottom: 12px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, .05);
        }
        .prediction-step-number {
            display: inline-flex;
            width: 28px;
            height: 28px;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: #DBEAFE;
            color: #1D4ED8 !important;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .prediction-step-title {
            color: #102A43 !important;
            font-weight: 750;
            font-size: 13px;
            margin-bottom: 4px;
        }
        .prediction-step-description {
            color: #475569 !important;
            font-size: 12px;
            line-height: 1.5;
        }
        .prediction-form-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, .04);
        }
        .prediction-section-heading {
            color: #102A43 !important;
            font-size: 20px;
            font-weight: 800;
            margin: 8px 0 6px;
        }
        .prediction-section-note {
            color: #475569 !important;
            font-size: 13px;
            line-height: 1.6;
        }

        /* Labels: prevent light text from disappearing on white backgrounds. */
        .stApp [data-testid="stForm"] [data-testid="stWidgetLabel"],
        .stApp [data-testid="stForm"] [data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stForm"] label,
        .stApp [data-testid="stForm"] label *,
        .stApp [data-testid="stForm"] [data-testid="stCaptionContainer"],
        .stApp [data-testid="stForm"] [data-testid="stCaptionContainer"] * {
            color: #102A43 !important;
            opacity: 1 !important;
        }
        .stApp [data-testid="stForm"] [data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stForm"] label {
            font-size: 14px !important;
            font-weight: 750 !important;
        }

        /* Number inputs */
        .stApp [data-testid="stForm"] input {
            color: #102A43 !important;
            -webkit-text-fill-color: #102A43 !important;
            background-color: #F1F5F9 !important;
            border: 1px solid #94A3B8 !important;
            border-radius: 9px !important;
            opacity: 1 !important;
        }
        .stApp [data-testid="stForm"] input:focus {
            border: 2px solid #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, .12) !important;
        }

        /* Dropdowns and selected values */
        .stApp [data-testid="stForm"] [data-baseweb="select"] > div {
            background: #F1F5F9 !important;
            border: 1px solid #94A3B8 !important;
            border-radius: 9px !important;
            min-height: 42px !important;
        }
        .stApp [data-testid="stForm"] [data-baseweb="select"] *,
        .stApp [data-testid="stForm"] [data-baseweb="select"] span {
            color: #102A43 !important;
            -webkit-text-fill-color: #102A43 !important;
            opacity: 1 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"],
        [role="option"] {
            background-color: #FFFFFF !important;
            color: #102A43 !important;
        }
        [role="option"] * {
            color: #102A43 !important;
        }

        /* Primary action button */
        .stApp [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #1D4ED8, #2563EB) !important;
            color: #FFFFFF !important;
            border: 1px solid #1D4ED8 !important;
            border-radius: 10px !important;
            font-weight: 800 !important;
            min-height: 46px !important;
        }
        .stApp [data-testid="stFormSubmitButton"] button *,
        .stApp [data-testid="stFormSubmitButton"] button p {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        .stApp [data-testid="stFormSubmitButton"] button:hover {
            box-shadow: 0 5px 16px rgba(29, 78, 216, .24) !important;
        }

        /* Prediction result cards */
        .prediction-result {
            padding: 23px;
            border-radius: 16px;
            margin: 16px 0;
            animation: predictionReveal .55s ease-out both;
            box-shadow: 0 7px 22px rgba(15, 23, 42, .08);
        }
        .prediction-result.review {
            background: linear-gradient(135deg, #FFF7ED, #FFFFFF);
            border: 1px solid #FDBA74;
        }
        .prediction-result.normal {
            background: linear-gradient(135deg, #ECFDF5, #FFFFFF);
            border: 1px solid #86EFAC;
        }
        .prediction-result h3 {
            color: #102A43 !important;
            margin: 8px 0;
            font-size: 24px;
        }
        .prediction-result p {
            color: #334155 !important;
            line-height: 1.65;
            margin-bottom: 0;
        }
        .prediction-status-pill {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: .3px;
        }
        .prediction-status-pill.review {
            color: #9A3412 !important;
            background: #FFEDD5;
        }
        .prediction-status-pill.normal {
            color: #166534 !important;
            background: #DCFCE7;
        }
        .prediction-result-label {
            color: #475569 !important;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .8px;
            font-weight: 750;
        }
        .prediction-result-value {
            color: #102A43 !important;
            font-size: 25px;
            font-weight: 800;
            margin-top: 6px;
            overflow-wrap: anywhere;
        }
        .prediction-next-action {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-left: 4px solid #1D4ED8;
            border-radius: 10px;
            padding: 15px;
            margin: 12px 0;
            color: #334155 !important;
            line-height: 1.6;
        }
        .prediction-next-action strong {
            color: #102A43 !important;
        }
        .prediction-disclaimer {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 12px 14px;
            margin-top: 16px;
            color: #475569 !important;
            font-size: 12px;
            line-height: 1.6;
        }
        .prediction-disclaimer strong {
            color: #102A43 !important;
        }

        @keyframes predictionReveal {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes scanSweep {
            0% { transform: translateX(-110%); }
            55% { transform: translateX(230%); }
            100% { transform: translateX(330%); }
        }
        @keyframes liveBlink {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(37,99,235,.25); }
            50% { opacity: .45; box-shadow: 0 0 0 5px rgba(37,99,235,0); }
        }
        @media (prefers-reduced-motion: reduce) {
            .prediction-result { animation: none !important; }
        }
        @media (max-width: 700px) {
            .prediction-intro { padding: 16px; }
            .prediction-result { padding: 17px; }
            .prediction-result h3 { font-size: 21px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_result(result: dict) -> None:
    """Display API output with a clear business interpretation."""

    if not isinstance(result, dict):
        st.error("The API response must be a JSON object.")
        st.caption("Check the response schema returned by the FastAPI service.")
        return

    required_fields = {
        "anomaly_label",
        "anomaly_score",
        "review_recommended",
        "interpretation",
    }
    missing_fields = required_fields.difference(result.keys())

    if missing_fields:
        st.error(
            "The API response is missing expected fields: "
            + ", ".join(sorted(missing_fields))
        )
        st.caption("Check the response schema in src/api/main.py.")
        return

    label = str(result["anomaly_label"])
    interpretation = str(result["interpretation"])
    review_recommended = result["review_recommended"]

    if not isinstance(review_recommended, bool):
        st.error("The API returned an invalid review_recommended value.")
        return

    try:
        anomaly_score = float(result["anomaly_score"])
    except (ValueError, TypeError):
        st.error("The API returned an invalid anomaly score.")
        return

    if review_recommended:
        result_class = "review"
        status_text = "MANUAL REVIEW RECOMMENDED"
        action_text = (
            "Review the claim details and supporting documents. Compare "
            "the claim with applicable policy terms and business rules "
            "before making a decision."
        )
    else:
        result_class = "normal"
        status_text = "NO REVIEW FLAG FROM THIS MODEL"
        action_text = (
            "Continue the normal claims assessment process. The absence "
            "of an anomaly flag does not guarantee that a claim is valid "
            "or that no further checks are needed."
        )

    st.markdown("### Screening result")
    generated_at = result.get("generated_at") or result.get("timestamp")
    if generated_at:
        st.caption(f"Result timestamp supplied by API: {html.escape(str(generated_at))}")
    else:
        st.caption(
            "Response received by dashboard at "
            + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            + ". This is the receipt time, not a server-side model timestamp."
        )
    st.markdown(
        f"""
        <div class="prediction-result {result_class}">
            <span class="prediction-status-pill {result_class}">
                {status_text}
            </span>
            <h3>{html.escape(label)}</h3>
            <p>{html.escape(interpretation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_col, review_col = st.columns(2)

    with score_col:
        st.markdown(
            '<div class="prediction-form-card">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="prediction-result-label">Raw anomaly score</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="prediction-result-value">{anomaly_score:.6f}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Model output only. Its scale and direction depend on the "
            "underlying algorithm; it is not a fraud probability."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with review_col:
        st.markdown(
            '<div class="prediction-form-card">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="prediction-result-label">Review recommendation</div>',
            unsafe_allow_html=True,
        )
        review_text = "Yes — review" if review_recommended else "No flag"
        st.markdown(
            '<div class="prediction-result-value">'
            f'{html.escape(review_text)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Recommendation generated by the current API.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Render explanatory fields only when the API actually supplies them.
    reasons = (
        result.get("reasons")
        or result.get("contributing_factors")
        or result.get("explanations")
    )
    if reasons:
        st.markdown("#### Model-provided reasons / contributing factors")
        if isinstance(reasons, (list, tuple)):
            for reason in reasons:
                st.markdown(f"- {html.escape(str(reason))}")
        elif isinstance(reasons, dict):
            st.json(reasons)
        else:
            st.write(str(reasons))

    st.markdown(
        f"""
        <div class="prediction-next-action">
            <strong>Suggested next step</strong><br>
            {action_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="prediction-disclaimer">
            <strong>Important:</strong> This is an ML-based anomaly-screening
            result, not a fraud determination. It does not approve or reject
            claims, calculate a payout, or replace an authorised claims
            assessor. An anomaly score is not a fraud probability unless the
            model has been specifically designed and validated for that use.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_prediction() -> None:
    """Render the live claim-screening form and prediction results."""

    _add_prediction_styles()

    st.markdown("## Live Claim Risk Screening")
    st.markdown(
        """
        <div class="prediction-intro">
            <div class="prediction-live-chip"><span></span> LIVE MODEL INTERFACE</div>
            <h3>Screen a claim for unusual patterns</h3>
            <p>
                Enter claim and policy information below. The trained model
                analyses the submitted details and returns a screening result
                to help identify cases that may need additional review.
                Pre-filled values are demonstrations; the model determines
                the actual result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            <div class="prediction-step">
                <div class="prediction-step-number">1</div>
                <div class="prediction-step-title">Enter details</div>
                <div class="prediction-step-description">
                    Provide claim, premium and customer information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="prediction-step">
                <div class="prediction-step-number">2</div>
                <div class="prediction-step-title">Analyse</div>
                <div class="prediction-step-description">
                    FastAPI sends the details through the model pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="prediction-step">
                <div class="prediction-step-number">3</div>
                <div class="prediction-step-title">Review the result</div>
                <div class="prediction-step-description">
                    Read the model flag, score and suggested next action.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    health_col, health_note_col = st.columns([1, 2])
    with health_col:
        check_api = st.button("🔌 Check API connection", key="check_screening_api")
    with health_note_col:
        st.caption("Tests the FastAPI health endpoint only when you press the button.")
    if check_api:
        health_url = API_URL.rsplit("/", 1)[0] + "/health"
        try:
            with urlopen(health_url, timeout=5) as health_response:
                health_body = health_response.read().decode("utf-8", errors="replace")
                if 200 <= health_response.status < 300:
                    st.success(f"API responded successfully ({health_response.status}).")
                    if health_body:
                        with st.expander("Health response details"):
                            st.code(health_body)
                else:
                    st.warning(f"API responded with status {health_response.status}.")
        except (HTTPError, URLError, TimeoutError, SocketTimeout) as exc:
            st.error(f"Could not confirm API health: {exc}")

    with st.expander("Field guide: what should I enter?", expanded=False):
        st.markdown(
            """
            - **Claim amount (LKR):** Amount being claimed. Example: `150000`.
            - **Annual premium (LKR):** Annual policy premium. Example: `25000`.
            - **Customer age:** Age in completed years, from `0` to `120`.
            - **Insurance product:** A category supported by the trained model.
            - **Customer type:** A category matching the model's training data.

            These are demonstration examples only. They are not guaranteed
            low-risk inputs and do not guarantee a normal model result.
            """
        )

    st.markdown(
        '<div class="prediction-section-heading">Claim and policy details</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="prediction-section-note">All monetary values are in '
        'Sri Lankan rupees (LKR). No currency conversion is performed.</div>',
        unsafe_allow_html=True,
    )

    with st.form("insuredata360_live_claim_form"):
        left, right = st.columns(2)

        with left:
            claim_amount = st.number_input(
                "Claim amount (LKR)",
                min_value=0.0,
                max_value=1_000_000_000_000.0,
                value=150_000.0,
                step=10_000.0,
                help="Demo value: LKR 150,000. Enter the claimed amount.",
                format="%.2f",
            )
            annual_premium = st.number_input(
                "Annual premium (LKR)",
                min_value=0.0,
                max_value=1_000_000_000_000.0,
                value=25_000.0,
                step=1_000.0,
                help="Demo value: LKR 25,000. Enter the annual premium.",
                format="%.2f",
            )
            age = st.number_input(
                "Customer age (years)",
                min_value=0,
                max_value=120,
                value=35,
                step=1,
                help="Demo value: 35 years. Enter completed years.",
            )

        with right:
            product = st.selectbox(
                "Insurance product",
                options=PRODUCT_OPTIONS,
                index=0,
                help=(
                    "Select a product category supported by the trained "
                    "model. Labels must match training data."
                ),
            )
            customer_type = st.selectbox(
                "Customer type",
                options=CUSTOMER_TYPE_OPTIONS,
                index=0,
                help=(
                    "Select the customer category. Labels must match those "
                    "used by the model's training pipeline."
                ),
            )
            st.info(
                "**Demonstration inputs**\n\n"
                "These values make testing easier. They do not represent a "
                "guaranteed no-risk or low-risk claim."
            )

        submitted = st.form_submit_button(
            "Analyse claim",
            type="primary",
            use_container_width=True,
        )

    retrying = False
    if not submitted:
        if st.session_state.get("last_failed_screening_payload") and st.button(
            "Retry last screening request", key="retry_last_screening"
        ):
            payload = dict(st.session_state["last_failed_screening_payload"])
            retrying = True
        else:
            st.caption(
                "Review the fields above, then select **Analyse claim** to run "
                "the live screening request."
            )
            return
    else:
        payload = {}

    if not retrying:
        if claim_amount <= 0:
            st.error("Claim amount must be greater than zero.")
            return

        if annual_premium <= 0:
            st.error("Annual premium must be greater than zero.")
            return

        payload = {
            "claim_amount": float(claim_amount),
            "annual_premium": float(annual_premium),
            "age": int(age),
            "product": product,
            "customer_type": customer_type,
        }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with st.status("Analysing claim...", expanded=True) as status:
            st.write("Step 1 of 3 — Preparing the submitted claim details.")
            st.write("Step 2 of 3 — Sending the request to the FastAPI model.")

            with urlopen(request, timeout=30) as response:
                response_body = response.read().decode("utf-8")
                result = json.loads(response_body)
                if not isinstance(result, dict):
                    raise ValueError("Expected the API to return a JSON object.")

            st.write("Step 3 of 3 — Checking the model response.")
            status.update(
                label="Claim screening completed",
                state="complete",
                expanded=False,
            )

        st.session_state.pop("last_failed_screening_payload", None)
        st.session_state["last_screening_result"] = result
        st.session_state["last_screening_payload"] = dict(payload)
        st.session_state["last_screening_at"] = datetime.now().isoformat(timespec="seconds")
        _show_result(result)
        export_col, detail_col = st.columns([1, 2])
        with export_col:
            st.download_button(
                "⬇️ Export screening result (JSON)",
                data=json.dumps(
                    {
                        "screened_at": st.session_state["last_screening_at"],
                        "input": payload,
                        "result": result,
                    },
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8"),
                file_name=f"claim_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_latest_screening_result",
            )
        with detail_col:
            st.caption(
                "Export includes the submitted fields and the API response. "
                "Treat it as operational review material and follow your organisation's data policy."
            )

    except HTTPError as exc:
        st.session_state["last_failed_screening_payload"] = payload
        details = exc.read().decode("utf-8", errors="replace")
        st.error(f"The API returned HTTP {exc.code}.")
        st.caption(
            "Check the FastAPI terminal and confirm the submitted fields "
            "match the API schema."
        )
        if details:
            with st.expander("Technical error details"):
                st.code(details)

    except URLError as exc:
        st.session_state["last_failed_screening_payload"] = payload
        st.error("Could not connect to the FastAPI service.")
        st.markdown(
            """
            **Check the following:**
            1. Start the API server in a separate CMD window.
            2. Open `http://127.0.0.1:8000/health` in your browser.
            3. Retry the prediction after the API is running.
            """
        )
        st.caption(f"Connection detail: {exc.reason}")

    except (TimeoutError, SocketTimeout):
        st.session_state["last_failed_screening_payload"] = payload
        st.error(
            "The prediction request timed out. Check the API server "
            "and try again."
        )

    except json.JSONDecodeError:
        st.session_state["last_failed_screening_payload"] = payload
        st.error("The API response was not valid JSON.")

    except (KeyError, TypeError, ValueError) as exc:
        st.session_state["last_failed_screening_payload"] = payload
        st.error(f"The API response could not be processed: {exc}")
