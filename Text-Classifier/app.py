"""Streamlit frontend for Text Classifier.

Provides a clean, professional, and accessible user interface for text
classification using OpenRouter models across Sentiment, Topic, Urgency,
and custom Zero-Shot categories.
"""

import streamlit as st

from openrouter_client import (
    DEFAULT_MODEL,
    POPULAR_MODELS,
    AuthenticationError,
    ModelUnavailableError,
    NetworkError,
    OpenRouterError,
    RateLimitError,
    get_api_key,
    is_api_key_configured,
)
from text_classifier import (
    ALLOWED_CATEGORIES,
    TASK_DESCRIPTIONS,
    TASK_SENTIMENT,
    TASK_TOPIC,
    TASK_URGENCY,
    TASK_ZERO_SHOT,
    ClassificationResult,
    ClassificationValidationError,
    classify_text,
    compare_all_tasks,
)

# Page configuration
st.set_page_config(
    page_title="Text Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern AI productivity-tool aesthetic (no emojis)
st.markdown(
    """
    <style>
    .main-header {
        margin-bottom: 0.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .result-badge {
        display: inline-block;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 0.95rem;
        color: #1e293b;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .char-counter {
        font-size: 0.82rem;
        color: #64748b;
        text-align: right;
        margin-top: -0.75rem;
        margin-bottom: 0.5rem;
    }
    .status-badge-ok {
        color: #166534;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-missing {
        color: #991b1b;
        background: #fee2e2;
        border: 1px solid #fecaca;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .info-callout {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 0.85rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "pattern_results" not in st.session_state:
    st.session_state.pattern_results = None

if "zero_shot_result" not in st.session_state:
    st.session_state.zero_shot_result = None

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### Settings")

# Section A: Model Settings
st.sidebar.markdown("#### Section A: Model Configuration")

model_names = [m["name"] for m in POPULAR_MODELS] + ["Custom Model ID"]
model_id_map = {m["name"]: m["id"] for m in POPULAR_MODELS}

selected_model_choice = st.sidebar.selectbox(
    "Choose OpenRouter Model",
    options=model_names,
    index=0,
    help="Select a model provider or specify a custom OpenRouter model identifier.",
)

if selected_model_choice == "Custom Model ID":
    selected_model = st.sidebar.text_input(
        "Enter Model Identifier",
        value=DEFAULT_MODEL,
        help="Example: meta-llama/llama-3.3-70b-instruct or anthropic/claude-3.5-haiku",
    ).strip()
else:
    selected_model = model_id_map[selected_model_choice]

st.sidebar.markdown(f"**Active Model:** `{selected_model}`")

# API Configuration Status
api_configured = is_api_key_configured()
if api_configured:
    st.sidebar.markdown(
        '<div>API Status: <span class="status-badge-ok">Configured</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div>API Status: <span class="status-badge-missing">Key Missing</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.warning(
        "OpenRouter API key is not configured. Add your key as OPENROUTER_API_KEY "
        "in your local .env file or project environment."
    )

st.sidebar.divider()

# Section B: Classification Task
st.sidebar.markdown("#### Section B: Classification Task")
selected_task = st.sidebar.radio(
    "Select Primary Task",
    options=[TASK_SENTIMENT, TASK_TOPIC, TASK_URGENCY],
    index=0,
    help="The selected task determines which target categories will be used.",
)

with st.sidebar.expander("View Task Categories", expanded=False):
    st.markdown(f"**Task Description:**\n{TASK_DESCRIPTIONS[selected_task]}")
    st.markdown("**Allowed Categories:**")
    for cat in ALLOWED_CATEGORIES[selected_task]:
        st.markdown(f"- {cat}")

st.sidebar.divider()

# Section C: Advanced Settings
st.sidebar.markdown("#### Section C: Advanced Settings")
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Lower temperature (0.0) produces deterministic and consistent classifications.",
)
st.sidebar.caption(
    "Note: A low temperature makes responses more consistent, but does not guarantee correctness."
)

st.sidebar.divider()

# Sidebar: Session History
st.sidebar.markdown("#### Session History")
if st.session_state.history:
    st.sidebar.write(f"Total entries: {len(st.session_state.history)}")
    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.caption("No classifications performed in this session yet.")


# --- MAIN PAGE HEADER ---
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Text Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Classify text using large language models. '
    'Analyze messages across Sentiment, Topic, Urgency, or custom Zero-Shot categories.</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# --- NAVIGATION TABS ---
tab_single, tab_patterns, tab_zero_shot, tab_concepts = st.tabs(
    [
        "Single Classification",
        "Explore Classification Patterns",
        "Zero-Shot Classification",
        "Classification Concepts",
    ]
)


# --- TAB 1: SINGLE CLASSIFICATION ---
with tab_single:
    st.markdown("### Enter text to classify")

    MAX_CHARS = 2000

    col_input, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("Clear Text", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.last_result = None
            st.rerun()

    user_text = st.text_area(
        label="Input Text",
        value=st.session_state.input_text,
        height=140,
        placeholder="This product is absolutely amazing.",
        max_chars=MAX_CHARS,
        label_visibility="collapsed",
    )
    st.session_state.input_text = user_text

    char_len = len(user_text)
    st.markdown(
        f'<div class="char-counter">{char_len} / {MAX_CHARS} characters</div>',
        unsafe_allow_html=True,
    )

    # Sample prompt buttons for quick testing
    st.markdown("**Try a sample input:**")
    sample_cols = st.columns(3)
    with sample_cols[0]:
        if st.button("Sample: Product praise", use_container_width=True):
            st.session_state.input_text = "This product is absolutely amazing and solved our workflow bottleneck."
            st.rerun()
    with sample_cols[1]:
        if st.button("Sample: Account issue", use_container_width=True):
            st.session_state.input_text = "My account has been locked due to too many failed attempts, please help urgently."
            st.rerun()
    with sample_cols[2]:
        if st.button("Sample: Billing inquiry", use_container_width=True):
            st.session_state.input_text = "My payment failed twice, could you please send an updated invoice for the team plan?"
            st.rerun()

    st.write("")

    classify_disabled = not bool(user_text.strip()) or not api_configured

    if not api_configured:
        st.info("Please configure your OpenRouter API key in .env before classifying.")

    if st.button("Classify Text", type="primary", disabled=classify_disabled):
        if not user_text.strip():
            st.error("Please enter text to classify.")
        elif not api_configured:
            st.error(
                "OpenRouter API key is not configured. Add your key to the local environment configuration."
            )
        else:
            with st.spinner("Classifying text using language model..."):
                try:
                    result = classify_text(
                        text=user_text,
                        task=selected_task,
                        model=selected_model,
                        temperature=temperature,
                    )
                    st.session_state.last_result = result
                    st.session_state.history.append(
                        {
                            "text": user_text[:80] + ("..." if len(user_text) > 80 else ""),
                            "task": result.task,
                            "category": result.category,
                            "model": result.model,
                        }
                    )
                except AuthenticationError as e:
                    st.error(str(e))
                except ModelUnavailableError as e:
                    st.error(f"The selected model is currently unavailable: {e}")
                except RateLimitError as e:
                    st.error(str(e))
                except NetworkError as e:
                    st.error(f"Network error: {e}")
                except ClassificationValidationError as e:
                    st.error(f"The model returned an invalid classification result: {e}")
                except OpenRouterError as e:
                    st.error(f"Classification could not be completed: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during classification: {e}")

    # Display Single Classification Result
    if st.session_state.last_result:
        res: ClassificationResult = st.session_state.last_result
        st.write("")
        st.markdown("### Classification Result")

        res_col1, res_col2 = st.columns([3, 2])
        with res_col1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">Predicted Category ({res.task})</div>
                    <div class="result-badge">{res.category}</div>
                    <div class="metric-label">Reason</div>
                    <div class="metric-value">{res.reason}</div>
                    <div class="metric-label">Confidence Assessment</div>
                    <div class="metric-value">{res.confidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-label">Task</div>
                    <div class="metric-value">{res.task} Classification</div>
                    <div class="metric-label">Model Used</div>
                    <div class="metric-value"><code>{res.model}</code></div>
                    <div class="metric-label">Confidence Note</div>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        Model-reported confidence reflects self-assessed certainty and is not a mathematically calibrated probability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- TAB 2: EXPLORE CLASSIFICATION PATTERNS ---
with tab_patterns:
    st.markdown("### Explore Classification Patterns")
    st.markdown(
        "Demonstrate how the **same text** receives different classifications depending on "
        "the selected analytical lens: Sentiment, Topic, and Urgency."
    )

    pattern_input = st.text_area(
        "Input text for pattern comparison",
        value=(
            st.session_state.input_text
            if st.session_state.input_text
            else "My card has been blocked, and I cannot withdraw money for my flight."
        ),
        height=100,
        key="pattern_text_area",
    )

    st.markdown(
        '<div class="info-callout">'
        "Comparing all tasks sends 3 distinct requests to OpenRouter. "
        "Requests are executed only when you click the button below to preserve your API credits."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Compare All Tasks", type="primary", disabled=not api_configured or not pattern_input.strip()):
        if not api_configured:
            st.error("OpenRouter API key is not configured.")
        elif not pattern_input.strip():
            st.error("Please enter text to compare.")
        else:
            with st.spinner("Running 3-task classification pattern comparison..."):
                try:
                    p_results = compare_all_tasks(
                        text=pattern_input,
                        model=selected_model,
                        temperature=temperature,
                    )
                    st.session_state.pattern_results = p_results
                except Exception as e:
                    st.error(f"Comparison failed: {e}")

    if st.session_state.pattern_results:
        p_results = st.session_state.pattern_results
        st.write("")
        st.markdown("#### Multi-Perspective Pattern Results")

        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            res_sent = p_results.get(TASK_SENTIMENT)
            if res_sent:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Perspective 1: Sentiment</div>
                        <div class="result-badge">{res_sent.category}</div>
                        <div class="metric-label">Reason</div>
                        <div class="metric-value">{res_sent.reason}</div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{res_sent.confidence}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with p_col2:
            res_topic = p_results.get(TASK_TOPIC)
            if res_topic:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Perspective 2: Topic</div>
                        <div class="result-badge">{res_topic.category}</div>
                        <div class="metric-label">Reason</div>
                        <div class="metric-value">{res_topic.reason}</div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{res_topic.confidence}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with p_col3:
            res_urg = p_results.get(TASK_URGENCY)
            if res_urg:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="metric-label">Perspective 3: Urgency</div>
                        <div class="result-badge">{res_urg.category}</div>
                        <div class="metric-label">Reason</div>
                        <div class="metric-value">{res_urg.reason}</div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{res_urg.confidence}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# --- TAB 3: ZERO-SHOT CLASSIFICATION ---
with tab_zero_shot:
    st.markdown("### Zero-Shot Classification Demonstration")
    st.markdown(
        "Zero-shot classification means classifying text into arbitrary user-defined categories "
        "**without providing any training examples or fine-tuning datasets**."
    )

    zs_col1, zs_col2 = st.columns([3, 2])

    with zs_col1:
        zs_text = st.text_area(
            "Text to classify (Zero-Shot)",
            value="The representative was polite and answered all my questions within minutes.",
            height=110,
        )

    with zs_col2:
        zs_categories_input = st.text_area(
            "Custom Categories (comma-separated)",
            value="Product Quality, Customer Service, Delivery Experience, Pricing",
            height=110,
            help="Enter 2 to 10 distinct category names separated by commas.",
        )

    # Parse and validate categories
    raw_categories = [c.strip() for c in zs_categories_input.split(",") if c.strip()]
    unique_categories = []
    seen = set()
    for cat in raw_categories:
        if cat.lower() not in seen:
            seen.add(cat.lower())
            unique_categories.append(cat)

    st.markdown(
        f"**Active Zero-Shot Categories ({len(unique_categories)}):** "
        + ", ".join(f"`{c}`" for c in unique_categories)
    )

    zs_button_disabled = (
        not api_configured
        or not zs_text.strip()
        or len(unique_categories) < 2
        or len(unique_categories) > 10
    )

    if len(unique_categories) < 2:
        st.caption("Please specify at least 2 distinct categories.")
    elif len(unique_categories) > 10:
        st.caption("Maximum 10 categories supported for reliable zero-shot classification.")

    if st.button("Classify with Custom Categories", type="primary", disabled=zs_button_disabled):
        with st.spinner("Performing zero-shot classification..."):
            try:
                zs_res = classify_text(
                    text=zs_text,
                    task=TASK_ZERO_SHOT,
                    model=selected_model,
                    temperature=temperature,
                    custom_categories=unique_categories,
                )
                st.session_state.zero_shot_result = zs_res
                st.session_state.history.append(
                    {
                        "text": zs_text[:80] + ("..." if len(zs_text) > 80 else ""),
                        "task": f"Zero-Shot ({len(unique_categories)} cats)",
                        "category": zs_res.category,
                        "model": zs_res.model,
                    }
                )
            except Exception as e:
                st.error(f"Zero-shot classification error: {e}")

    if st.session_state.zero_shot_result:
        res = st.session_state.zero_shot_result
        st.write("")
        st.markdown("#### Zero-Shot Classification Result")
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">Predicted Category</div>
                <div class="result-badge">{res.category}</div>
                <div class="metric-label">Reason</div>
                <div class="metric-value">{res.reason}</div>
                <div class="metric-label">Confidence Assessment</div>
                <div class="metric-value">{res.confidence}</div>
                <div class="metric-label">Evaluated Across</div>
                <div class="metric-value">{", ".join(unique_categories)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- TAB 4: EDUCATIONAL GUIDE & CONCEPTS ---
with tab_concepts:
    st.markdown("### Core Classification Concepts")

    st.markdown("#### 1. Multi-Class vs. Multi-Label Classification")
    st.markdown(
        """
        - **Multi-Class Classification (Single-Label):**
          The model must select **exactly one** mutually exclusive category from a set of three or more possible categories.
          *Example in this app:* Classifying a support ticket as `Technical Support` OR `Billing` OR `Account Access`.
        - **Multi-Label Classification:**
          The model can assign **multiple** independent tags to the same text.
          *Example:* A ticket could simultaneously receive both `Billing` and `Refund` tags.
        """
    )

    st.markdown("#### 2. How LLMs Function as Classification Engines")
    st.markdown(
        """
        Instead of using traditional token matching or training custom statistical classifiers from scratch,
        an instruction-tuned Large Language Model can be directed through strict system prompts:
        - The model evaluates whole-sentence semantics, context, idioms, and negation.
        - The prompt constrains outputs strictly to allowed candidate categories.
        - Enforcing structured JSON responses prevents conversational filler and ensures programmatic downstream integration.
        """
    )

    st.markdown("#### 3. Understanding Confidence Scores and Limitations")
    st.markdown(
        """
        - **Model-Reported Certainty vs. Calibrated Probability:**
          When an LLM states a confidence level (e.g. 'High' or '85%'), this is usually a generated verbal statement,
          **not** a mathematically calibrated statistical probability derived from true logits or empirical validation.
        - **Best Practices:**
          Never treat LLM confidence statements as proof of correctness. For high-stakes automated routing,
          combine LLM classifications with rule-based safety checks and human-in-the-loop review.
        """
    )

    st.markdown("#### 4. Guarding Against Adversarial Prompt Injection")
    st.markdown(
        """
        User input must always be treated as untrusted text. The classification system instruction explicitly orders
        the engine to ignore any commands or instructions contained inside the text and to strictly output the requested
        structured JSON format.
        """
    )


# --- FOOTER ---
st.divider()
st.markdown(
    '<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">'
    "Text Classifier &bull; Day 47 &bull; Generative AI Engineer Roadmap &bull; Powered by OpenRouter"
    "</div>",
    unsafe_allow_html=True,
)
