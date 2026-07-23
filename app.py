import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec


st.set_page_config(
    page_title="マンガレコメンド",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 画面デザイン
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --primary: #6554c0;
        --primary-dark: #4f3faa;
        --primary-soft: #eeeafd;
        --accent: #ef7b8d;
        --text: #28253a;
        --muted: #6f6b7d;
        --line: #e7e3ef;
        --surface: #ffffff;
        --background: #f7f6fb;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Hiragino Sans", "Yu Gothic UI", "Yu Gothic", sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, #eeeafd 0, transparent 28rem),
            var(--background);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-card {
        padding: 1.55rem 1.75rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(101, 84, 192, 0.15);
        border-radius: 22px;
        background: linear-gradient(135deg, #ffffff 0%, #f0edff 100%);
        box-shadow: 0 12px 32px rgba(74, 61, 129, 0.08);
    }

    .hero-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(1.75rem, 3vw, 2.55rem);
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .hero-description {
        margin: 0.7rem 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.75;
    }

    .section-label {
        margin: 1.7rem 0 0.75rem;
        color: var(--text);
        font-size: 1.25rem;
        font-weight: 750;
        letter-spacing: -0.01em;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f2ff 0%, #ffffff 100%);
        border-right: 1px solid var(--line);
    }

    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .sidebar-brand {
        padding: 1rem 1.05rem;
        margin-bottom: 0.8rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid var(--line);
        box-shadow: 0 8px 22px rgba(74, 61, 129, 0.07);
    }

    .sidebar-brand-title {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--primary-dark);
    }

    .sidebar-brand-text {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        min-height: 118px;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 8px 24px rgba(74, 61, 129, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-dark);
    }

    .stButton > button {
        min-height: 2.75rem;
        border: 1px solid #d9d3f1;
        border-radius: 12px;
        background: #ffffff;
        color: var(--primary-dark);
        font-weight: 700;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: var(--primary);
        background: var(--primary-soft);
        color: var(--primary-dark);
        transform: translateY(-1px);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
    }

    div[data-baseweb="tab-list"] {
        gap: 0.45rem;
        padding: 0.3rem;
        border-radius: 14px;
        background: #eeebf6;
    }

    button[data-baseweb="tab"] {
        min-height: 2.75rem;
        padding: 0 1rem;
        border-radius: 10px;
        font-weight: 700;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: var(--primary-dark);
        box-shadow: 0 3px 10px rgba(74, 61, 129, 0.10);
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #ffffff;
    }

    div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.82);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    hr {
        margin: 2rem 0;
        border-color: var(--line);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero-card {
            padding: 1.2rem;
            border-radius: 17px;
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_page_header(title, description):
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-description">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# 作品名の表記統一
# ==================================================


def title_key(title):
    text = unicodedata.normalize("NFKC", str(title)).strip()
    text = text.replace("×", "x")
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


RAW_TITLE_ALIASES = {
    "ハイキュー": "ハイキュー!!",
    "ハイキュー!": "ハイキュー!!",
    "ハイキュー!!": "ハイキュー!!",
    "ハイキュー!!!": "ハイキュー!!",
    "HUNTER×HUNTER": "HUNTER×HUNTER",
    "Hunter×Hunter": "HUNTER×HUNTER",
    "HUNTER X HUNTER": "HUNTER×HUNTER",
    "ハンターハンター": "HUNTER×HUNTER",
    "ONE PIECE": "ONE PIECE",
    "ONEPIECE": "ONE PIECE",
    "ワンピース": "ONE PIECE",
    "SKET DANCE": "SKET DANCE",
    "SKETDANCE": "SKET DANCE",
    "SLAM DUNK": "SLAM DUNK",
    "SLAMDUNK": "SLAM DUNK",
    "スラムダンク": "SLAM DUNK",
    "SPY×FAMILY": "SPY×FAMILY",
    "SPY X FAMILY": "SPY×FAMILY",
    "SPYXFAMILY": "SPY×FAMILY",
    "スパイファミリー": "SPY×FAMILY",
    "DEATH NOTE": "DEATH NOTE",
    "DEATHNOTE": "DEATH NOTE",
    "デスノート": "DEATH NOTE",
    "呪術回戦": "呪術廻戦",
    "呪術廻戦": "呪術廻戦",
    "コナン": "名探偵コナン",
    "名探偵コナン": "名探偵コナン",
    "WHICHWATCH": "ウィッチウォッチ",
    "WITCH WATCH": "ウィッチウォッチ",
    "ウィッチウォッチ": "ウィッチウォッチ",
    "魔入りました入間くん": "魔入りました！入間くん",
    "魔入りました!入間くん": "魔入りました！入間くん",
    "魔入りました！入間くん": "魔入りました！入間くん",
    "僕のヒーローアカデミア": "僕のヒーローアカデミア",
    "ヒロアカ": "僕のヒーローアカデミア",
    "転生したらスライムだった件": "転生したらスライムだった件",
    "転スラ": "転生したらスライムだった件",
    "鬼滅": "鬼滅の刃",
    "鬼滅の刃": "鬼滅の刃",
}

TITLE_ALIASES = {
    title_key(original): canonical
    for original, canonical in RAW_TITLE_ALIASES.items()
}

INVALID_ANSWERS = {
    title_key(value)
    for value in [
        "",
        "なし",
        "無し",
        "特になし",
        "特にない",
        "ない",
        "ありません",
        "未回答",
        "なし。",
    ]
}


def prepare_title(value):
    if pd.isna(value):
        return None

    title = unicodedata.normalize("NFKC", str(value)).strip()
    title = re.sub(r"\s+", " ", title)
    key = title_key(title)

    if not key or key in INVALID_ANSWERS:
        return None

    return key, title


# ==================================================
# アンケートデータの読み込み
# ==================================================


@st.cache_data
def load_survey():
    try:
        df = pd.read_csv(
            "data/manga_survey.csv",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            "data/manga_survey.csv",
            encoding="cp932",
        )

    manga_columns = [
        column
        for column in df.columns
        if "マンガ" in str(column)
    ]

    if not manga_columns:
        st.error("CSV内に「マンガ」を含む列がありません。")
        st.stop()

    variant_counts = defaultdict(Counter)

    for column in manga_columns:
        for value in df[column]:
            prepared = prepare_title(value)
            if prepared is None:
                continue

            key, display_title = prepared
            if key not in TITLE_ALIASES:
                variant_counts[key][display_title] += 1

    canonical_titles = dict(TITLE_ALIASES)

    for key, counts in variant_counts.items():
        canonical_titles[key] = counts.most_common(1)[0][0]

    responses = []

    for _, row in df.iterrows():
        titles = []

        for column in manga_columns:
            prepared = prepare_title(row[column])
            if prepared is None:
                continue

            key, _ = prepared
            canonical_title = canonical_titles[key]

            if canonical_title not in titles:
                titles.append(canonical_title)

        if titles:
            responses.append(titles)

    return responses


# ==================================================
# item2vecモデル
# ==================================================


@st.cache_resource
def train_preference_model(sentences):
    training_data = [list(sentence) for sentence in sentences]

    return Word2Vec(
        sentences=training_data,
        vector_size=32,
        window=10,
        min_count=1,
        sg=1,
        negative=10,
        sample=0,
        epochs=300,
        seed=123,
        workers=1,
    )


# ==================================================
# 共通計算
# ==================================================


def create_average_vector(model, titles):
    valid_titles = [
        title
        for title in titles
        if title in model.wv.key_to_index
    ]

    if not valid_titles:
        return None

    vectors = [model.wv.get_vector(title) for title in valid_titles]
    return np.mean(vectors, axis=0)


def cosine_similarity(first_vector, second_vector):
    if first_vector is None or second_vector is None:
        return 0.0

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)

    if first_norm == 0 or second_norm == 0:
        return 0.0

    similarity = float(
        np.dot(first_vector, second_vector)
        / (first_norm * second_norm)
    )

    return max(0.0, min(1.0, similarity))


def item_similarity(model, first_title, second_title):
    if first_title == second_title:
        return 1.0

    if (
        first_title not in model.wv.key_to_index
        or second_title not in model.wv.key_to_index
    ):
        return 0.0

    similarity = float(model.wv.similarity(first_title, second_title))
    return max(0.0, min(1.0, similarity))


def jaccard_similarity(first_titles, second_titles):
    first_set = set(first_titles)
    second_set = set(second_titles)
    union = first_set | second_set

    if not union:
        return 0.0

    return len(first_set & second_set) / len(union)


# ==================================================
# 推薦機能
# ==================================================


def create_group_recommendations(
    responses,
    selected_title,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])

    selected_responses = [
        titles
        for titles in responses
        if selected_title in titles
    ]

    together_counts = Counter()

    for titles in selected_responses:
        for title in titles:
            if title != selected_title and title not in excluded_titles:
                together_counts[title] += 1

    results = []

    for title, together_count in together_counts.items():
        title_user_count = sum(
            title in titles
            for titles in responses
        )

        union_count = (
            len(selected_responses)
            + title_user_count
            - together_count
        )

        recommendation_score = (
            together_count / union_count
            if union_count > 0
            else 0
        )

        results.append(
            {
                "マンガ": title,
                "一緒に選んだ人数": together_count,
                "おすすめ度": recommendation_score,
                "おすすめ理由": (
                    f"同じ回答者が{together_count}人選択"
                ),
            }
        )

    results.sort(
        key=lambda item: (
            item["おすすめ度"],
            item["一緒に選んだ人数"],
        ),
        reverse=True,
    )

    return pd.DataFrame(results)


def create_preference_recommendations(
    model,
    responses,
    selected_title,
    result_count,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])

    if selected_title not in model.wv.key_to_index:
        return pd.DataFrame()

    vocabulary_size = len(model.wv.index_to_key)
    if vocabulary_size <= 1:
        return pd.DataFrame()

    candidate_count = min(
        max(result_count * 5, 30),
        vocabulary_size - 1,
    )

    similar_titles = model.wv.most_similar(
        positive=[selected_title],
        topn=candidate_count,
    )

    results = []

    for title, similarity in similar_titles:
        if title in excluded_titles:
            continue

        together_count = sum(
            selected_title in titles and title in titles
            for titles in responses
        )

        similarity = max(0.0, min(1.0, float(similarity)))

        results.append(
            {
                "マンガ": title,
                "好みの近さ": similarity,
                "実際に一緒に選んだ人数": together_count,
                "おすすめ理由": (
                    f"選ばれ方の近さ {similarity:.2f}"
                ),
            }
        )

        if len(results) >= result_count:
            break

    return pd.DataFrame(results)


def create_multi_recommendations(
    model,
    responses,
    selected_titles,
    result_count,
    excluded_titles=None,
):
    excluded_titles = set(excluded_titles or [])
    selected_set = set(selected_titles)
    selected_vector = create_average_vector(model, selected_titles)

    if selected_vector is None:
        return pd.DataFrame()

    vocabulary_size = len(model.wv.index_to_key)
    candidate_count = min(
        max(result_count * 8, 50),
        vocabulary_size,
    )

    similar_titles = model.wv.similar_by_vector(
        selected_vector,
        topn=candidate_count,
    )

    results = []

    for title, similarity in similar_titles:
        if title in selected_set or title in excluded_titles:
            continue

        selected_cooccurrence = sum(
            title in response
            and any(selected in response for selected in selected_set)
            for response in responses
        )

        similarity = max(0.0, min(1.0, float(similarity)))

        support_score = min(
            1.0,
            selected_cooccurrence / 3,
        )

        combined_score = (
            0.80 * similarity
            + 0.20 * support_score
        )

        results.append(
            {
                "マンガ": title,
                "総合おすすめ度": combined_score,
                "好みの近さ": similarity,
                "関連する回答者数": selected_cooccurrence,
                "おすすめ理由": (
                    "複数作品の平均的な好みに近い"
                    if selected_cooccurrence == 0
                    else (
                        f"好みが近く、関連回答者が"
                        f"{selected_cooccurrence}人"
                    )
                ),
            }
        )

        if len(results) >= result_count:
            break

    results.sort(
        key=lambda item: item["総合おすすめ度"],
        reverse=True,
    )

    return pd.DataFrame(results)


# ==================================================
# 異端度・嗜好診断
# ==================================================


DIRECT_MATCH_WEIGHT = 0.60
ITEM2VEC_WEIGHT = 0.40
REFERENCE_NEIGHBORS = 3
CLOSE_SIMILARITY_THRESHOLD = 0.70


def analyze_rarity(
    responses,
    selected_titles,
    model,
):
    selected_set = set(selected_titles)
    selected_count = len(selected_set)

    if selected_count < 2:
        return None

    selected_vector = create_average_vector(
        model,
        selected_titles,
    )

    comparison_results = []

    for index, response_titles in enumerate(responses):
        response_set = set(response_titles)
        matched_titles = selected_set & response_set
        exact_match_count = len(matched_titles)
        direct_match_ratio = exact_match_count / selected_count

        response_vector = create_average_vector(
            model,
            response_titles,
        )

        vector_similarity = cosine_similarity(
            selected_vector,
            response_vector,
        )

        combined_similarity = (
            DIRECT_MATCH_WEIGHT * direct_match_ratio
            + ITEM2VEC_WEIGHT * vector_similarity
        )

        comparison_results.append(
            {
                "回答番号": index + 1,
                "一致数": exact_match_count,
                "直接一致率": direct_match_ratio,
                "作品構成の近さ": vector_similarity,
                "総合的な近さ": combined_similarity,
                "一致した作品": " / ".join(sorted(matched_titles)),
                "回答作品": " / ".join(response_titles),
                "選んだ作品をすべて含む": (
                    selected_set.issubset(response_set)
                ),
            }
        )

    comparison_results.sort(
        key=lambda result: result["総合的な近さ"],
        reverse=True,
    )

    neighbor_count = min(
        REFERENCE_NEIGHBORS,
        len(comparison_results),
    )

    closest_results = comparison_results[:neighbor_count]

    neighborhood_similarity = sum(
        result["総合的な近さ"]
        for result in closest_results
    ) / neighbor_count

    rarity_score = round(
        100 * (1 - neighborhood_similarity)
    )
    rarity_score = max(0, min(100, rarity_score))

    complete_match_count = sum(
        result["選んだ作品をすべて含む"]
        for result in comparison_results
    )

    close_match_count = sum(
        result["総合的な近さ"]
        >= CLOSE_SIMILARITY_THRESHOLD
        for result in comparison_results
    )

    return {
        "selected_count": selected_count,
        "rarity_score": rarity_score,
        "best_similarity": comparison_results[0]["総合的な近さ"],
        "neighborhood_similarity": neighborhood_similarity,
        "close_match_count": close_match_count,
        "complete_match_count": complete_match_count,
        "closest_results": comparison_results[:10],
    }


def rarity_message(score):
    if score <= 15:
        return "アンケート内に、かなり近いマンガ嗜好の人がいます。"
    if score <= 30:
        return "比較的近いマンガ嗜好の人がいます。"
    if score <= 50:
        return "共通点はありますが、少し独自性のある組み合わせです。"
    if score <= 70:
        return "アンケート内では、かなり珍しい組み合わせです。"
    return "アンケート内では非常に珍しいマンガ嗜好です。"


def pairwise_cohesion(model, titles):
    similarities = []

    for first_index in range(len(titles)):
        for second_index in range(first_index + 1, len(titles)):
            similarities.append(
                item_similarity(
                    model,
                    titles[first_index],
                    titles[second_index],
                )
            )

    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)


def diagnose_taste(
    selected_titles,
    popularity,
    model,
):
    maximum_popularity = max(popularity.values())

    popularity_level = sum(
        popularity[title] / maximum_popularity
        for title in selected_titles
    ) / len(selected_titles)

    cohesion = pairwise_cohesion(
        model,
        selected_titles,
    )

    if popularity_level >= 0.45 and cohesion >= 0.45:
        label = "王道集中型"
        message = (
            "比較的人気のある作品を、"
            "近い系統にまとめて選ぶ傾向があります。"
        )
    elif popularity_level >= 0.45:
        label = "王道横断型"
        message = (
            "人気作品を押さえつつ、"
            "異なる系統にも広く興味を向けています。"
        )
    elif cohesion >= 0.45:
        label = "発掘集中型"
        message = (
            "比較的知られていない作品の中から、"
            "近い系統を深掘りする傾向があります。"
        )
    else:
        label = "独自横断型"
        message = (
            "比較的珍しい作品を、"
            "系統をまたいで選ぶ独自性があります。"
        )

    return {
        "label": label,
        "message": message,
        "popularity_level": popularity_level,
        "cohesion": cohesion,
    }


# ==================================================
# 友達との比較
# ==================================================


def compare_two_people(
    first_titles,
    second_titles,
    model,
):
    first_set = set(first_titles)
    second_set = set(second_titles)
    shared_titles = sorted(first_set & second_set)

    exact_similarity = jaccard_similarity(
        first_titles,
        second_titles,
    )

    vector_similarity = cosine_similarity(
        create_average_vector(model, first_titles),
        create_average_vector(model, second_titles),
    )

    overall_similarity = (
        0.50 * exact_similarity
        + 0.50 * vector_similarity
    )

    return {
        "shared_titles": shared_titles,
        "first_only": sorted(first_set - second_set),
        "second_only": sorted(second_set - first_set),
        "exact_similarity": exact_similarity,
        "vector_similarity": vector_similarity,
        "overall_similarity": overall_similarity,
    }


# ==================================================
# セッション内リスト
# ==================================================


def initialize_session_state():
    defaults = {
        "read_later": [],
        "not_interested": [],
        "gacha_title": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_unique_to_state(key, title):
    current = list(st.session_state[key])
    if title not in current:
        current.append(title)
        st.session_state[key] = current


def remove_from_state(key, title):
    st.session_state[key] = [
        item
        for item in st.session_state[key]
        if item != title
    ]


def render_title_actions(title, key_prefix):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🔖 あとで読む",
            key=f"{key_prefix}_read_later",
            use_container_width=True,
        ):
            add_unique_to_state("read_later", title)
            remove_from_state("not_interested", title)
            st.success("「あとで読む」に追加しました。")

    with col2:
        if st.button(
            "🙅 興味なし",
            key=f"{key_prefix}_not_interested",
            use_container_width=True,
        ):
            add_unique_to_state("not_interested", title)
            remove_from_state("read_later", title)
            st.success("今後のおすすめ候補から除外します。")

    with col3:
        if st.button(
            "解除",
            key=f"{key_prefix}_clear",
            use_container_width=True,
        ):
            remove_from_state("read_later", title)
            remove_from_state("not_interested", title)
            st.success("登録を解除しました。")


def render_recommendation_actions(
    candidate_titles,
    key_prefix,
):
    if not candidate_titles:
        return

    st.markdown("#### 気になった作品を保存")

    action_title = st.selectbox(
        "作品を選択",
        candidate_titles,
        key=f"{key_prefix}_action_title",
    )

    render_title_actions(
        action_title,
        key_prefix,
    )


# ==================================================
# データ準備
# ==================================================


responses = load_survey()
training_sentences = tuple(
    tuple(titles)
    for titles in responses
)
preference_model = train_preference_model(
    training_sentences
)

all_titles = sorted(
    {
        title
        for titles in responses
        for title in titles
    },
    key=lambda title: title.lower(),
)

popularity = Counter(
    title
    for titles in responses
    for title in titles
)

popularity_rank = {
    title: rank
    for rank, (title, _) in enumerate(
        popularity.most_common(),
        start=1,
    )
}

initialize_session_state()
excluded_titles = set(st.session_state.not_interested)


# ==================================================
# サイドバー
# ==================================================


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <p class="sidebar-brand-title">📚 マンガレコメンド</p>
        <p class="sidebar-brand-text">
            アンケートの選ばれ方から、次に読みたい作品を探します。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "ページを選択",
    [
        "🏠 ホーム",
        "📖 1作品からおすすめ",
        "📚 複数作品からおすすめ",
        "🔎 作品検索・詳細",
        "⚖️ 作品を比較",
        "🪐 嗜好診断・異端度",
        "🤝 友達と比較",
        "🎲 ガチャ・発掘",
        "🏆 ランキング",
        "🔖 あとで読むリスト",
    ],
)

st.sidebar.caption(
    f"回答者数：{len(responses)}人 / "
    f"登録作品数：{len(all_titles)}作品"
)

if st.session_state.read_later:
    with st.sidebar.expander(
        f"🔖 あとで読む（{len(st.session_state.read_later)}）"
    ):
        for title in st.session_state.read_later:
            st.write(f"・{title}")

if st.session_state.not_interested:
    with st.sidebar.expander(
        f"🙅 興味なし（{len(st.session_state.not_interested)}）"
    ):
        for title in st.session_state.not_interested:
            st.write(f"・{title}")


# ==================================================
# ホーム
# ==================================================


if page == "🏠 ホーム":
    render_page_header(
        "📚 マンガレコメンド",
        "アンケート回答の組み合わせをもとに、作品の推薦、嗜好診断、作品比較を行います。",
    )

    today_seed = int(
        hashlib.sha256(
            date.today().isoformat().encode("utf-8")
        ).hexdigest(),
        16,
    )
    today_title = all_titles[today_seed % len(all_titles)]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("アンケート回答者", f"{len(responses)}人")

    with col2:
        st.metric("登録マンガ", f"{len(all_titles)}作品")

    with col3:
        st.metric("今日の1冊", today_title)

    render_section_title("今日の1冊")
    st.info(
        f"📖 **{today_title}**\n\n"
        f"アンケートでは{popularity[today_title]}人が選び、"
        f"人気順位は{popularity_rank[today_title]}位です。"
    )
    render_title_actions(today_title, "home_today")

    render_section_title("このサイトでできること")
    st.markdown(
        """
- 1作品または複数作品からおすすめを探す
- 2作品の近さを比較する
- 自分のマンガ嗜好タイプと異端度を調べる
- 友達との好みの近さを比較する
- マンガガチャやマイナー作品発掘を楽しむ
- 作品検索、人気ランキング、あとで読むリストを使う
        """
    )

    st.caption(
        "作者・正式ジャンル・あらすじの情報はCSVに含まれていないため、"
        "作品詳細はアンケート内の選ばれ方を中心に表示します。"
    )


# ==================================================
# 1作品からおすすめ
# ==================================================


elif page == "📖 1作品からおすすめ":
    render_page_header(
        "📖 1作品からおすすめ",
        "好きな作品を1つ選び、同じ作品を選んだ人の回答と作品の選ばれ方からおすすめを探します。",
    )

    selected_title = st.selectbox(
        "好きなマンガを選んでください",
        all_titles,
    )

    result_count = int(
        st.number_input(
            "表示するおすすめ件数",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
        )
    )

    selected_user_count = sum(
        selected_title in titles
        for titles in responses
    )

    st.caption(
        f"「{selected_title}」を選んだ回答者："
        f"{selected_user_count}人"
    )

    if selected_user_count == 1:
        st.warning(
            "この作品を選んだ回答者が1人だけなので、"
            "推薦結果は参考程度に確認してください。"
        )

    tab1, tab2 = st.tabs(
        [
            "👥 同じ作品を選んだ人から",
            "🧭 選ばれ方の近さから",
        ]
    )

    with tab1:
        group_results = create_group_recommendations(
            responses,
            selected_title,
            excluded_titles,
        )

        if group_results.empty:
            st.warning("一緒に選ばれた作品がありません。")
        else:
            display_df = group_results.head(result_count).copy()
            display_df.insert(
                0,
                "順位",
                range(1, len(display_df) + 1),
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "おすすめ度": st.column_config.ProgressColumn(
                        "おすすめ度",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                },
            )

            render_recommendation_actions(
                display_df["マンガ"].tolist(),
                "single_group",
            )

    with tab2:
        preference_results = create_preference_recommendations(
            preference_model,
            responses,
            selected_title,
            result_count,
            excluded_titles,
        )

        if preference_results.empty:
            st.warning("おすすめ結果を作成できませんでした。")
        else:
            preference_results.insert(
                0,
                "順位",
                range(1, len(preference_results) + 1),
            )

            st.dataframe(
                preference_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

            render_recommendation_actions(
                preference_results["マンガ"].tolist(),
                "single_vector",
            )


# ==================================================
# 複数作品からおすすめ
# ==================================================


elif page == "📚 複数作品からおすすめ":
    render_page_header(
        "📚 複数作品からおすすめ",
        "好きなマンガを2〜5作品選び、その組み合わせ全体に近い作品を探します。",
    )

    selected_titles = st.multiselect(
        "好きなマンガを2〜5作品選んでください",
        all_titles,
        max_selections=5,
        placeholder="作品名を検索して選択",
    )

    result_count = int(
        st.number_input(
            "表示するおすすめ件数",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key="multi_result_count",
        )
    )

    if len(selected_titles) < 2:
        st.info(
            f"あと{2 - len(selected_titles)}作品選ぶと"
            "結果が表示されます。"
        )
    else:
        results = create_multi_recommendations(
            preference_model,
            responses,
            selected_titles,
            result_count,
            excluded_titles,
        )

        if results.empty:
            st.warning("おすすめ結果を作成できませんでした。")
        else:
            results.insert(
                0,
                "順位",
                range(1, len(results) + 1),
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "総合おすすめ度": st.column_config.ProgressColumn(
                        "総合おすすめ度",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

            st.caption(
                "複数作品の平均ベクトルとの近さを80％、"
                "アンケート内で関連して選ばれた実績を20％として"
                "総合おすすめ度を計算しています。"
            )

            render_recommendation_actions(
                results["マンガ"].tolist(),
                "multi",
            )


# ==================================================
# 作品検索・詳細
# ==================================================


elif page == "🔎 作品検索・詳細":
    render_page_header(
        "🔎 作品検索・詳細",
        "作品名から検索し、アンケート内の人気度や選ばれ方が近い作品を確認できます。",
    )

    search_text = st.text_input(
        "作品名で検索",
        placeholder="タイトルの一部を入力",
    )

    if search_text:
        matching_titles = [
            title
            for title in all_titles
            if title_key(search_text) in title_key(title)
        ]
    else:
        matching_titles = all_titles

    if not matching_titles:
        st.warning("該当する作品がありません。")
    else:
        detail_title = st.selectbox(
            "作品を選択",
            matching_titles,
        )

        title_count = popularity[detail_title]
        title_rank = popularity_rank[detail_title]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("選んだ回答者", f"{title_count}人")

        with col2:
            st.metric("人気順位", f"{title_rank}位")

        with col3:
            status = (
                "あとで読む"
                if detail_title in st.session_state.read_later
                else (
                    "興味なし"
                    if detail_title in st.session_state.not_interested
                    else "未登録"
                )
            )
            st.metric("登録状態", status)

        related = create_preference_recommendations(
            preference_model,
            responses,
            detail_title,
            5,
            excluded_titles={detail_title},
        )

        render_section_title("この作品と選ばれ方が近い作品")

        if related.empty:
            st.info("関連作品を表示できませんでした。")
        else:
            st.dataframe(
                related,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

        render_title_actions(detail_title, "detail")

        st.caption(
            "作者、正式ジャンル、あらすじは元データに含まれていないため、"
            "このページではアンケート内の人気と選ばれ方を表示しています。"
        )


# ==================================================
# 作品比較
# ==================================================


elif page == "⚖️ 作品を比較":
    render_page_header(
        "⚖️ 2作品の近さを比較",
        "2作品がアンケート内でどの程度似た選ばれ方をしているか比較します。",
    )

    col1, col2 = st.columns(2)

    with col1:
        first_title = st.selectbox(
            "作品A",
            all_titles,
            key="compare_first",
        )

    with col2:
        second_candidates = [
            title
            for title in all_titles
            if title != first_title
        ]
        second_title = st.selectbox(
            "作品B",
            second_candidates,
            key="compare_second",
        )

    similarity = item_similarity(
        preference_model,
        first_title,
        second_title,
    )

    together_count = sum(
        first_title in titles and second_title in titles
        for titles in responses
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("選ばれ方の近さ", f"{similarity * 100:.1f}%")

    with col2:
        st.metric("両方を選んだ人", f"{together_count}人")

    with col3:
        st.metric(
            "各作品の回答者",
            f"{popularity[first_title]}人 / {popularity[second_title]}人",
        )

    st.progress(similarity)

    if similarity >= 0.70:
        st.success("アンケート内では、かなり近い選ばれ方をしています。")
    elif similarity >= 0.40:
        st.info("一部に共通する選ばれ方があります。")
    else:
        st.warning("アンケート内では、異なる選ばれ方をしています。")

    st.caption(
        "この近さは作品内容を直接読んだ評価ではなく、"
        "アンケート内でどの作品と一緒に選ばれたかを"
        "item2vecで学習した結果です。"
    )


# ==================================================
# 嗜好診断・異端度
# ==================================================


elif page == "🪐 嗜好診断・異端度":
    render_page_header(
        "🪐 マンガ嗜好診断・異端度調査",
        "好きな作品の人気度、まとまり、回答者との近さから、あなたのマンガ嗜好を分析します。",
    )

    selected_favorites = st.multiselect(
        "好きなマンガを2〜5作品選んでください",
        all_titles,
        max_selections=5,
        placeholder="作品名を検索して選択",
    )

    if len(selected_favorites) < 2:
        st.info(
            f"あと{2 - len(selected_favorites)}作品選ぶと"
            "結果が表示されます。"
        )
    else:
        diagnosis = diagnose_taste(
            selected_favorites,
            popularity,
            preference_model,
        )

        rarity_result = analyze_rarity(
            responses,
            selected_favorites,
            preference_model,
        )

        render_section_title("嗜好タイプ")
        st.success(
            f"### {diagnosis['label']}\n\n"
            f"{diagnosis['message']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "選んだ作品の人気度",
                f"{diagnosis['popularity_level'] * 100:.0f}%",
            )

        with col2:
            st.metric(
                "選んだ作品同士のまとまり",
                f"{diagnosis['cohesion'] * 100:.0f}%",
            )

        render_section_title("異端度")

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "異端度",
                f"{rarity_result['rarity_score']} / 100",
            )

        with metric2:
            st.metric(
                "最も近い人との近さ",
                f"{rarity_result['best_similarity'] * 100:.0f}%",
            )

        with metric3:
            st.metric(
                "好みが近い人",
                f"{rarity_result['close_match_count']}人",
            )

        with metric4:
            st.metric(
                "全作品を含む人",
                f"{rarity_result['complete_match_count']}人",
            )

        st.progress(rarity_result["rarity_score"])
        st.write(rarity_message(rarity_result["rarity_score"]))

        st.caption(
            "異端度は、作品名の直接一致を60％、"
            "item2vecによる作品構成の近さを40％として"
            "回答者ごとの近さを計算し、"
            "近い上位3人の平均から算出しています。"
        )

        with st.expander("あなたに近い回答者ランキング"):
            closest_df = pd.DataFrame(
                rarity_result["closest_results"]
            )
            closest_df.insert(
                0,
                "順位",
                range(1, len(closest_df) + 1),
            )

            st.dataframe(
                closest_df[
                    [
                        "順位",
                        "一致数",
                        "直接一致率",
                        "作品構成の近さ",
                        "総合的な近さ",
                        "一致した作品",
                        "回答作品",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "直接一致率": st.column_config.ProgressColumn(
                        "直接一致率",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "作品構成の近さ": st.column_config.ProgressColumn(
                        "作品構成の近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    "総合的な近さ": st.column_config.ProgressColumn(
                        "総合的な近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                },
            )


# ==================================================
# 友達と比較
# ==================================================


elif page == "🤝 友達と比較":
    render_page_header(
        "🤝 友達とのマンガ嗜好比較",
        "2人が選んだ作品名の一致と、作品構成全体の近さを比較します。",
    )

    col1, col2 = st.columns(2)

    with col1:
        first_person = st.multiselect(
            "あなたの好きなマンガ（2〜5作品）",
            all_titles,
            max_selections=5,
            key="friend_first",
        )

    with col2:
        second_person = st.multiselect(
            "友達の好きなマンガ（2〜5作品）",
            all_titles,
            max_selections=5,
            key="friend_second",
        )

    if len(first_person) < 2 or len(second_person) < 2:
        st.info("それぞれ2作品以上選ぶと結果が表示されます。")
    else:
        comparison = compare_two_people(
            first_person,
            second_person,
            preference_model,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "総合一致度",
                f"{comparison['overall_similarity'] * 100:.0f}%",
            )

        with col2:
            st.metric(
                "作品名の一致度",
                f"{comparison['exact_similarity'] * 100:.0f}%",
            )

        with col3:
            st.metric(
                "作品構成の近さ",
                f"{comparison['vector_similarity'] * 100:.0f}%",
            )

        st.progress(comparison["overall_similarity"])

        st.write(
            "**共通して好きな作品：** "
            + (
                " / ".join(comparison["shared_titles"])
                if comparison["shared_titles"]
                else "なし"
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**あなたにだけある作品**")
            st.write(
                " / ".join(comparison["first_only"])
                or "なし"
            )

        with col2:
            st.write("**友達にだけある作品**")
            st.write(
                " / ".join(comparison["second_only"])
                or "なし"
            )

        st.caption(
            "総合一致度は、作品名のJaccard類似度50％と、"
            "item2vecによる作品構成の近さ50％で計算しています。"
        )


# ==================================================
# ガチャ・発掘
# ==================================================


elif page == "🎲 ガチャ・発掘":
    render_page_header(
        "🎲 マンガガチャ・作品発掘",
        "偶然の出会いを楽しむガチャと、回答人数の少ない作品を探す発掘ページです。",
    )

    available_titles = [
        title
        for title in all_titles
        if title not in excluded_titles
    ]

    minor_titles = [
        title
        for title in available_titles
        if popularity[title] <= 2
    ]

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🎲 全作品からガチャ",
            use_container_width=True,
        ):
            st.session_state.gacha_title = random.choice(
                available_titles
            )

    with col2:
        if st.button(
            "💎 マイナー作品ガチャ",
            use_container_width=True,
        ):
            candidate_pool = minor_titles or available_titles
            st.session_state.gacha_title = random.choice(
                candidate_pool
            )

    if st.session_state.gacha_title:
        gacha_title = st.session_state.gacha_title

        st.success(
            f"## {gacha_title}\n\n"
            f"アンケートで選んだ人：{popularity[gacha_title]}人\n\n"
            f"人気順位：{popularity_rank[gacha_title]}位"
        )

        related = create_preference_recommendations(
            preference_model,
            responses,
            gacha_title,
            3,
            excluded_titles={gacha_title},
        )

        if not related.empty:
            st.write("**この作品と選ばれ方が近い作品**")
            st.dataframe(
                related[["マンガ", "好みの近さ"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "好みの近さ": st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    ),
                },
            )

        render_title_actions(gacha_title, "gacha")

    render_section_title("マイナー作品発掘")

    minor_df = pd.DataFrame(
        sorted(
            [
                {
                    "マンガ": title,
                    "回答人数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in minor_titles
            ],
            key=lambda item: (
                item["回答人数"],
                item["マンガ"],
            ),
        )
    )

    if minor_df.empty:
        st.info("回答人数2人以下の作品はありません。")
    else:
        st.dataframe(
            minor_df,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# ランキング
# ==================================================


elif page == "🏆 ランキング":
    render_page_header(
        "🏆 アンケート内ランキング",
        "アンケートで多く選ばれた作品と、まだ選んだ人が少ない作品を一覧で確認できます。",
    )

    ranking_size = int(
        st.number_input(
            "表示件数",
            min_value=5,
            max_value=min(50, len(all_titles)),
            value=min(20, len(all_titles)),
            step=1,
        )
    )

    popular_df = pd.DataFrame(
        popularity.most_common(ranking_size),
        columns=["マンガ", "回答人数"],
    )
    popular_df.insert(
        0,
        "順位",
        range(1, len(popular_df) + 1),
    )

    render_section_title("人気ランキング")
    st.dataframe(
        popular_df,
        use_container_width=True,
        hide_index=True,
    )

    render_section_title("発掘ランキング")
    st.write(
        "回答人数が少ない作品を、"
        "アンケート内で見つけにくい順に表示します。"
    )

    discovery_df = pd.DataFrame(
        sorted(
            [
                {
                    "マンガ": title,
                    "回答人数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in all_titles
            ],
            key=lambda item: (
                item["回答人数"],
                item["マンガ"],
            ),
        )[:ranking_size]
    )

    st.dataframe(
        discovery_df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# あとで読むリスト
# ==================================================


elif page == "🔖 あとで読むリスト":
    render_page_header(
        "🔖 あとで読むリスト",
        "気になった作品と、今後おすすめから外した作品を整理します。",
    )
    st.caption(
        "このリストは現在のブラウザセッション内だけで保存されます。"
    )

    if not st.session_state.read_later:
        st.info("まだ作品が登録されていません。")
    else:
        list_df = pd.DataFrame(
            [
                {
                    "マンガ": title,
                    "回答人数": popularity[title],
                    "人気順位": popularity_rank[title],
                }
                for title in st.session_state.read_later
            ]
        )

        st.dataframe(
            list_df,
            use_container_width=True,
            hide_index=True,
        )

        remove_title = st.selectbox(
            "リストから削除する作品",
            st.session_state.read_later,
        )

        if st.button("選択した作品を削除"):
            remove_from_state("read_later", remove_title)
            st.rerun()

    render_section_title("興味なしリスト")

    if not st.session_state.not_interested:
        st.info("興味なしに登録された作品はありません。")
    else:
        st.write(
            " / ".join(st.session_state.not_interested)
        )

        restore_title = st.selectbox(
            "興味なしを解除する作品",
            st.session_state.not_interested,
        )

        if st.button("興味なしを解除"):
            remove_from_state("not_interested", restore_title)
            st.rerun()
