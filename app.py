import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec


st.set_page_config(
    page_title="マンガレコメンド",
    page_icon="📚",
    layout="wide",
)

st.title("📚 マンガレコメンド")
st.write(
    "アンケート結果を使い、2つの方法でおすすめマンガを表示します。"
)


# ==================================================
# 作品名の表記統一
# ==================================================

def title_key(title):
    text = unicodedata.normalize(
        "NFKC",
        str(title),
    ).strip()

    text = text.replace("×", "x")
    text = re.sub(
        r"[\W_]+",
        "",
        text,
        flags=re.UNICODE,
    )

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

    "魔入りました入間くん":
        "魔入りました！入間くん",
    "魔入りました!入間くん":
        "魔入りました！入間くん",
    "魔入りました！入間くん":
        "魔入りました！入間くん",

    "僕のヒーローアカデミア":
        "僕のヒーローアカデミア",
    "ヒロアカ":
        "僕のヒーローアカデミア",

    "転生したらスライムだった件":
        "転生したらスライムだった件",
    "転スラ":
        "転生したらスライムだった件",

    "鬼滅":
        "鬼滅の刃",
    "鬼滅の刃":
        "鬼滅の刃",
}


TITLE_ALIASES = {
    title_key(original): canonical
    for original, canonical
    in RAW_TITLE_ALIASES.items()
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

    title = unicodedata.normalize(
        "NFKC",
        str(value),
    ).strip()

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

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
        st.error(
            "CSV内に「マンガ」を含む列がありません。"
        )
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
        canonical_titles[key] = (
            counts.most_common(1)[0][0]
        )

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
# 好みの近さを学習するモデル
# 技術的にはitem2vec
# ==================================================

@st.cache_resource
def train_preference_model(sentences):
    training_data = [
        list(sentence)
        for sentence in sentences
    ]

    model = Word2Vec(
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

    return model


# ==================================================
# 同じ回答者が選んだ作品から推薦
# 技術的には共起ベース
# ==================================================

def create_group_recommendations(
    responses,
    selected_title,
):
    selected_responses = [
        titles
        for titles in responses
        if selected_title in titles
    ]

    together_counts = Counter()

    for titles in selected_responses:
        for title in titles:
            if title != selected_title:
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


# ==================================================
# 学習した好みの近さから推薦
# 技術的にはitem2vec
# ==================================================

def create_preference_recommendations(
    model,
    responses,
    selected_title,
    result_count,
):
    if selected_title not in model.wv.key_to_index:
        return pd.DataFrame()

    vocabulary_size = len(
        model.wv.index_to_key
    )

    if vocabulary_size <= 1:
        return pd.DataFrame()

    candidate_count = min(
        max(result_count * 3, 20),
        vocabulary_size - 1,
    )

    similar_titles = model.wv.most_similar(
        positive=[selected_title],
        topn=candidate_count,
    )

    results = []

    for title, similarity in similar_titles:
        together_count = sum(
            selected_title in titles
            and title in titles
            for titles in responses
        )

        results.append(
            {
                "マンガ": title,
                "好みの近さ": float(similarity),
                "実際に一緒に選んだ人数":
                    together_count,
            }
        )

    return pd.DataFrame(results).head(
        result_count
    )


# ==================================================
# 異端度の計算
# ==================================================

DIRECT_MATCH_WEIGHT = 0.60
ITEM2VEC_WEIGHT = 0.40

REFERENCE_NEIGHBORS = 3
CLOSE_SIMILARITY_THRESHOLD = 0.70


def create_average_vector(
    model,
    titles,
):
    valid_titles = [
        title
        for title in titles
        if title in model.wv.key_to_index
    ]

    if not valid_titles:
        return None

    vectors = [
        model.wv.get_vector(title)
        for title in valid_titles
    ]

    return np.mean(
        vectors,
        axis=0,
    )


def cosine_similarity(
    first_vector,
    second_vector,
):
    if first_vector is None or second_vector is None:
        return 0.0

    first_norm = np.linalg.norm(
        first_vector
    )

    second_norm = np.linalg.norm(
        second_vector
    )

    if first_norm == 0 or second_norm == 0:
        return 0.0

    similarity = float(
        np.dot(
            first_vector,
            second_vector,
        )
        / (
            first_norm
            * second_norm
        )
    )

    # 負の値は「近くない」とみなし、0にする
    return max(
        0.0,
        min(1.0, similarity),
    )


def analyze_rarity(
    responses,
    selected_titles,
    model,
):
    selected_set = set(
        selected_titles
    )

    selected_count = len(
        selected_set
    )

    if selected_count < 2:
        return None

    selected_vector = create_average_vector(
        model,
        selected_titles,
    )

    comparison_results = []

    for index, response_titles in enumerate(
        responses
    ):
        response_set = set(
            response_titles
        )

        matched_titles = (
            selected_set
            & response_set
        )

        exact_match_count = len(
            matched_titles
        )

        # ユーザーが選んだ作品のうち、
        # 何作品が回答者と直接一致したか
        direct_match_ratio = (
            exact_match_count
            / selected_count
        )

        response_vector = create_average_vector(
            model,
            response_titles,
        )

        # 一致しない作品も含めた、
        # 回答全体のitem2vec上の近さ
        item2vec_similarity = cosine_similarity(
            selected_vector,
            response_vector,
        )

        combined_similarity = (
            DIRECT_MATCH_WEIGHT
            * direct_match_ratio
            + ITEM2VEC_WEIGHT
            * item2vec_similarity
        )

        comparison_results.append(
            {
                "回答番号": index + 1,
                "一致数": exact_match_count,
                "直接一致率": direct_match_ratio,
                "作品構成の近さ":
                    item2vec_similarity,
                "総合的な近さ":
                    combined_similarity,
                "一致した作品": " / ".join(
                    sorted(matched_titles)
                ),
                "回答作品": " / ".join(
                    response_titles
                ),
                "選んだ作品をすべて含む":
                    selected_set.issubset(
                        response_set
                    ),
            }
        )

    if not comparison_results:
        return None

    comparison_results.sort(
        key=lambda result: result[
            "総合的な近さ"
        ],
        reverse=True,
    )

    neighbor_count = min(
        REFERENCE_NEIGHBORS,
        len(comparison_results),
    )

    closest_results = (
        comparison_results[
            :neighbor_count
        ]
    )

    # 最も近い上位3人との平均的な近さ
    neighborhood_similarity = (
        sum(
            result["総合的な近さ"]
            for result in closest_results
        )
        / neighbor_count
    )

    rarity_score = round(
        100
        * (
            1
            - neighborhood_similarity
        )
    )

    rarity_score = max(
        0,
        min(100, rarity_score),
    )

    complete_match_count = sum(
        result[
            "選んだ作品をすべて含む"
        ]
        for result in comparison_results
    )

    close_match_count = sum(
        result["総合的な近さ"]
        >= CLOSE_SIMILARITY_THRESHOLD
        for result in comparison_results
    )

    return {
        "selected_count":
            selected_count,
        "rarity_score":
            rarity_score,
        "best_similarity":
            comparison_results[0][
                "総合的な近さ"
            ],
        "neighborhood_similarity":
            neighborhood_similarity,
        "close_match_count":
            close_match_count,
        "complete_match_count":
            complete_match_count,
        "closest_results":
            closest_results,
    }


def rarity_message(score):
    if score <= 15:
        return (
            "アンケート内に、かなり近い"
            "マンガ嗜好の人がいます。"
        )

    if score <= 30:
        return (
            "比較的近いマンガ嗜好の人がいます。"
        )

    if score <= 50:
        return (
            "共通点はありますが、"
            "少し独自性のある組み合わせです。"
        )

    if score <= 70:
        return (
            "アンケート内では、"
            "かなり珍しい組み合わせです。"
        )

    return (
        "アンケート内では非常に珍しい"
        "マンガ嗜好です。"
    )


# ==================================================
# データとモデルの準備
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


st.info(
    f"回答者数：{len(responses)}人　／　"
    f"登録マンガ数：{len(all_titles)}作品"
)


with st.expander(
    "2つのおすすめ方法について"
):
    st.markdown(
        """
**みんなの回答からおすすめ**

選択したマンガを挙げた人が、ほかにどの作品を
選んでいたかを集計します。  
技術的には「共起ベース」の推薦です。

**好みの近さからおすすめ**

アンケートに記入された作品の組み合わせを学習し、
似た選ばれ方をしている作品を探します。  
技術的には「item2vec」を使用しています。
        """
    )


# ==================================================
# 通常のレコメンド
# ==================================================

st.subheader(
    "おすすめマンガを探す"
)

selected_title = st.selectbox(
    "好きなマンガを選んでください",
    all_titles,
)

result_count = st.number_input(
    "表示するおすすめ件数",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
)

result_count = int(
    result_count
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
        "👥 みんなの回答から",
        "🧭 好みの近さから",
    ]
)


with tab1:
    st.subheader(
        f"「{selected_title}」を選んだ人のおすすめ作品"
    )

    st.write(
        "同じマンガを好きな人が一緒に選んだ作品を"
        "おすすめします。"
    )

    group_results = (
        create_group_recommendations(
            responses,
            selected_title,
        )
    )

    if group_results.empty:
        st.warning(
            "このマンガと一緒に選ばれた"
            "作品がありません。"
        )

    else:
        display_df = group_results.head(
            result_count
        ).copy()

        display_df.insert(
            0,
            "順位",
            range(
                1,
                len(display_df) + 1,
            ),
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "順位":
                    st.column_config.NumberColumn(
                        "順位",
                        format="%d",
                    ),
                "マンガ":
                    "おすすめマンガ",
                "一緒に選んだ人数":
                    st.column_config.NumberColumn(
                        "一緒に選んだ人数",
                        format="%d人",
                    ),
                "おすすめ度":
                    st.column_config.ProgressColumn(
                        "おすすめ度",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
            },
        )


with tab2:
    st.subheader(
        f"「{selected_title}」と"
        "系統が近いおすすめ作品"
    )

    st.write(
        "アンケート内の選ばれ方を学習し、"
        "好みが近い作品をおすすめします。"
    )

    st.caption(
        "回答数が少ないため、結果は参考値です。"
    )

    preference_results = (
        create_preference_recommendations(
            preference_model,
            responses,
            selected_title,
            result_count,
        )
    )

    if preference_results.empty:
        st.warning(
            "おすすめ結果を作成できませんでした。"
        )

    else:
        preference_results.insert(
            0,
            "順位",
            range(
                1,
                len(preference_results) + 1,
            ),
        )

        st.dataframe(
            preference_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "順位":
                    st.column_config.NumberColumn(
                        "順位",
                        format="%d",
                    ),
                "マンガ":
                    "おすすめマンガ",
                "好みの近さ":
                    st.column_config.ProgressColumn(
                        "好みの近さ",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                        help=(
                            "1に近いほど、"
                            "アンケート内で似た選ばれ方を"
                            "している作品です。"
                        ),
                    ),
                "実際に一緒に選んだ人数":
                    st.column_config.NumberColumn(
                        "実際に一緒に選んだ人数",
                        format="%d人",
                    ),
            },
        )


# ==================================================
# 異端度ミニコーナー
# ==================================================

st.divider()

st.subheader(
    "🪐 マンガ嗜好異端度調査"
)

st.write(
    "好きなマンガを2〜5作品選ぶと、"
    "作品名の一致だけでなく、"
    "選ばれた作品全体の系統も考慮して"
    "異端度を判定します。"
)

selected_favorites = st.multiselect(
    "好きなマンガを2〜5作品選んでください",
    all_titles,
    max_selections=5,
    placeholder="作品名を検索して選択",
)

selected_favorite_count = len(
    selected_favorites
)

if selected_favorite_count < 2:
    st.info(
        f"あと{2 - selected_favorite_count}作品"
        "選ぶと結果が表示されます。"
    )

else:
    rarity_result = analyze_rarity(
        responses,
        selected_favorites,
        preference_model,
    )

    if rarity_result is not None:
        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:
            st.metric(
                "異端度",
                f"{rarity_result['rarity_score']} / 100",
            )

        with metric2:
            st.metric(
                "最も近い人との近さ",
                (
                    f"{rarity_result['best_similarity'] * 100:.0f}%"
                ),
            )

        with metric3:
            st.metric(
                "好みが近い人",
                f"{rarity_result['close_match_count']}人",
                help=(
                    "作品名の一致と作品構成の近さを"
                    "合わせた値が70％以上の回答者です。"
                ),
            )

        with metric4:
            st.metric(
                "選んだ作品をすべて含む人",
                f"{rarity_result['complete_match_count']}人",
            )

        st.progress(
            rarity_result["rarity_score"]
        )

        st.write(
            rarity_message(
                rarity_result["rarity_score"]
            )
        )

        st.caption(
            "異端度は、作品名の直接一致を60％、"
            "item2vecによる作品構成の近さを40％として"
            "回答者ごとの近さを計算し、"
            "最も近い上位3人の平均から算出しています。"
            f"対象は{len(responses)}人のアンケート回答です。"
        )

        with st.expander(
            "あなたに最も近い回答例を見る"
        ):
            closest_df = pd.DataFrame(
                rarity_result[
                    "closest_results"
                ]
            )

            st.dataframe(
                closest_df[
                    [
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
                    "一致数":
                        st.column_config.NumberColumn(
                            "一致数",
                            format="%d作品",
                        ),
                    "直接一致率":
                        st.column_config.ProgressColumn(
                            "直接一致率",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        ),
                    "作品構成の近さ":
                        st.column_config.ProgressColumn(
                            "作品構成の近さ",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        ),
                    "総合的な近さ":
                        st.column_config.ProgressColumn(
                            "総合的な近さ",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        ),
                },
            )


# ==================================================
# 人気作品
# ==================================================

st.divider()

st.subheader(
    "アンケート内で人気のマンガ"
)

popularity = Counter(
    title
    for titles in responses
    for title in titles
)

popular_df = pd.DataFrame(
    popularity.most_common(10),
    columns=[
        "マンガ",
        "回答人数",
    ],
)

popular_df.insert(
    0,
    "順位",
    range(
        1,
        len(popular_df) + 1,
    ),
)

st.dataframe(
    popular_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "順位":
            st.column_config.NumberColumn(
                "順位",
                format="%d",
            ),
        "マンガ":
            "マンガ",
        "回答人数":
            st.column_config.NumberColumn(
                "回答人数",
                format="%d人",
            ),
    },
)
