import re
import unicodedata
from collections import Counter, defaultdict

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
    "アンケート結果を使い、共起ベースとitem2vecの"
    "2種類の方法でおすすめマンガを表示します。"
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
# CSVの読み込み
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
# item2vecモデルの学習
# ==================================================

@st.cache_resource
def train_item2vec(sentences):
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
# 共起ベースの推薦
# ==================================================

def create_cooccurrence_recommendations(
    responses,
    selected_title,
):
    selected_responses = [
        titles
        for titles in responses
        if selected_title in titles
    ]

    cooccurrence_counts = Counter()

    for titles in selected_responses:
        for title in titles:
            if title != selected_title:
                cooccurrence_counts[title] += 1

    results = []

    for title, together_count in cooccurrence_counts.items():
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
# item2vecによる推薦
# ==================================================

def create_item2vec_recommendations(
    model,
    responses,
    selected_title,
    result_count,
):
    if selected_title not in model.wv.key_to_index:
        return pd.DataFrame()

    vocabulary_size = len(model.wv.index_to_key)

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
            selected_title in titles and title in titles
            for titles in responses
        )

        results.append(
            {
                "マンガ": title,
                "item2vec類似度": float(similarity),
                "一緒に選んだ人数": together_count,
            }
        )

    return pd.DataFrame(results).head(result_count)


# ==================================================
# データとモデルの準備
# ==================================================

responses = load_survey()

item2vec_sentences = tuple(
    tuple(titles)
    for titles in responses
)

item2vec_model = train_item2vec(
    item2vec_sentences
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


# ==================================================
# 画面
# ==================================================

selected_title = st.selectbox(
    "好きなマンガを選んでください",
    all_titles,
)

result_count = st.slider(
    "表示するおすすめ件数",
    min_value=1,
    max_value=20,
    value=10,
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
        "共起ベース",
        "item2vec",
    ]
)


# 共起ベース
with tab1:
    st.subheader(
        f"「{selected_title}」を選んだ人のおすすめ"
    )

    st.write(
        "同じ回答者が一緒に選んだ作品をもとに推薦します。"
    )

    cooccurrence_results = (
        create_cooccurrence_recommendations(
            responses,
            selected_title,
        )
    )

    if cooccurrence_results.empty:
        st.warning(
            "このマンガと一緒に選ばれた作品がありません。"
        )

    else:
        display_df = cooccurrence_results.head(
            result_count
        ).copy()

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
                "順位": st.column_config.NumberColumn(
                    "順位",
                    format="%d",
                ),
                "マンガ": "おすすめマンガ",
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


# item2vec
with tab2:
    st.subheader(
        f"item2vecによる「{selected_title}」のおすすめ"
    )

    st.write(
        "5作品の組み合わせを学習し、"
        "ベクトルが近い作品を推薦します。"
    )

    item2vec_results = (
        create_item2vec_recommendations(
            item2vec_model,
            responses,
            selected_title,
            result_count,
        )
    )

    if item2vec_results.empty:
        st.warning(
            "item2vecの推薦結果を作成できませんでした。"
        )

    else:
        item2vec_results.insert(
            0,
            "順位",
            range(1, len(item2vec_results) + 1),
        )

        st.dataframe(
            item2vec_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "順位": st.column_config.NumberColumn(
                    "順位",
                    format="%d",
                ),
                "マンガ": "おすすめマンガ",
                "item2vec類似度":
                    st.column_config.NumberColumn(
                        "item2vec類似度",
                        format="%.3f",
                        help=(
                            "1に近いほど、item2vec上で"
                            "似た作品として学習されています。"
                        ),
                    ),
                "一緒に選んだ人数":
                    st.column_config.NumberColumn(
                        "実際に一緒に選んだ人数",
                        format="%d人",
                    ),
            },
        )


# ==================================================
# 人気作品
# ==================================================

st.divider()
st.subheader("アンケート内で人気のマンガ")

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
    range(1, len(popular_df) + 1),
)

st.dataframe(
    popular_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "順位": st.column_config.NumberColumn(
            "順位",
            format="%d",
        ),
        "マンガ": "マンガ",
        "回答人数": st.column_config.NumberColumn(
            "回答人数",
            format="%d人",
        ),
    },
)
