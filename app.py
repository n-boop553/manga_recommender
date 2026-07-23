import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="マンガレコメンド",
    page_icon="📚",
    layout="wide",
)

st.title("📚 マンガレコメンド")
st.write(
    "好きなマンガを選ぶと、同じ回答者が一緒に挙げたマンガをおすすめします。"
)


def title_key(title):
    """
    同一作品か比較するための文字列を作る。
    空白、記号、英字の大文字・小文字などを無視する。
    """
    text = unicodedata.normalize("NFKC", str(title)).strip()
    text = text.replace("×", "x")
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text.lower()


# 明示的に統一したい作品名
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

    "鬼滅の刃": "鬼滅の刃",
    "鬼滅": "鬼滅の刃",
}


# 辞書のキーにも同じ正規化処理をかける
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
    """
    CSV内の値を整形し、比較用キーと表示用タイトルを返す。
    """
    if pd.isna(value):
        return None

    title = unicodedata.normalize("NFKC", str(value)).strip()
    title = re.sub(r"\s+", " ", title)

    key = title_key(title)

    if not key or key in INVALID_ANSWERS:
        return None

    return key, title


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
            "CSV内に「マンガ」を含む列が見つかりません。"
            "列名を確認してください。"
        )
        st.stop()

    # 明示的な別名辞書にない作品について、
    # 同じ比較用キーを持つ表記を自動的にまとめる
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
        # 最も多く使われている表記を表示用タイトルにする
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

            # 同じ回答者が同一作品を重複記入した場合は1件にする
            if canonical_title not in titles:
                titles.append(canonical_title)

        if titles:
            responses.append(titles)

    return responses


def create_recommendations(responses, selected_title):
    """
    選択作品を挙げた人が、ほかに挙げていた作品を集計する。
    おすすめ度にはJaccard係数を使用する。
    """
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


responses = load_survey()

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

recommendations = create_recommendations(
    responses,
    selected_title,
)

st.subheader(
    f"「{selected_title}」が好きな人へのおすすめ"
)

if recommendations.empty:
    st.warning(
        "このマンガと一緒に選ばれた作品がありません。"
    )

else:
    display_df = recommendations.head(
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
            "一緒に選んだ人数": st.column_config.NumberColumn(
                "一緒に選んだ人数",
                format="%d人",
            ),
            "おすすめ度": st.column_config.ProgressColumn(
                "おすすめ度",
                help=(
                    "選択した作品と、ほかの作品が"
                    "同じ回答内に登場した割合です。"
                ),
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )


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
