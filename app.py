import re
import unicodedata
from collections import Counter

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="マンガレコメンド",
    page_icon="📚",
    layout="wide",
)

st.title("📚 マンガレコメンド")
st.write("好きなマンガを選ぶと、同じ回答者が一緒に挙げたマンガをおすすめします。")


# 表記揺れの統一
TITLE_ALIASES = {
    "ハイキュー!!": "ハイキュー",
    "ハイキュー!!!": "ハイキュー",
    "HUNTER×HUNTER": "HUNTER×HUNTER",
    "Hunter×Hunter": "HUNTER×HUNTER",
    "ハンターハンター": "HUNTER×HUNTER",
    "ONEPIECE": "ONE PIECE",
    "ワンピース": "ONE PIECE",
    "SLAMDUNK": "SLAM DUNK",
    "スラムダンク": "SLAM DUNK",
    "スパイファミリー": "SPY×FAMILY",
    "SPYXFAMILY": "SPY×FAMILY",
    "呪術回戦": "呪術廻戦",
    "デスノート": "DEATH NOTE",
    "コナン": "名探偵コナン",
    "魔入りました!入間くん": "魔入りました！入間くん",
    "魔入りました入間くん": "魔入りました！入間くん",
}


def normalize_title(title):
    if pd.isna(title):
        return None

    title = unicodedata.normalize("NFKC", str(title))
    title = title.strip()

    if not title or title in {"なし", "無し", "特になし", "ない"}:
        return None

    # 検索・統一用に余分な空白を削除
    compact_title = re.sub(r"\s+", "", title)

    if compact_title in TITLE_ALIASES:
        return TITLE_ALIASES[compact_title]

    if title in TITLE_ALIASES:
        return TITLE_ALIASES[title]

    return title


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
        if "マンガ" in column
    ]

    responses = []

    for _, row in df.iterrows():
        titles = []

        for column in manga_columns:
            normalized = normalize_title(row[column])

            if normalized is not None and normalized not in titles:
                titles.append(normalized)

        if titles:
            responses.append(titles)

    return responses


def create_recommendations(responses, selected_title):
    selected_users = [
        titles
        for titles in responses
        if selected_title in titles
    ]

    counts = Counter()

    for titles in selected_users:
        for title in titles:
            if title != selected_title:
                counts[title] += 1

    results = []

    for title, count in counts.most_common():
        title_users = sum(
            title in titles
            for titles in responses
        )

        union_count = (
            len(selected_users)
            + title_users
            - count
        )

        similarity = (
            count / union_count
            if union_count > 0
            else 0
        )

        results.append(
            {
                "マンガ": title,
                "一緒に選んだ人数": count,
                "おすすめ度": similarity,
            }
        )

    return pd.DataFrame(results)


responses = load_survey()

all_titles = sorted(
    {
        title
        for titles in responses
        for title in titles
    }
)

st.info(
    f"回答者数：{len(responses)}人　"
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

st.subheader(f"「{selected_title}」が好きな人へのおすすめ")

if recommendations.empty:
    st.warning(
        "このマンガと一緒に選ばれた作品がありません。"
    )
else:
    display_df = recommendations.head(result_count).copy()
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
    columns=["マンガ", "回答人数"],
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
)
