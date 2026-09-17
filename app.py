
import os
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Egypt AI Studio",
    page_icon="🇪🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: "Cairo", sans-serif;
}
.stApp { background: #f7f8fc; }
.block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #263244; }
[data-testid="stSidebar"] * { color: #f8fafc !important; }

.brand {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #334155;
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 18px;
}
.brand-title { font-size: 28px; font-weight: 800; }
.brand-sub { color: #cbd5e1; font-size: 12px; margin-top: 3px; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #312e81 100%);
    color: white;
    border-radius: 28px;
    padding: 34px 38px;
    margin-bottom: 22px;
    box-shadow: 0 18px 45px rgba(15, 23, 42, .16);
}
.hero h1 { margin: 0 0 8px 0; font-size: 38px; font-weight: 800; }
.hero p { margin: 0; color: #dbeafe; font-size: 15px; }

.card, .result-card, .metric {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(15, 23, 42, .05);
}
.card { padding: 20px; margin-bottom: 16px; }
.metric { padding: 18px; min-height: 112px; }
.metric .label { color: #64748b; font-size: 13px; }
.metric .value { color: #0f172a; font-size: 30px; font-weight: 800; margin-top: 5px; }

.section-title { font-size: 23px; font-weight: 800; margin: 12px 0; color: #0f172a; }
.small-muted { color: #64748b; font-size: 13px; }
.result-card { padding: 18px; margin: 10px 0; }
.result-title { font-size: 17px; font-weight: 800; color: #111827; }
.result-meta { color: #64748b; font-size: 12px; margin-top: 5px; }

button[kind="primary"] { border-radius: 12px !important; font-weight: 800 !important; }
div[data-testid="stTabs"] button { font-weight: 700; }
textarea, input { border-radius: 12px !important; }
.footer { text-align: center; color: #94a3b8; font-size: 12px; padding: 30px 0 5px; }
</style>
""",
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        return str(value) if value else default
    except Exception:
        return default


def get_openai_client(api_key: str) -> Optional[OpenAI]:
    key = (api_key or "").strip()
    return OpenAI(api_key=key) if key else None


def call_ai(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    client = get_openai_client(api_key)
    if not client:
        raise ValueError("أدخل مفتاح OpenAI API أولاً.")

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
    )
    text = getattr(response, "output_text", None)
    return text.strip() if text else str(response)


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.S)
        if match:
            return json.loads(match.group(1))
        raise


def ai_json(api_key: str, model: str, system_prompt: str, user_prompt: str) -> Any:
    return extract_json(call_ai(api_key, model, system_prompt, user_prompt))


def youtube_search(
    api_key: str,
    query: str,
    max_results: int = 10,
    days: int = 30,
) -> List[Dict[str, Any]]:
    if not api_key.strip():
        raise ValueError("أدخل YouTube Data API Key لاستخدام البحث الحقيقي.")

    published_after = (
        datetime.utcnow() - timedelta(days=max(days, 1))
    ).isoformat(timespec="seconds") + "Z"

    search_response = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "viewCount",
            "maxResults": max_results,
            "publishedAfter": published_after,
            "key": api_key,
        },
        timeout=25,
    )
    search_response.raise_for_status()
    data = search_response.json()

    video_ids = [
        item["id"]["videoId"]
        for item in data.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return []

    stats_response = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=25,
    )
    stats_response.raise_for_status()
    stats = {
        item["id"]: item.get("statistics", {})
        for item in stats_response.json().get("items", [])
    }

    results = []
    for item in data.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue

        snippet = item.get("snippet", {})
        stat = stats.get(video_id, {})
        results.append(
            {
                "id": video_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published": snippet.get("publishedAt", ""),
                "views": int(stat.get("viewCount", 0)),
                "likes": int(stat.get("likeCount", 0)),
                "comments": int(stat.get("commentCount", 0)),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )
    return results


def human_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "library" not in st.session_state:
    st.session_state.library = []

openai_key = st.sidebar.text_input(
    "🔐 OpenAI API Key",
    value=get_secret("OPENAI_API_KEY"),
    type="password",
    help="الأفضل وضع المفتاح في Streamlit Secrets وعدم كتابته داخل app.py.",
)
youtube_key = st.sidebar.text_input(
    "▶️ YouTube Data API Key",
    value=get_secret("YOUTUBE_API_KEY"),
    type="password",
)

model = st.sidebar.selectbox(
    "🧠 AI Model",
    ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
)
language = st.sidebar.selectbox(
    "🌐 لغة المحتوى",
    ["العربية", "English", "عربي + English"],
)
platform = st.sidebar.selectbox(
    "📱 المنصة",
    ["YouTube", "YouTube Shorts", "TikTok", "Instagram Reels", "Facebook"],
)
tone = st.sidebar.selectbox(
    "🎙️ أسلوب المحتوى",
    ["احترافي", "قصصي", "فضولي ومشوق", "تعليمي", "مرح", "سينمائي"],
)

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-title">🇪🇬 EGYPT</div>
        <div class="brand-sub">AI Content Intelligence Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🇪🇬 Egypt AI Studio</h1>
        <p>ابحثي، حللي، ولّدي، اكتبي، واحفظي — في مساحة واحدة لصناعة المحتوى.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "🏠 Dashboard",
    "🔥 Trend Intelligence",
    "🎯 Keyword Intelligence",
    "💡 Idea Generator",
    "🎬 Script Studio",
    "🎨 Prompt Studio",
    "📺 YouTube SEO",
    "📚 Content Library",
]
page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")


if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">لوحة التحكم</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Saved Projects", str(len(st.session_state.library)))
    with c2:
        render_metric("AI Status", "Ready" if openai_key else "No Key")
    with c3:
        render_metric("YouTube Search", "Ready" if youtube_key else "API Needed")
    with c4:
        render_metric("Platform", platform)

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">🚀 ابدأي مشروعًا جديدًا</div>
                <p class="small-muted">ابدئي من موضوع واحد وحوّليه إلى سلسلة محتوى كاملة.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "موضوع المشروع",
            placeholder="مثال: قصص أطفال عن المغامرات",
        )
        a, b, c = st.columns(3)
        with a:
            st.button("🔥 تحليل تريند", use_container_width=True)
        with b:
            st.button("💡 توليد أفكار", use_container_width=True)
        with c:
            st.button("🎬 كتابة سكريبت", use_container_width=True)

        if topic:
            st.info("اختاري القسم المناسب من شريط التنقل بالأعلى لبناء المشروع.")

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">🧠 سير العمل</div>
                <p>1. Research</p><p>↓</p>
                <p>2. Analyze</p><p>↓</p>
                <p>3. Ideas</p><p>↓</p>
                <p>4. Script + Prompts</p><p>↓</p>
                <p>5. SEO + Library</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "🔥 Trend Intelligence":
    st.markdown('<div class="section-title">🔥 Trend Intelligence</div>', unsafe_allow_html=True)
    st.caption("البحث هنا يعتمد على YouTube Data API عند توفر المفتاح؛ لا توجد نتائج أو أرقام وهمية.")

    topic = st.text_input("موضوع البحث", placeholder="مثال: kids stories / AI tools / Egyptian recipes")
    c1, c2 = st.columns(2)
    with c1:
        days = st.selectbox("الفترة", [7, 14, 30, 90], index=2)
    with c2:
        limit = st.slider("عدد النتائج", 5, 25, 10)

    if st.button("🔍 ابحث وحلل الآن", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("اكتبي موضوعًا أولاً.")
        elif not youtube_key.strip():
            st.warning("أضيفي YouTube Data API Key من الشريط الجانبي.")
        else:
            try:
                with st.spinner("جاري البحث في YouTube..."):
                    results = youtube_search(youtube_key, topic, limit, days)

                if not results:
                    st.info("لم يتم العثور على نتائج.")
                else:
                    total_views = sum(x["views"] for x in results)
                    avg_views = int(total_views / len(results))
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        render_metric("Videos Found", str(len(results)))
                    with c2:
                        render_metric("Total Views", human_number(total_views))
                    with c3:
                        render_metric("Average Views", human_number(avg_views))

                    for item in results:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-title">{item["title"]}</div>
                                <div class="result-meta">
                                    {item["channel"]} • 👁️ {human_number(item["views"])}
                                    • 👍 {human_number(item["likes"])}
                                    • 💬 {human_number(item["comments"])}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"[فتح الفيديو على YouTube]({item['url']})")

                    if openai_key and st.button("🧠 حلّل الأنماط والفرص"):
                        payload = json.dumps(results, ensure_ascii=False, indent=2)
                        prompt = f"""
حلل نتائج YouTube التالية لموضوع "{topic}".
لا تخترع أرقامًا.
استخرج:
1) أنماط العناوين.
2) الزوايا المتكررة.
3) فجوات المحتوى.
4) 10 أفكار أصلية.
5) Hooks.
6) توصيات عملية.
البيانات:
{payload}
"""
                        with st.spinner("جاري التحليل..."):
                            analysis = call_ai(
                                openai_key,
                                model,
                                "أنت محلل محتوى دقيق. لا تختلق بيانات غير موجودة.",
                                prompt,
                            )
                        st.markdown("### 🧠 تحليل Egypt")
                        st.markdown(analysis)

            except requests.HTTPError as exc:
                st.error(f"YouTube API error: {exc}")
            except Exception as exc:
                st.error(f"حدث خطأ: {exc}")


elif page == "🎯 Keyword Intelligence":
    st.markdown('<div class="section-title">🎯 Keyword Intelligence</div>', unsafe_allow_html=True)
    topic = st.text_input("الموضوع أو الكلمة الأساسية", placeholder="مثال: AI video generator")

    if st.button("🧠 بناء خريطة الكلمات", type="primary"):
        if not openai_key:
            st.warning("أضيفي OpenAI API Key.")
        elif not topic.strip():
            st.warning("اكتبي موضوعًا.")
        else:
            prompt = f"""
الموضوع: {topic}
المنصة: {platform}
اللغة: {language}

أنشئ خريطة كلمات مفتاحية مخصصة.
أعد JSON فقط:
{{
 "primary_keywords": [],
 "long_tail_keywords": [],
 "questions": [],
 "content_clusters": [],
 "search_intents": [],
 "content_angles": []
}}
لا تخترع أرقام search volume.
"""
            try:
                with st.spinner("جاري بناء الخريطة..."):
                    data = ai_json(
                        openai_key,
                        model,
                        "أنت خبير SEO وتحليل نية البحث.",
                        prompt,
                    )
                for key, value in data.items():
                    st.markdown(f"### {key.replace('_', ' ').title()}")
                    if isinstance(value, list):
                        for item in value:
                            st.markdown(f"- {item}")
                    else:
                        st.write(value)
            except Exception as exc:
                st.error(f"تعذر إنشاء الخريطة: {exc}")


elif page == "💡 Idea Generator":
    st.markdown('<div class="section-title">💡 Idea Generator</div>', unsafe_allow_html=True)
    topic = st.text_input("موضوع / مجال المحتوى", placeholder="مثال: مغامرات حيوانات للأطفال")
    count = st.slider("عدد الأفكار", 5, 20, 10)
    audience = st.text_input("الجمهور", value="Kids / Parents")

    if st.button("✨ توليد فرص محتوى", type="primary", use_container_width=True):
        if not openai_key:
            st.warning("أضيفي OpenAI API Key.")
        elif not topic.strip():
            st.warning("اكتبي موضوعًا.")
        else:
            prompt = f"""
أنت مدير محتوى محترف.
الموضوع: {topic}
الجمهور: {audience}
المنصة: {platform}
اللغة: {language}
الأسلوب: {tone}

ولّد {count} أفكار أصلية.
أعد JSON فقط:
[
 {{
  "title": "...",
  "hook": "...",
  "angle": "...",
  "format": "...",
  "audience": "...",
  "retention_strategy": "...",
  "thumbnail_concept": "..."
 }}
]
لا تكرر الأفكار.
"""
            try:
                with st.spinner("جاري ابتكار الأفكار..."):
                    ideas = ai_json(
                        openai_key,
                        model,
                        "أنت استراتيجي محتوى. ابتكر أفكارًا عملية ومحددة.",
                        prompt,
                    )
                for idx, idea in enumerate(ideas, 1):
                    with st.expander(f"{idx}. {idea.get('title', 'Idea')}"):
                        st.write(f"**Hook:** {idea.get('hook', '')}")
                        st.write(f"**Angle:** {idea.get('angle', '')}")
                        st.write(f"**Format:** {idea.get('format', '')}")
                        st.write(f"**Audience:** {idea.get('audience', '')}")
                        st.write(f"**Retention:** {idea.get('retention_strategy', '')}")
                        st.write(f"**Thumbnail:** {idea.get('thumbnail_concept', '')}")
                        if st.button("💾 حفظ الفكرة", key=f"save_idea_{idx}"):
                            st.session_state.library.append(
                                {
                                    "type": "Idea",
                                    "title": idea.get("title", "Untitled"),
                                    "content": idea,
                                    "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                }
                            )
                            st.success("تم الحفظ.")
            except Exception as exc:
                st.error(f"تعذر التوليد: {exc}")


elif page == "🎬 Script Studio":
    st.markdown('<div class="section-title">🎬 Script Studio</div>', unsafe_allow_html=True)
    topic = st.text_input("موضوع الفيديو", placeholder="اكتب الفكرة بالتفصيل")
    duration = st.selectbox("المدة", ["30 sec", "60 sec", "3 min", "5 min", "10 min", "15+ min"])
    goal = st.selectbox("هدف الفيديو", ["Views", "Education", "Storytelling", "Sales", "Subscribers"])

    if st.button("🎬 Build Full Script", type="primary", use_container_width=True):
        if not openai_key:
            st.warning("أضيفي OpenAI API Key.")
        elif not topic.strip():
            st.warning("اكتبي موضوع الفيديو.")
        else:
            prompt = f"""
اكتب سكريبت احترافي كامل عن: {topic}
المنصة: {platform}
المدة: {duration}
الهدف: {goal}
اللغة: {language}
الأسلوب: {tone}

قسّم الناتج إلى:
Hook
Open Loop
Main Sections
Retention Beats
Visual Direction
Voiceover
CTA
Title Ideas

اجعل كل جزء مرتبطًا بالموضوع نفسه.
"""
            try:
                with st.spinner("جاري بناء السكريبت..."):
                    script = call_ai(
                        openai_key,
                        model,
                        "أنت كاتب سيناريو ومحرر فيديو محترف.",
                        prompt,
                    )
                st.markdown(script)
                st.download_button(
                    "⬇️ تنزيل السكريبت TXT",
                    data=script,
                    file_name="egypt_script.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                if st.button("💾 حفظ السكريبت"):
                    st.session_state.library.append(
                        {
                            "type": "Script",
                            "title": topic,
                            "content": script,
                            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                    )
                    st.success("تم الحفظ.")
            except Exception as exc:
                st.error(f"تعذر إنشاء السكريبت: {exc}")


elif page == "🎨 Prompt Studio":
    st.markdown('<div class="section-title">🎨 Prompt Studio</div>', unsafe_allow_html=True)
    description = st.text_area(
        "صف الشخصية / المشهد / الصورة",
        placeholder="مثال: cute 3D baby fox with a blue backpack in a magical forest",
        height=140,
    )
    prompt_type = st.selectbox(
        "نوع الـPrompt",
        ["Character Sheet", "Image Prompt", "Video Prompt", "Thumbnail Prompt"],
    )

    if st.button("🎨 Generate Professional Prompt", type="primary", use_container_width=True):
        if not openai_key:
            st.warning("أضيفي OpenAI API Key.")
        elif not description.strip():
            st.warning("اكتبي وصفًا.")
        else:
            prompt = f"""
حوّل الوصف التالي إلى Prompt احترافي باللغة الإنجليزية:
{description}

نوع الـPrompt: {prompt_type}
أضف composition, lighting, camera, visual style, subject details,
وcharacter consistency عند الحاجة.
أخرج الـPrompt فقط.
"""
            try:
                with st.spinner("جاري تحسين الـPrompt..."):
                    result = call_ai(
                        openai_key,
                        model,
                        "أنت Prompt Engineer متخصص في الصور والفيديو.",
                        prompt,
                    )
                st.code(result, language="text")
                st.download_button(
                    "⬇️ تنزيل الـPrompt",
                    data=result,
                    file_name="egypt_prompt.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"تعذر توليد الـPrompt: {exc}")


elif page == "📺 YouTube SEO":
    st.markdown('<div class="section-title">📺 YouTube SEO Studio</div>', unsafe_allow_html=True)
    topic = st.text_input("موضوع الفيديو")
    details = st.text_area("ملخص المحتوى", height=120)

    if st.button("🚀 Build YouTube SEO Pack", type="primary", use_container_width=True):
        if not openai_key:
            st.warning("أضيفي OpenAI API Key.")
        elif not topic.strip():
            st.warning("اكتبي الموضوع.")
        else:
            prompt = f"""
أنشئ YouTube SEO Pack مخصصًا.

Topic: {topic}
Details: {details}
Language: {language}

أعد JSON فقط:
{{
 "titles": [],
 "description": "",
 "keywords": [],
 "tags": [],
 "hashtags": [],
 "thumbnail_text": [],
 "thumbnail_concept": ""
}}
لا تخترع search volume أو أرقامًا.
"""
            try:
                with st.spinner("جاري بناء SEO Pack..."):
                    seo = ai_json(
                        openai_key,
                        model,
                        "أنت خبير YouTube SEO. ركز على relevance وclarity وCTR دون ادعاءات زائفة.",
                        prompt,
                    )

                st.markdown("### 🎯 Titles")
                for title in seo.get("titles", []):
                    st.markdown(f"- **{title}**")

                st.markdown("### 📝 Description")
                st.text_area("Description", seo.get("description", ""), height=220)

                st.markdown("### 🔑 Keywords")
                st.write(", ".join(seo.get("keywords", [])))

                st.markdown("### 🏷️ Tags")
                st.code(", ".join(seo.get("tags", [])))

                st.markdown("### #️⃣ Hashtags")
                st.write(" ".join(seo.get("hashtags", [])))

                st.markdown("### 🖼️ Thumbnail")
                st.write(seo.get("thumbnail_concept", ""))
                st.write(", ".join(seo.get("thumbnail_text", [])))
            except Exception as exc:
                st.error(f"تعذر إنشاء SEO Pack: {exc}")


elif page == "📚 Content Library":
    st.markdown('<div class="section-title">📚 Content Library</div>', unsafe_allow_html=True)

    if not st.session_state.library:
        st.info("المكتبة فارغة. احفظي أول فكرة أو سكريبت.")
    else:
        for idx, item in enumerate(reversed(st.session_state.library)):
            with st.expander(
                f"{item.get('type', 'Content')} — {item.get('title', 'Untitled')} — {item.get('created', '')}"
            ):
                content = item.get("content", "")
                if isinstance(content, (dict, list)):
                    st.json(content)
                    export_data = json.dumps(content, ensure_ascii=False, indent=2)
                    extension = "json"
                else:
                    st.markdown(str(content))
                    export_data = str(content)
                    extension = "txt"

                st.download_button(
                    "⬇️ Export",
                    data=export_data,
                    file_name=f"egypt_content_{idx}.{extension}",
                    mime="application/json" if extension == "json" else "text/plain",
                    key=f"download_library_{idx}",
                )

        if st.button("🗑️ مسح المكتبة"):
            st.session_state.library = []
            st.rerun()

st.markdown(
    '<div class="footer">🇪🇬 Egypt AI Studio — Intelligent Content Workflow</div>',
    unsafe_allow_html=True,
)
