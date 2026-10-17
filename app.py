"""競艇予想の画面（Streamlit）。  起動: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="競艇予想", page_icon="🚤", layout="wide")
st.navigation([
    st.Page("views/yosou.py", title="予想", icon="🚤", default=True),
    st.Page("views/guide.py", title="見方ガイド", icon="📖"),
]).run()
