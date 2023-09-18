import streamlit as st
from htbuilder import details, div, p, styles
from htbuilder import summary as smry
from typing import List, Dict
import re

def customstoggle(summary: str, content: list, metadata_keys: list):
    formatted_content = "<hr/>"
    for doc in content:
        page_content = doc.get("page_content", "")
        
        # Find URLs and replace them with HTML anchor tags
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', page_content)
        for url in urls:
            page_content = page_content.replace(url, f'<a href="{url}" target="_blank">{url}</a>')

        formatted_content += "<div><h3>Source Content:</h3><p>{}</p><h3>Metadata:</h3><ul>".format(page_content)
        for key in metadata_keys:
            value = doc.get("metadata", {}).get(key, 'N/A')
            if value != 'N/A':
                formatted_content += "<li>{}: {}</li>".format(key, value)
        formatted_content += "</ul></div><hr/>"
    
    # Using st.write() to inject raw HTML
    st.write(
        str(
            div(
                style=styles(
                    line_height=1.8,
                )
            )(details(smry(summary), p(formatted_content)))
        ),
        unsafe_allow_html=True,
    )


def example():
    customstoggle(
        "Click me!",
        """🥷 Surprise! Here's some additional content""",
    )


__title__ = "Toggle button"
__desc__ = "Toggle button just like in Notion!"
__icon__ = "➡️"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__github_repo__ = "arnaudmiribel/stoggle"
__streamlit_cloud_url__ = "http://stoggle.streamlitapp.com"
__experimental_playground__ = True
