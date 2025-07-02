import streamlit as st
from typing import Optional
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io

from analysis.utils import get_message_text
from utils.paths import WORDCLOUD_STOPWORD_PATH, WORDCLOUD_FONT_PATH

class WordCloudGenerator:
    def __init__(self, 
                 font_path: str = WORDCLOUD_FONT_PATH, 
                 stopwords_path: str = WORDCLOUD_STOPWORD_PATH, 
                 min_length: int = 2):
        self.font_path = font_path
        self.stopwords = self.load_stopwords(stopwords_path)
        self.min_length = min_length
        self.okt = Okt()

    def load_stopwords(self, path: str) -> set:
        df = pd.read_csv(path)
        return set(df["word"].to_list())

    def extract_nouns(self, text: str) -> list:
        """extract nouns under conditions"""
        return [word for word in self.okt.nouns(text) if len(word) >= self.min_length]

    def clean_text(self, nouns: list) -> str:
        """create text for wordcloud after remove stopwords"""
        return " ".join([word for word in nouns if word not in self.stopwords])

    def create_wordcloud(self, text: str) -> WordCloud:
        return WordCloud(
            width=800, height=400, background_color="white",
            stopwords=self.stopwords, font_path=self.font_path
        ).generate(text)

    def save_to_buffer(self, wordcloud: WordCloud) -> io.BytesIO:
        """save wordcloud image into image buffer"""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return buffer
    
    def generate_wordcloud(self, text: str) -> tuple[io.BytesIO, str]:
        nouns = self.extract_nouns(text)
        cleaned = self.clean_text(nouns)
        wc = self.create_wordcloud(cleaned)
        buffer = self.save_to_buffer(wc)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("UTF-8")
        return buffer, image_base64
