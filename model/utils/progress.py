import logging
import streamlit as st


class BaseProgress:

    def update(self, step, text):
        pass

    def finish(self):
        pass


class StreamlitProgress(BaseProgress):

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.bar = st.progress(0)

    def update(self, step, text):
        percent = int((step / self.total_steps) * 100)
        self.bar.progress(percent, text=text)

    def finish(self):
        self.bar.progress(100, text="Completado")


class ConsoleProgress(BaseProgress):

    def __init__(self, total_steps):
        self.total_steps = total_steps

    def update(self, step, text):
        logging.info("[%d/%d] %s", step, self.total_steps, text)

    def finish(self):
        logging.info("Completado")


class NullProgress(BaseProgress):

    pass


def get_progress_bar(
    total_steps,
    mode="console"
):
    if mode == "streamlit":
        return StreamlitProgress(total_steps)

    if mode == "console":
        return ConsoleProgress(total_steps)

    return NullProgress()