import io
import time
import json

import networkx as nx

from rich.console import Console


debug = Console()
console = Console()
log_buffer = io.StringIO()
logger = Console(file=log_buffer, width=120, record=True)

def create_tag():   
    return time.strftime("%y%m%d_%H%M", time.localtime())

def log_and_print(text):
    console.print(text)
    logger.print(text)

def log_and_input(text):
    input = console.input(text)
    logger.print(f"{text}{input}")

    return input

def save_log(path):
    logger.save_html(path)