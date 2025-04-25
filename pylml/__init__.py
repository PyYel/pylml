"""
PyYel packages initializer
"""

import os, sys
from dotenv import load_dotenv

def login(hf_token: str = None, dotenv_path: str = None):
    """
    PyYel log in helper.
    Loads the relevant credentials to your environment variables. Those should be removed
    """

    if dotenv_path: load_dotenv(dotenv_path=dotenv_path)
    if hf_token: os.environ["HF_TOKEN"] = hf_token

    return True

def logout():
    """
    PyYel log in helper.
    Loads the relevant credentials to your environment variables. Those should be removed
    """

    if "HF_TOKEN" in os.environ.keys(): del os.environ["HF_TOKEN"]

    return True