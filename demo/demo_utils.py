import glob
from pathlib import Path

import pandas as pd
from classifai.indexers.dataclasses import VectorStoreSearchInput
from classifai.vectorisers import FastEmbedVectoriser

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "mock_soc_dataset.csv"
KNOWLEDGEBASE_PATH = DATA_DIR / "mock_soc_knowledgebase.csv"
UNCODED_RESPONSES_PATH = DATA_DIR / "mock_uncoded_soc_responses.csv"


def load_uncoded_input():
    # Prepare queries - VectorStore search expects 'id' and 'query' columns
    uncoded_df = pd.read_csv(UNCODED_RESPONSES_PATH)
    uncoded_df["query"] = uncoded_df["role"] + ": " + uncoded_df["description"]
    uncoded_df["id"] = uncoded_df.index
    uncoded_input = VectorStoreSearchInput.from_data(uncoded_df)
    return uncoded_input


def load_vectoriser():
    # We use a local model as mybinder cant download reliably from huggingface or other model repositories.
    # Most applications of ClassifAI would just specify a model_name and it will automatically download the files.
    search_pattern = "../data/fastembed_cache/models--*--*/snapshots/*"
    matching_dirs = glob.glob(search_pattern)

    if matching_dirs:
        model_path = matching_dirs[0]
    else:
        raise RuntimeError(
            "The pre-cached FastEmbed model directory could not be located. "
            "Please ensure the environment's build step (postBuild) completed successfully "
            "or locate/download the model weights manually. If this issue persists in the "
            "live demo environment, please report it to the ClassifAI team via GitHub."
        )

    # Build vector store
    vectoriser = FastEmbedVectoriser(
        model_name="BAAI/bge-small-en-v1.5",
        specific_model_path=model_path,
    )
    return vectoriser


def prepare_knowledgebase():
    # Prepare knowledgebase - VectorStore expects 'label' and 'text' columns
    coded_df = pd.read_csv(DATASET_PATH)
    coded_df["label"] = coded_df["soc_code"]
    coded_df["text"] = coded_df["role"] + ": " + coded_df["description"]
    coded_df.to_csv(KNOWLEDGEBASE_PATH, index=False)


def make_query_input(input_list: list[str]):
    query_input = VectorStoreSearchInput({
        "id": list(range(len(input_list))),
        "query": input_list,
    })
    return query_input
