"""Utils to help keep the DEMO notebook clean and focussed on the user experiance."""

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
    """Load uncoded responses from a CSV into `VectorStoreSearchInput` format.

    The `VectorStoreSearchInput` requires `query` and `id` columns to be
    valid.

    Returns:
        VectorStoreSearchInput: The uncoded responses with generated queries
            and row IDs.
    """
    uncoded_df = pd.read_csv(UNCODED_RESPONSES_PATH)
    uncoded_df["query"] = uncoded_df["role"] + ": " + uncoded_df["description"]
    uncoded_df["id"] = uncoded_df.index
    uncoded_input = VectorStoreSearchInput.from_data(uncoded_df)
    return uncoded_input


def load_vectoriser():
    """Load a predownloaded `FastEmbedVectoriser`.

    A local model is used because MyBinder cannot reliably download from
    Hugging Face or other model repositories.

    Returns:
        FastEmbedVectoriser: A vectoriser configured with the cached model.

    Raises:
        RuntimeError: If the pre-cached FastEmbed model directory cannot be
            located.
    """
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

    # Most applications of ClassifAI would just specify a model_name and it will automatically download the files.
    vectoriser = FastEmbedVectoriser(
        model_name="BAAI/bge-small-en-v1.5",
        specific_model_path=model_path,
    )
    return vectoriser


def prepare_knowledgebase():
    """Load and save the mock dataset as a knowledgebase CSV.

    The `VectorStore` requires `label` and `text` columns to be valid.
    """
    coded_df = pd.read_csv(DATASET_PATH)
    coded_df["label"] = coded_df["soc_code"]
    coded_df["text"] = coded_df["role"] + ": " + coded_df["description"]
    coded_df.to_csv(KNOWLEDGEBASE_PATH, index=False)


def make_query_input(input_list: list[str]):
    """Create vector store search input from a list of queries.

    Args:
        input_list: The queries to convert into search input records.

    Returns:
        VectorStoreSearchInput: Search input containing a generated ID for
            each query.
    """
    query_input = VectorStoreSearchInput({
        "id": list(range(len(input_list))),
        "query": input_list,
    })
    return query_input
