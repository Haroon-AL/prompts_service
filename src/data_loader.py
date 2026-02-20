import pandas as pd
import json
from src.vector_store import VectorStore
from src.logger import setup_logging

logger = setup_logging()


def load_and_index_data(
    attributes_csv: str, mapper_csv: str, prompt_csv: str, vector_store: VectorStore
):
    logger.info("Loading CSVs...")
    try:
        df_attrs = pd.read_csv(attributes_csv)
        df_mapper = pd.read_csv(mapper_csv)
        df_prompt = pd.read_csv(prompt_csv)
    except FileNotFoundError as e:
        logger.error(f"Could not find CSV file: {e}")
        return

    logger.info("Merging data...")
    df_prompt_uniq = df_prompt.drop_duplicates(subset=["llm_mapper_id"])
    df_merged = df_mapper.merge(df_prompt_uniq, on="llm_mapper_id", how="inner")
    df_attrs_uniq = df_attrs.drop_duplicates(subset=["attribute_id"])
    df_final = df_merged.merge(df_attrs_uniq, on="attribute_id", how="inner")
    df_final = df_final.drop_duplicates(subset=["attribute_name"])

    ids = []
    texts = []
    metadatas = []

    for _, row in df_final.iterrows():
        attr_name = row["attribute_name"]

        if pd.isna(attr_name):
            continue

        attr_name = str(attr_name).strip()

        args_str = row["arguments"]
        prompt_text = ""

        if pd.notna(args_str):
            try:
                args = json.loads(args_str)
                prompt_text = args.get("prompt", "")
                system_role = args.get("system_role", "")
            except json.JSONDecodeError:
                pass

        if prompt_text:
            ids.append(f"attr_{attr_name.lower().replace(' ', '_')}")
            texts.append(attr_name)
            metadatas.append(
                {
                    "prompt": prompt_text,
                    "system_role": system_role,
                    "original_id": str(row["attribute_id"]),
                }
            )

    logger.info(f"Found {len(texts)} attributes with prompts to index.")

    vector_store.add_texts(ids=ids, texts=texts, metadatas=metadatas)
    logger.info("Indexing complete.")
