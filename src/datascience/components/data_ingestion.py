from src.datascience import log
from src.datascience.constants import paths
from src.datascience.entity.config_entity import DataIngestionConfig
from src.datascience.utils.io import write_jsonl
from src.datascience.utils.runtime import retry


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @retry(times=3, base_delay=3.0)
    def _load_hf(self):
        from datasets import load_dataset

        return load_dataset(self.config.dataset_name, split="train")

    def ingest(self):
        """Produce artifacts/data_ingestion/data.jsonl as {text,label} rows.

        Default: pull a small HF dataset sample. To use your own data, set
        `ingestion.dataset_name: null` and drop your JSONL at that path.
        """
        if paths.RAW_DATA.exists() and not self.config.dataset_name:
            log.info(f"data already present, skipping ingest: {paths.RAW_DATA}")
            return
        if not self.config.dataset_name:
            raise FileNotFoundError(
                f"no ingestion.dataset_name set and no data at {paths.RAW_DATA} — provide one"
            )

        log.info(f"loading HF dataset '{self.config.dataset_name}' (train split)")
        ds = self._load_hf()

        df = ds.to_pandas()
        label_feature = ds.features.get("label")
        names = getattr(label_feature, "names", None)
        if names:  # map int ClassLabel -> readable name
            df["label"] = df["label"].map(lambda i: names[i])

        df = df[["text", "label"]].dropna()
        if self.config.sample_size:
            n = min(int(self.config.sample_size), len(df))
            df = df.sample(n=n, random_state=42).reset_index(drop=True)

        rows = df.to_dict(orient="records")
        write_jsonl(paths.RAW_DATA, rows)
        log.info(f"saved {len(rows)} rows -> {paths.RAW_DATA} (classes: {sorted(df['label'].unique())})")
