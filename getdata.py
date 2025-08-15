import pandas as pd
from datasets import load_dataset

def download_exorde_sample(sample_size: int = 50000, output_path: str = "exorde_raw_sample.csv") -> pd.DataFrame | None:
    print(f"Downloading {sample_size} unprocessed rows from Exorde dataset...")

    try:
        dataset = load_dataset(
            "Exorde/exorde-social-media-december-2024-week1",
            streaming=True,
            split='train'
        )

        sample_rows = []
        for i, row in enumerate(dataset):
            if i >= sample_size:
                break
            sample_rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"Downloaded {i + 1} rows...")

        sample_df = pd.DataFrame(sample_rows)
        sample_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully downloaded {len(sample_df)} rows")
        print(f"Sample saved to: {output_path}\n")
        print("Dataset columns:", sample_df.columns.tolist())
        print("First 5 rows:\n", sample_df.head())

        return sample_df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

# Download the sample
download_exorde_sample()
