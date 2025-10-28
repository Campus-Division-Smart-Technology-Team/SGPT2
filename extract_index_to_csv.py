# File: export_index.py
import os
import csv
import logging
from typing import Optional, List
from dotenv import load_dotenv
from pinecone import Pinecone

# ---------------- Env & constants ----------------
load_dotenv()

# ---- logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

# Initialise Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def get_all_vector_ids(index, namespace: Optional[str] = None) -> List[str]:
    """Get all vector IDs from the index using the list method."""
    all_ids = []

    try:
        # Use list() method to iterate through all vector IDs
        for ids_batch in index.list(namespace=namespace):
            if ids_batch and len(ids_batch) > 0:
                all_ids.extend(ids_batch)
            else:
                break  # No more IDs to fetch

        logging.info("Found %d vector IDs", len(all_ids))
        return all_ids

    except Exception as e:  # pylint: disable=broad-except
        logging.error("Error listing vector IDs: %s", e)
        return []


def export_index_to_csv(idx_name: str, output_path: str, namespace: Optional[str] = None) -> None:
    """Export Pinecone index to a CSV file."""
    try:
        index = pc.Index(idx_name)

        # Get basic index stats
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count if stats else 0
        logging.info("Total vectors in index '%s': %d",
                     idx_name, total_vectors)

        # Get all vector IDs
        logging.info("Fetching all vector IDs...")
        all_ids = get_all_vector_ids(index, namespace)

        if not all_ids:
            logging.warning("No vector IDs found in the index.")
            return

        # Create a CSV file to write the data
        with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header
            csv_writer.writerow([
                "ID", "Property alternative names", "Property campus", "Property code", "Property names", "Property postcode", "UsrFRACondensedPropertyName",
                "building_aliases", "building_name", "canonical_building_name", "chunk", "document_type", "ext", "key",
                "last_modified", "original_file", "size", "source", "text"
            ])

            # Process vectors in batches
            batch_size = 100
            total_processed = 0
            # Calculate total number of batches
            total_batches = (len(all_ids) + batch_size - 1) // batch_size

            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                batch_num = i // batch_size + 1
                start_id = i + 1
                end_id = min(i + batch_size, len(all_ids))

                try:
                    # Improved logging with progress indicator
                    logging.info(
                        "Processing batch %d/%d: Fetching vectors %d-%d (%d vectors, %d/%d completed)",
                        batch_num, total_batches, start_id, end_id, len(
                            batch_ids), total_processed, len(all_ids)
                    )

                    response = index.fetch(ids=batch_ids, namespace=namespace)

                    if not response or not response.vectors:
                        logging.warning(
                            "No vectors returned for batch %d (IDs %d-%d)",
                            batch_num, start_id, end_id
                        )
                        continue

                    # Process each vector in the batch
                    batch_count = 0
                    for vector_id, vector_data in response.vectors.items():
                        metadata = vector_data.metadata or {}

                        # Extract the required fields from metadata
                        row_data = [
                            vector_id,
                            metadata.get("Property alternative names", ""),
                            metadata.get("Property campus", ""),
                            metadata.get("Property code", ""),
                            metadata.get("Property names", ""),
                            metadata.get("Property postcode", ""),
                            metadata.get("UsrFRACondensedPropertyName", ""),
                            metadata.get("building_aliases", ""),
                            metadata.get("building_name", ""),
                            metadata.get("canonical_building_name", ""),
                            metadata.get("chunk", ""),
                            metadata.get("document_type", ""),
                            metadata.get("ext", ""),
                            metadata.get("key", ""),
                            metadata.get("last_modified", ""),
                            metadata.get("original_file", ""),
                            metadata.get("size", ""),
                            metadata.get("source", ""),
                            metadata.get("text", "")
                        ]

                        # Write the row to CSV
                        csv_writer.writerow(row_data)
                        total_processed += 1
                        batch_count += 1

                    # Log batch completion
                    logging.info(
                        "Batch %d/%d completed: Wrote %d vectors to CSV",
                        batch_num, total_batches, batch_count
                    )

                except Exception as e:  # pylint: disable=broad-except
                    logging.error(
                        "Error processing batch %d (IDs %d-%d): %s",
                        batch_num, start_id, end_id, e
                    )
                    continue

            # Final summary
            logging.info("=" * 60)
            logging.info("Export completed successfully!")
            logging.info("Total vectors processed: %d/%d",
                         total_processed, len(all_ids))
            logging.info("Output file: %s", output_path)
            logging.info("=" * 60)

    except Exception as e:
        logging.error("Error exporting index '%s': %s", idx_name, e)
        raise


if __name__ == "__main__":
    INDEX_NAME = "operational-docs"  # Pinecone Index name
    # Path to save the CSV
    OUTPUT_FILE = r"C:\Users\rd23091\OneDrive - University of Bristol\Documents\Python exports\audit.csv"

    try:
        export_index_to_csv(INDEX_NAME, OUTPUT_FILE)
    except Exception as e:  # pylint: disable=broad-except
        logging.error("Export failed: %s", e)
        print(f"Export failed: {e}")
# ---------------- End of export_index_to_csv.py ----------------
