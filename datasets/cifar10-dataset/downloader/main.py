from downloader.download import download_and_preprocess
import logging

logger = logging.getLogger(__name__)


def main():
    logger.info("Downloading and preprocessing CIFAR-10 dataset...")
    download_and_preprocess()
    logger.info("CIFAR-10 dataset downloaded and preprocessed")


if __name__ == "__main__":
    main()
