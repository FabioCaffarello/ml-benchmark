"""Hello unit test module."""

from downloader.hello import hello


def test_hello():
    """Test the hello function."""
    assert hello() == "Hello datasets-cifar10-dataset"
