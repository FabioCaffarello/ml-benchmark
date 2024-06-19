import unittest
from unittest.mock import patch
import numpy as np
from downloader.download import download_and_preprocess, _download_cifar10, _preprocess_cifar10


class TestCifar10Downloader(unittest.TestCase):

    @patch('downloader.download.cifar10.load_data')
    def test_download_cifar10(self, mock_load_data):
        # Mock the cifar10.load_data function
        mock_load_data.return_value = (np.random.rand(50000, 32, 32, 3), np.random.randint(10, size=(50000, 1))), \
                                      (np.random.rand(10000, 32, 32, 3), np.random.randint(10, size=(10000, 1)))

        x_train, y_train, x_test, y_test = _download_cifar10()

        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_test.shape, (10000, 1))

    @patch('downloader.download.np.savez_compressed')
    def test_preprocess_cifar10(self, mock_savez_compressed):
        x_train = np.random.rand(50000, 32, 32, 3)
        y_train = np.random.randint(10, size=(50000, 1))
        x_test = np.random.rand(10000, 32, 32, 3)
        y_test = np.random.randint(10, size=(10000, 1))

        _preprocess_cifar10(x_train, y_train, x_test, y_test)

        self.assertTrue(mock_savez_compressed.called)
        call_args_list = mock_savez_compressed.call_args_list
        expected_files = ['data/cifar10_train.npz', 'data/cifar10_test.npz']
        actual_files = [args[0] for args, _ in call_args_list]

        self.assertCountEqual(actual_files, expected_files)

    @patch('downloader.download._download_cifar10')
    @patch('downloader.download._preprocess_cifar10')
    def test_download_and_preprocess(self, mock_preprocess, mock_download):
        mock_download.return_value = (
            np.random.rand(50000, 32, 32, 3), np.random.randint(10, size=(50000, 1)),
            np.random.rand(10000, 32, 32, 3), np.random.randint(10, size=(10000, 1))
        )

        download_and_preprocess()

        self.assertTrue(mock_download.called)
        self.assertTrue(mock_preprocess.called)


if __name__ == '__main__':
    unittest.main()
