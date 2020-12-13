import urllib.request
from tqdm import tqdm

url = "http://bitly.ws/9HPZ"
file_path = "lucene-index-cord19.tar.gz"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Updates the tqdm progress bar for every byte of data that is downloaded.

            Parameters:
                    b (list): initial value for the bytes.
                    bsize (int): byte size for updating.
                    tsize (int): total size of the download in bytes.

            Returns:
                    none : No return type since the function doesn't return any value.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Downloads the data from the url and places the data in the output_path

        Parameters:
                url (list): list of tokens.
                output_path (int): starting index position.

        Returns:
                none : No return type since the function doesn't return any value.
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

download_url(url, file_path)
