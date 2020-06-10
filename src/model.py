import pandas as pd
import os


class MlModel:

    def __init__(self):
        self.dataset = pd.DataFrame()

    def read_dataset(self, address):
        """Read a dataset into a pandas dataframe from a file address.

        Args:
            address (str): file address.

        Returns:
            int: The return value. 1 for Invalid file extension
                                   2 for reading exception
        """
        filename, file_extension = os.path.splitext(address)
        try:
            if file_extension == '.csv':
                self.dataset = pd.read_csv(address)
                return 0

            elif file_extension == '.xls' or file_extension == '.xlsx':
                self.dataset = pd.read_excel(address)
                return 0
            else:
                return 1  # Invalid file extension
        except:
            return 2  # Exception
