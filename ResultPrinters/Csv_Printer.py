import ResultPrinters.Printer as Printer
import pandas as pd


def is_true(string):
    return string.lower() in ["true", "t", "1", "positive", "not negative"]


class CsvPrinter(Printer):
    def __init__(self, file_path, df):
        self.file_path = file_path
        self.df = df

    def print(self, options):
        result = pd.DataFrame(data=self.df)
        result.to_csv(path_or_buf=self.file_path + options["file_name"] + '.csv',
                      index=is_true(options["index"]), header=is_true(options["header"]))
