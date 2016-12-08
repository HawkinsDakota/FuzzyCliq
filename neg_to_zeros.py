import pandas

def neg_to_zeros(file_name, sep = ',', header = 0, index_col = 0):
    data_frame = pandas.read_csv(file_name,
      sep = sep,
      header = header,
      index_col = index_col)

    negative_values = data_frame < 0
    data_frame[negative_values] = 0
    new_file = "no_neg_" + file_name
    data_frame.to_csv(new_file, header = True, index = True)
