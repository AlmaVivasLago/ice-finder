def read_txt_to_list(fpath, cast=None):
  with open(fpath, 'r') as fid:
    l = []
    for line in fid:
      if line.strip():
        l.append(line if cast is None else cast(line))

    return l